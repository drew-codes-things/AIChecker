import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .humanise import Cancelled
from .upstream import UpstreamError
from .voice import VoiceError

JOB_TTL_SECONDS = 15 * 60
MAX_JOBS = 4
DONE = "done"
RUNNING = "running"
FAILED = "failed"
STOPPED = "stopped"


class TooManyJobs(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    created: float
    status: str = RUNNING
    step: int = 0
    steps: int = 1
    message: str = "Starting"
    result: dict | None = None
    error: str | None = None
    busy: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)


class JobRunner:
    def __init__(self, clock: Callable[[], float] = time.time, run_in_thread: bool = True):
        self.clock = clock
        self.run_in_thread = run_in_thread
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def _forget_expired(self) -> None:
        cutoff = self.clock() - JOB_TTL_SECONDS
        for job_id in [key for key, job in self.jobs.items() if job.created < cutoff]:
            del self.jobs[job_id]

    def start(self, work: Callable[[Callable[[int, int, str], None], Callable[[], bool]], dict]) -> str:
        with self.lock:
            self._forget_expired()
            if sum(1 for job in self.jobs.values() if job.status == RUNNING) >= MAX_JOBS:
                raise TooManyJobs("too many rewrites are running; wait for one to finish")
            job = Job(secrets.token_urlsafe(12), self.clock())
            self.jobs[job.id] = job
        if self.run_in_thread:
            threading.Thread(target=self._run, args=(job, work), daemon=True, name=f"job-{job.id}").start()
        else:
            self._run(job, work)
        return job.id

    def _run(self, job: Job, work) -> None:
        def progress(step: int, steps: int, message: str) -> None:
            job.step, job.steps, job.message = step, steps, message

        try:
            job.result = work(progress, job.cancel.is_set)
            job.status = DONE
        except Cancelled:
            job.status = STOPPED
        except (UpstreamError, VoiceError) as error:
            job.status, job.error = FAILED, str(error)
            job.busy = bool(getattr(error, "busy", False))
        except Exception:
            job.status, job.error = FAILED, "the rewrite failed unexpectedly; see the app log"
            raise

    def view(self, job_id: str) -> dict | None:
        with self.lock:
            self._forget_expired()
            job = self.jobs.get(job_id)
            if job is None:
                return None
            body = {"id": job.id, "status": job.status, "step": job.step, "steps": job.steps, "message": job.message,
                    "error": job.error, "busy": job.busy}
            if job.status == DONE:
                body["result"] = job.result
                del self.jobs[job_id]
            elif job.status in (FAILED, STOPPED):
                del self.jobs[job_id]
            return body

    def stop(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            job.cancel.set()
            return True
