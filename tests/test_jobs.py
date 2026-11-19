import threading

import pytest

from aichecker.web.humanise import Cancelled
from aichecker.web.jobs import JOB_TTL_SECONDS, MAX_JOBS, JobRunner, TooManyJobs
from aichecker.web.upstream import UpstreamError
from aichecker.web.voice import VoiceError


def test_a_finished_job_returns_its_result_once_and_is_then_forgotten():
    runner = JobRunner(run_in_thread=False)
    job_id = runner.start(lambda progress, cancelled: {"answer": 42})
    first = runner.view(job_id)
    assert first["status"] == "done" and first["result"] == {"answer": 42}
    assert runner.view(job_id) is None


def test_progress_is_reported_while_running():
    started = threading.Event()
    release = threading.Event()

    def blocking(progress, cancelled):
        progress(2, 5, "Rewriting part 2")
        started.set()
        release.wait(2)
        return {}

    runner = JobRunner()
    job_id = runner.start(blocking)
    started.wait(2)
    view = runner.view(job_id)
    assert view["status"] == "running" and (view["step"], view["steps"], view["message"]) == (2, 5, "Rewriting part 2")
    release.set()


def test_failures_become_a_message_and_a_busy_flag_when_the_worker_was_busy():
    runner = JobRunner(run_in_thread=False)
    failed = runner.start(lambda progress, cancelled: (_ for _ in ()).throw(VoiceError("add writing first")))
    assert runner.view(failed)["error"] == "add writing first"
    busy = runner.start(lambda progress, cancelled: (_ for _ in ()).throw(UpstreamError("busy", 503, True)))
    view = runner.view(busy)
    assert view["status"] == "failed" and view["busy"] is True


def test_cancelling_stops_the_job_at_its_next_check():
    runner = JobRunner()
    ready = threading.Event()
    go = threading.Event()

    def work(progress, cancelled):
        ready.set()
        go.wait(2)
        if cancelled():
            raise Cancelled("stopped")
        return {"x": 1}

    job_id = runner.start(work)
    ready.wait(2)
    assert runner.stop(job_id) is True
    go.set()
    for _ in range(100):
        view = runner.view(job_id)
        if view is None or view["status"] != "running":
            break
        threading.Event().wait(0.02)
    assert view["status"] == "stopped"
    assert runner.stop("unknown") is False


def test_old_jobs_expire_and_too_many_running_jobs_are_refused():
    now = [1000.0]
    runner = JobRunner(clock=lambda: now[0])
    gate = threading.Event()
    ids = [runner.start(lambda progress, cancelled: gate.wait(5) or {}) for _ in range(MAX_JOBS)]
    with pytest.raises(TooManyJobs):
        runner.start(lambda progress, cancelled: {})
    now[0] += JOB_TTL_SECONDS + 1
    assert runner.view(ids[0]) is None
    gate.set()
