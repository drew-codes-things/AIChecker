import errno
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .engine import EngineLink, ControllerStatus

log = logging.getLogger("aichecker.gate")

OWN_OWNER = "ai-checker"
OWN_ACTIVITY_TOLERANCE_SECONDS = 2.0
FREE_MEMORY_POLL_SECONDS = 1.0
FREE_MEMORY_WAIT_SECONDS = 60.0


class GateBusy(RuntimeError):
    def __init__(self, reason: str, retry_after: int = 15):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


@dataclass(frozen=True)
class GateSettings:
    lock_file: Path
    quiet_seconds: float
    wait_seconds: float
    poll_seconds: float
    min_free_mb: int


def engine_is_quiet(status: ControllerStatus | None, seconds_since_own_event: float | None, quiet_seconds: float) -> bool:
    if status is None:
        return True
    if status.in_flight > 0:
        return False
    last_activity_was_ours = (seconds_since_own_event is not None
                              and status.seconds_since_used >= seconds_since_own_event - OWN_ACTIVITY_TOLERANCE_SECONDS)
    return last_activity_was_ours or status.seconds_since_used >= quiet_seconds


def process_is_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def parse_pid(holder: str) -> int | None:
    for part in holder.split():
        if part.startswith("pid="):
            digits = part[4:]
            return int(digits) if digits.isdigit() else None
    return None


class GpuGate:
    def __init__(self, settings: GateSettings, link: EngineLink, free_memory_mb: Callable[[], int],
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 pid: int | None = None, pid_alive: Callable[[int], bool] = process_is_alive,
                 wall_clock: Callable[[], float] = time.time):
        self.settings = settings
        self.link = link
        self.free_memory_mb = free_memory_mb
        self.clock = clock
        self.sleep = sleep
        self.pid = os.getpid() if pid is None else pid
        self.pid_alive = pid_alive
        self.wall_clock = wall_clock
        self.request_id = f"ai-checker-{self.pid}"
        self.holding = False
        self.last_own_event: float | None = None
        self.restore_models: list[str] = []

    def _lock_text(self) -> str:
        return f"{OWN_OWNER} {int(self.wall_clock())} pid={self.pid}\n"

    def _read_holder(self) -> str | None:
        try:
            return self.settings.lock_file.read_text().strip()
        except FileNotFoundError:
            return None

    def _holder_is_dead_checker(self, holder: str) -> bool:
        if not holder.startswith(OWN_OWNER + " "):
            return False
        pid = parse_pid(holder)
        return pid is not None and pid != self.pid and not self.pid_alive(pid)

    def _try_create_lock(self) -> bool:
        try:
            descriptor = os.open(self.settings.lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            return False
        except OSError as error:
            if error.errno == errno.ENOENT:
                raise GateBusy(f"the lock folder for {self.settings.lock_file} does not exist") from error
            raise
        with os.fdopen(descriptor, "w") as handle:
            handle.write(self._lock_text())
        return True

    def _take_lock(self) -> str | None:
        if self._try_create_lock():
            return None
        holder = self._read_holder()
        if holder is not None and self._holder_is_dead_checker(holder):
            log.warning("removing the lock left by a stopped checker process: %s", holder)
            self.settings.lock_file.unlink(missing_ok=True)
            if self._try_create_lock():
                return None
            holder = self._read_holder()
        return holder or "a lock was just taken"

    def seconds_since_own_event(self) -> float | None:
        return None if self.last_own_event is None else self.clock() - self.last_own_event

    def note_own_activity(self) -> None:
        self.last_own_event = self.clock()

    def _wait_reason(self) -> str | None:
        status = self.link.controller_status()
        if not engine_is_quiet(status, self.seconds_since_own_event(), self.settings.quiet_seconds):
            return "the model engine is answering or was used a moment ago"
        holder = self._take_lock()
        if holder is not None:
            return f"another GPU job holds the lock ({holder.split()[0] if holder.split() else 'unknown'})"
        return None

    def acquire(self) -> None:
        if self.holding:
            return
        deadline = self.clock() + self.settings.wait_seconds
        while True:
            reason = self._wait_reason()
            if reason is None:
                break
            if self.clock() >= deadline:
                raise GateBusy(reason)
            self.sleep(self.settings.poll_seconds)
        try:
            self._claim_gpu()
        except BaseException:
            self._drop_lock()
            self.link.controller_done(self.request_id)
            self.note_own_activity()
            raise
        self.holding = True

    def _claim_gpu(self) -> None:
        self.link.controller_busy(self.request_id)
        self.restore_models = self.link.unload_engine_models()
        if self._others_in_flight() > 0:
            raise GateBusy("the model engine started answering while the GPU was being taken")
        self._wait_for_free_memory()

    def _others_in_flight(self) -> int:
        status = self.link.controller_status()
        return 0 if status is None else max(status.in_flight - 1, 0)

    def _wait_for_free_memory(self) -> None:
        deadline = self.clock() + FREE_MEMORY_WAIT_SECONDS
        while self.free_memory_mb() < self.settings.min_free_mb:
            if self.clock() >= deadline:
                raise GateBusy(f"the GPU did not free {self.settings.min_free_mb} MB in {int(FREE_MEMORY_WAIT_SECONDS)} s")
            self.sleep(FREE_MEMORY_POLL_SECONDS)

    def engine_is_active(self) -> bool:
        if self._others_in_flight() > 0:
            return True
        loaded = self.link.engine_models()
        return bool(loaded)

    def unload_engine_models_left_partly_on_cpu(self) -> list[str]:
        if self.holding:
            return []
        status = self.link.controller_status()
        if not engine_is_quiet(status, self.seconds_since_own_event(), self.settings.quiet_seconds):
            raise GateBusy("the model engine is answering or was used a moment ago")
        partial = self.link.engine_partial_models()
        if partial:
            log.warning("unloading %s: it was loaded while the checker held the GPU and sits partly on the CPU", partial)
            self.link.unload_named_engine_models(partial)
        return partial

    def _drop_lock(self) -> None:
        holder = self._read_holder()
        if holder is not None and holder.startswith(OWN_OWNER + " ") and parse_pid(holder) == self.pid:
            self.settings.lock_file.unlink(missing_ok=True)

    def release(self) -> list[str]:
        if not self.holding:
            return []
        self.holding = False
        self._drop_lock()
        self.link.controller_done(self.request_id)
        self.note_own_activity()
        restore, self.restore_models = self.restore_models, []
        return restore
