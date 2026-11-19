import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

from .gate import GateBusy, GpuGate

log = logging.getLogger("aichecker.session")

WATCH_INTERVAL_SECONDS = 0.5
ACTIVITY_CHECK_INTERVAL_SECONDS = 0.5


class WorkerBusy(RuntimeError):
    pass


class Scorer(Protocol):
    loaded: bool

    def load(self) -> None: ...

    def unload(self) -> None: ...


class GpuSession:
    def __init__(self, gate: GpuGate, scorer: Scorer, idle_unload_seconds: float, job_wait_seconds: float,
                 clock: Callable[[], float] = time.monotonic, restore: Callable[[list[str]], None] | None = None):
        self.gate = gate
        self.scorer = scorer
        self.idle_unload_seconds = idle_unload_seconds
        self.job_wait_seconds = job_wait_seconds
        self.clock = clock
        self.restore = restore or (lambda models: None)
        self.job_lock = threading.Lock()
        self.idle_since: float | None = None
        self.engine_check_due = False
        self.engine_waiting = False
        self.engine_waiting = False
        self.last_activity_check = 0.0
        self.stopping = threading.Event()
        self.watcher: threading.Thread | None = None

    def _enter_job(self) -> None:
        if not self.job_lock.acquire(timeout=self.job_wait_seconds):
            raise WorkerBusy("another request is using the checker; try again in a moment")

    def _yield_if_engine_asked(self) -> None:
        if not self.engine_waiting:
            return
        self.engine_waiting = False
        if self.gate.holding:
            log.info("the model engine asked for the GPU while a check was running: giving it back before the next one")
            self._give_back_gpu()

    def _give_back_gpu(self) -> None:
        self.scorer.unload()
        models = self.gate.release()
        self.idle_since = None
        self.engine_check_due = True
        if models:
            threading.Thread(target=self.restore, args=(models,), daemon=True, name="restore-engine-models").start()

    def should_yield(self) -> bool:
        now = self.clock()
        if now - self.last_activity_check < ACTIVITY_CHECK_INTERVAL_SECONDS:
            return False
        self.last_activity_check = now
        return self.gate.engine_is_active()

    @contextmanager
    def scoring(self) -> Iterator[Scorer]:
        self._enter_job()
        try:
            self.idle_since = None
            self._yield_if_engine_asked()
            self.gate.acquire()
            try:
                self.scorer.load()
            except BaseException:
                self._give_back_gpu()
                raise
            self.last_activity_check = self.clock()
            yield self.scorer
        except GateBusy:
            raise
        except BaseException:
            if self.gate.holding:
                self._give_back_gpu()
            raise
        else:
            self.idle_since = self.clock()
        finally:
            self.job_lock.release()

    @contextmanager
    def engine_work(self) -> Iterator[None]:
        self._enter_job()
        try:
            if self.gate.holding:
                self._give_back_gpu()
            yield
        finally:
            self.gate.note_own_activity()
            self.job_lock.release()

    def release(self) -> None:
        with self.job_lock:
            if self.gate.holding:
                self._give_back_gpu()

    def settle(self) -> bool:
        if not self.engine_check_due:
            return True
        if not self.job_lock.acquire(blocking=False):
            return False
        try:
            if self.gate.holding:
                return False
            try:
                self.gate.unload_engine_models_left_partly_on_cpu()
            except GateBusy:
                return False
            self.engine_check_due = False
            return True
        finally:
            self.job_lock.release()

    def watch_once(self) -> None:
        self.settle()
        if self.gate.holding and self.gate.engine_is_active():
            self.engine_waiting = True
        if not self.job_lock.acquire(blocking=False):
            return
        try:
            if not self.gate.holding:
                return
            idle_for = self.clock() - (self.idle_since if self.idle_since is not None else self.clock())
            if self.engine_waiting or self.gate.engine_is_active():
                log.info("the model engine is active: giving the GPU back")
                self.engine_waiting = False
                self._give_back_gpu()
            elif idle_for >= self.idle_unload_seconds:
                log.info("idle for %.0f s: unloading the pair and giving the GPU back", idle_for)
                self._give_back_gpu()
        finally:
            self.job_lock.release()

    def start_watcher(self) -> None:
        self.watcher = threading.Thread(target=self._watch_loop, daemon=True, name="gpu-watch")
        self.watcher.start()

    def _watch_loop(self) -> None:
        while not self.stopping.wait(WATCH_INTERVAL_SECONDS):
            try:
                self.watch_once()
            except Exception:
                log.exception("the idle watcher failed")

    def stop(self) -> None:
        self.stopping.set()
        self.release()
