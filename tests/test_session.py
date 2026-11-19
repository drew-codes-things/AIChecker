import threading
import time

import pytest

from aichecker.worker.engine import ControllerStatus
from aichecker.worker.gate import GateBusy, GateSettings, GpuGate
from aichecker.worker.session import GpuSession, WorkerBusy
from fakes import ENGINE_ANSWERING, QUIET, FakeClock, FakeLink


class FakeScorer:
    def __init__(self):
        self.loaded = False
        self.fail_on_load = False

    def load(self):
        if self.fail_on_load:
            raise RuntimeError("no memory")
        self.loaded = True

    def unload(self):
        self.loaded = False


def make_session(tmp_path, link, idle=45.0, job_wait=0.2):
    clock = FakeClock()
    gate = GpuGate(GateSettings(tmp_path / "gpu-lock", 20.0, 5.0, 1.0, 9000), link, lambda: 12000, clock=clock,
                   sleep=clock.sleep, pid=4242, pid_alive=lambda pid: False, wall_clock=lambda: 1.0)
    restored = []
    scorer = FakeScorer()
    session = GpuSession(gate, scorer, idle, job_wait, clock=clock, restore=restored.extend)
    return session, scorer, gate, clock, restored


def test_scoring_takes_the_gate_loads_the_pair_and_keeps_both_until_idle(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, clock, _ = make_session(tmp_path, link)
    with session.scoring():
        assert gate.holding and scorer.loaded
    assert gate.holding and scorer.loaded
    clock.now += 30
    session.watch_once()
    assert gate.holding and scorer.loaded


def test_after_the_idle_time_the_pair_is_unloaded_and_the_gate_released(tmp_path):
    link = FakeLink(QUIET, loaded=["qwen3:14b"])
    session, scorer, gate, clock, restored = make_session(tmp_path, link)
    with session.scoring():
        pass
    clock.now += 46
    session.watch_once()
    assert not gate.holding and not scorer.loaded
    assert not (tmp_path / "gpu-lock").exists()
    assert link.calls[-1][0] in ("done", "load")
    deadline = time.time() + 2
    while not restored and time.time() < deadline:
        time.sleep(0.01)
    assert restored == ["qwen3:14b"]


def test_engine_becoming_active_while_the_pair_is_idle_gives_the_gpu_back_at_once(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, clock, _ = make_session(tmp_path, link)
    with session.scoring():
        pass
    link.status = ControllerStatus(in_flight=2, seconds_since_used=0.0)
    clock.now += 1
    session.watch_once()
    assert not gate.holding and not scorer.loaded


def test_engine_loading_a_model_is_noticed_even_before_its_request_count_shows(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, clock, _ = make_session(tmp_path, link)
    with session.scoring():
        pass
    link.loaded = ["qwen3:14b"]
    session.watch_once()
    assert not gate.holding


def test_should_yield_is_rate_limited_and_reports_the_engine(tmp_path):
    link = FakeLink(QUIET)
    session, _, _, clock, _ = make_session(tmp_path, link)
    with session.scoring():
        assert session.should_yield() is False
        link.status = ControllerStatus(in_flight=2, seconds_since_used=0.0)
        assert session.should_yield() is False
        clock.now += 2
        assert session.should_yield() is True


def test_a_busy_gate_leaves_nothing_held(tmp_path):
    link = FakeLink(ENGINE_ANSWERING)
    session, scorer, gate, _, _ = make_session(tmp_path, link)
    with pytest.raises(GateBusy):
        with session.scoring():
            pytest.fail("must not run")
    assert not gate.holding and not scorer.loaded
    assert session.job_lock.acquire(blocking=False)
    session.job_lock.release()


def test_an_error_inside_the_job_gives_the_gpu_back_and_frees_the_lock(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, _, _ = make_session(tmp_path, link)
    with pytest.raises(ValueError):
        with session.scoring():
            raise ValueError("bad text")
    assert not gate.holding and not scorer.loaded
    assert not (tmp_path / "gpu-lock").exists()
    assert session.job_lock.acquire(blocking=False)


def test_a_failed_model_load_gives_the_gpu_back(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, _, _ = make_session(tmp_path, link)
    scorer.fail_on_load = True
    with pytest.raises(RuntimeError, match="no memory"):
        with session.scoring():
            pass
    assert not gate.holding
    assert not (tmp_path / "gpu-lock").exists()


def test_engine_work_gives_the_pair_back_first(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, _, _ = make_session(tmp_path, link)
    with session.scoring():
        pass
    with session.engine_work():
        assert not gate.holding and not scorer.loaded
    assert gate.last_own_event is not None


def test_a_second_request_waits_then_reports_busy(tmp_path):
    link = FakeLink(QUIET)
    session, _, _, _, _ = make_session(tmp_path, link, job_wait=0.1)
    started = threading.Event()
    release = threading.Event()

    def hold():
        with session.engine_work():
            started.set()
            release.wait(2)

    worker = threading.Thread(target=hold)
    worker.start()
    started.wait(2)
    with pytest.raises(WorkerBusy):
        with session.engine_work():
            pass
    release.set()
    worker.join()


def test_the_watcher_leaves_a_running_job_alone(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, clock, _ = make_session(tmp_path, link)
    with session.scoring():
        clock.now += 500
        session.watch_once()
        assert gate.holding and scorer.loaded


def test_stop_gives_the_gpu_back(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, _, _ = make_session(tmp_path, link)
    with session.scoring():
        pass
    session.stop()
    assert not gate.holding and not scorer.loaded


def test_after_giving_way_to_engine_a_partly_offloaded_model_is_unloaded_when_engine_is_quiet(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, clock, _ = make_session(tmp_path, link)
    with session.scoring():
        pass
    link.status = ENGINE_ANSWERING
    link.loaded = ["qwen3:14b"]
    link.partial = ["qwen3:14b"]
    session.watch_once()
    assert not gate.holding
    assert session.engine_check_due
    assert link.loaded == ["qwen3:14b"]
    assert session.settle() is False
    link.status = QUIET
    clock.now += 60
    assert session.settle() is True
    assert not session.engine_check_due
    assert link.loaded == []


def test_settle_has_nothing_to_do_when_the_gpu_was_never_taken(tmp_path):
    link = FakeLink(QUIET, loaded=["qwen3:14b"])
    link.partial = ["qwen3:14b"]
    session, *_ = make_session(tmp_path, link)
    assert session.settle() is True
    assert link.loaded == ["qwen3:14b"]


def test_engine_asking_while_a_check_holds_the_job_lock_frees_the_gpu_before_the_next_check(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, clock, _ = make_session(tmp_path, link)
    with session.scoring():
        link.status = ControllerStatus(in_flight=2, seconds_since_used=0.0)
        session.watch_once()
        assert gate.holding and scorer.loaded
        assert session.engine_waiting
    link.status = ENGINE_ANSWERING
    with pytest.raises(GateBusy):
        with session.scoring():
            pass
    assert not gate.holding and not scorer.loaded
    assert not session.engine_waiting


def test_without_engine_the_waiting_flag_stays_down_and_checks_keep_the_gpu(tmp_path):
    link = FakeLink(QUIET)
    session, scorer, gate, clock, _ = make_session(tmp_path, link)
    with session.scoring():
        session.watch_once()
        assert not session.engine_waiting
    with session.scoring():
        assert gate.holding and scorer.loaded
