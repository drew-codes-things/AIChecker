import pytest

from aichecker.worker.engine import ControllerStatus
from aichecker.worker.gate import GateBusy, GateSettings, GpuGate, engine_is_quiet, parse_pid
from fakes import ENGINE_ANSWERING, QUIET, FakeClock, FakeLink


def make_gate(tmp_path, link, free_mb=12000, clock=None, pid=4242, alive=lambda pid: False, **overrides):
    clock = clock or FakeClock()
    settings = GateSettings(tmp_path / "gpu-lock", overrides.get("quiet", 20.0), overrides.get("wait", 10.0), 1.0,
                            overrides.get("min_free", 9000))
    gate = GpuGate(settings, link, lambda: free_mb if isinstance(free_mb, int) else free_mb(), clock=clock,
                   sleep=clock.sleep, pid=pid, pid_alive=alive, wall_clock=lambda: 1795000000.0)
    return gate, clock


def test_quiet_rules_for_engine_activity():
    assert engine_is_quiet(None, None, 20)
    assert not engine_is_quiet(ControllerStatus(1, 999), None, 20)
    assert not engine_is_quiet(ControllerStatus(0, 5), None, 20)
    assert engine_is_quiet(ControllerStatus(0, 25), None, 20)


def test_activity_that_was_our_own_does_not_count_as_the_engine():
    assert engine_is_quiet(ControllerStatus(0, 5.0), 5.5, 20)
    assert engine_is_quiet(ControllerStatus(0, 5.0), 5.0, 20)
    assert not engine_is_quiet(ControllerStatus(0, 2.0), 30.0, 20)


def test_parse_pid_reads_the_marker_only():
    assert parse_pid("ai-checker 1795000000 pid=77") == 77
    assert parse_pid("gpu-owner 1795000000") is None
    assert parse_pid("ai-checker 1 pid=abc") is None


def test_acquire_takes_lock_tells_controller_unloads_engine_and_release_undoes_it(tmp_path):
    link = FakeLink(QUIET, loaded=["qwen3:14b"])
    gate, _ = make_gate(tmp_path, link)
    gate.acquire()
    assert gate.holding
    assert (tmp_path / "gpu-lock").read_text() == "ai-checker 1795000000 pid=4242\n"
    assert [call[0] for call in link.calls] == ["busy", "unload"]
    restored = gate.release()
    assert restored == ["qwen3:14b"]
    assert not (tmp_path / "gpu-lock").exists()
    assert link.calls[-1] == ("done", "ai-checker-4242")
    assert not gate.holding


def test_release_when_not_holding_does_nothing(tmp_path):
    link = FakeLink(QUIET)
    gate, _ = make_gate(tmp_path, link)
    assert gate.release() == []
    assert link.calls == []


def test_acquire_twice_is_harmless(tmp_path):
    link = FakeLink(QUIET)
    gate, _ = make_gate(tmp_path, link)
    gate.acquire()
    gate.acquire()
    assert [call[0] for call in link.calls].count("busy") == 1


def test_gate_refuses_while_engine_is_answering_and_never_takes_the_lock(tmp_path):
    link = FakeLink(ENGINE_ANSWERING)
    gate, _ = make_gate(tmp_path, link)
    with pytest.raises(GateBusy, match="engine"):
        gate.acquire()
    assert not (tmp_path / "gpu-lock").exists()
    assert link.calls == []


def test_gate_waits_for_engine_to_finish_then_proceeds(tmp_path):
    link = FakeLink(ENGINE_ANSWERING)
    clock = FakeClock()
    gate, _ = make_gate(tmp_path, link, clock=clock)
    original_sleep = clock.sleep

    def sleep_then_engine_finishes(seconds):
        original_sleep(seconds)
        if clock.now > 1003:
            link.status = QUIET

    gate.sleep = sleep_then_engine_finishes
    gate.acquire()
    assert gate.holding


def test_another_jobs_lock_is_respected_and_left_alone(tmp_path):
    (tmp_path / "gpu-lock").write_text("brain 1795000000")
    link = FakeLink(QUIET)
    gate, _ = make_gate(tmp_path, link)
    with pytest.raises(GateBusy, match="brain"):
        gate.acquire()
    assert (tmp_path / "gpu-lock").read_text() == "brain 1795000000"
    assert link.calls == []


def test_a_lock_left_by_a_dead_checker_is_replaced(tmp_path):
    (tmp_path / "gpu-lock").write_text("ai-checker 1700000000 pid=999")
    link = FakeLink(QUIET)
    gate, _ = make_gate(tmp_path, link, alive=lambda pid: False)
    gate.acquire()
    assert "pid=4242" in (tmp_path / "gpu-lock").read_text()


def test_a_lock_held_by_a_live_checker_process_is_respected(tmp_path):
    (tmp_path / "gpu-lock").write_text("ai-checker 1700000000 pid=999")
    link = FakeLink(QUIET)
    gate, _ = make_gate(tmp_path, link, alive=lambda pid: True)
    with pytest.raises(GateBusy):
        gate.acquire()
    assert "pid=999" in (tmp_path / "gpu-lock").read_text()


def test_release_never_removes_a_lock_that_is_not_ours(tmp_path):
    link = FakeLink(QUIET)
    gate, _ = make_gate(tmp_path, link)
    gate.acquire()
    (tmp_path / "gpu-lock").write_text("brain 1795000001")
    gate.release()
    assert (tmp_path / "gpu-lock").read_text() == "brain 1795000001"


def test_when_the_gpu_does_not_free_up_everything_is_handed_back(tmp_path):
    link = FakeLink(QUIET, loaded=["qwen3:14b"])
    gate, _ = make_gate(tmp_path, link, free_mb=3000)
    with pytest.raises(GateBusy, match="did not free"):
        gate.acquire()
    assert not (tmp_path / "gpu-lock").exists()
    assert link.calls[-1] == ("done", "ai-checker-4242")
    assert not gate.holding


def test_engine_starting_while_the_gpu_is_taken_makes_the_gate_back_off(tmp_path):
    link = FakeLink(QUIET)
    original_unload = link.unload_engine_models

    def unload_then_engine_starts():
        link.status = ControllerStatus(in_flight=2, seconds_since_used=0.0)
        return original_unload()

    link.unload_engine_models = unload_then_engine_starts
    gate, _ = make_gate(tmp_path, link)
    with pytest.raises(GateBusy, match="engine"):
        gate.acquire()
    assert not (tmp_path / "gpu-lock").exists()


def test_engine_is_active_sees_other_requests_but_not_our_own_busy(tmp_path):
    link = FakeLink(QUIET)
    gate, _ = make_gate(tmp_path, link)
    gate.acquire()
    link.status = ControllerStatus(in_flight=1, seconds_since_used=0.0)
    assert not gate.engine_is_active()
    link.status = ControllerStatus(in_flight=2, seconds_since_used=0.0)
    assert gate.engine_is_active()
    link.status = QUIET
    link.loaded = ["qwen3:14b"]
    assert gate.engine_is_active()


def test_an_unreachable_controller_does_not_block_checks(tmp_path):
    link = FakeLink(QUIET)
    link.unreachable = True
    gate, _ = make_gate(tmp_path, link)
    gate.acquire()
    assert gate.holding


def test_a_model_left_partly_on_the_cpu_is_unloaded_once_engine_is_quiet(tmp_path):
    link = FakeLink(QUIET, loaded=["qwen3:14b", "nomic-embed-text"])
    link.partial = ["qwen3:14b"]
    gate, _ = make_gate(tmp_path, link)
    assert gate.unload_engine_models_left_partly_on_cpu() == ["qwen3:14b"]
    assert link.loaded == ["nomic-embed-text"]
    assert link.calls == [("unload-named", ("qwen3:14b",))]


def test_a_partly_loaded_model_is_left_alone_while_engine_is_answering_or_the_gate_is_held(tmp_path):
    link = FakeLink(ENGINE_ANSWERING, loaded=["qwen3:14b"])
    link.partial = ["qwen3:14b"]
    gate, _ = make_gate(tmp_path, link)
    with pytest.raises(GateBusy):
        gate.unload_engine_models_left_partly_on_cpu()
    assert link.loaded == ["qwen3:14b"]
    link.status = QUIET
    link.loaded = []
    gate.acquire()
    link.loaded = ["qwen3:14b"]
    assert gate.unload_engine_models_left_partly_on_cpu() == []
    assert link.loaded == ["qwen3:14b"]
