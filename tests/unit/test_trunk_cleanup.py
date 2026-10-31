"""Native trunk-test cleanup must reap its child without killing the tracker."""
from types import SimpleNamespace
from unittest.mock import Mock, call
import subprocess

import pytest

from tests.integration import test_trunk_identity as trunk


@pytest.fixture
def cleanup(monkeypatch):
    proc = Mock()
    proc.pid = 4029
    proc.poll.return_value = 0
    proc.wait.return_value = 0
    probe = Mock(side_effect=ProcessLookupError)
    sleep = Mock()
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(trunk.os, "killpg", probe)
    monkeypatch.setattr(trunk.time, "sleep", sleep)
    monkeypatch.setattr(trunk.time, "monotonic", clock)
    yield SimpleNamespace(proc=proc, probe=probe, sleep=sleep, clock=clock)
    # Even failure paths may only inspect the group; no TERM/KILL to a stale PGID.
    assert all(args == call(proc.pid, 0) for args in probe.call_args_list)


def test_exited_child_is_not_signalled_or_waited_again(cleanup):
    trunk._cleanup(cleanup.proc)

    cleanup.proc.terminate.assert_not_called()
    cleanup.proc.kill.assert_not_called()
    cleanup.proc.wait.assert_not_called()
    cleanup.probe.assert_called_once_with(cleanup.proc.pid, 0)
    cleanup.sleep.assert_not_called()


def test_exited_child_allows_its_resource_tracker_time_to_finish(cleanup):
    cleanup.probe.side_effect = [None, None, ProcessLookupError]
    cleanup.clock.side_effect = [10.0, 10.1, 10.2]

    trunk._cleanup(cleanup.proc)

    cleanup.proc.terminate.assert_not_called()
    cleanup.proc.kill.assert_not_called()
    cleanup.proc.wait.assert_not_called()
    assert cleanup.sleep.call_args_list == [call(0.05), call(0.05)]
    assert cleanup.probe.call_count == 3


def test_live_child_gets_graceful_termination_and_is_reaped(cleanup):
    cleanup.proc.poll.return_value = None

    trunk._cleanup(cleanup.proc)

    cleanup.proc.terminate.assert_called_once_with()
    cleanup.proc.wait.assert_called_once_with(timeout=5)
    cleanup.proc.kill.assert_not_called()
    assert cleanup.proc.method_calls == [call.poll(), call.terminate(), call.wait(timeout=5)]


def test_live_child_is_killed_only_after_graceful_wait_times_out(cleanup):
    cleanup.proc.poll.return_value = None
    cleanup.proc.wait.side_effect = [subprocess.TimeoutExpired("child", 5), 0]

    trunk._cleanup(cleanup.proc)

    assert cleanup.proc.method_calls == [call.poll(), call.terminate(), call.wait(timeout=5),
                                         call.kill(), call.wait(timeout=5)]
    cleanup.probe.assert_called_once_with(cleanup.proc.pid, 0)


def test_child_that_survives_kill_still_fails_after_bounded_wait(cleanup):
    cleanup.proc.poll.return_value = None
    cleanup.proc.wait.side_effect = subprocess.TimeoutExpired("child", 5)

    with pytest.raises(subprocess.TimeoutExpired):
        trunk._cleanup(cleanup.proc)

    assert cleanup.proc.wait.call_args_list == [call(timeout=5), call(timeout=5)]
    cleanup.proc.kill.assert_called_once_with()
    cleanup.probe.assert_not_called()


def test_surviving_descendant_is_not_hidden_after_grace_period(cleanup):
    cleanup.probe.side_effect = None
    cleanup.clock.side_effect = [10.0, 10.1, 15.0]

    with pytest.raises(AssertionError, match="process group 4029 survived cleanup"):
        trunk._cleanup(cleanup.proc)

    cleanup.proc.terminate.assert_not_called()
    cleanup.proc.kill.assert_not_called()
    cleanup.sleep.assert_called_once_with(0.05)
    assert cleanup.probe.call_count == 2


def test_unverifiable_group_does_not_report_success(cleanup):
    cleanup.probe.side_effect = PermissionError("group cannot be inspected")

    with pytest.raises(PermissionError):
        trunk._cleanup(cleanup.proc)

    cleanup.proc.terminate.assert_not_called()
    cleanup.proc.kill.assert_not_called()
    cleanup.sleep.assert_not_called()
