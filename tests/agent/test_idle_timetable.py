"""Durable timetable and bounded event-driven idle scheduling contracts."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from agent.idle_timetable import IdleTimetable


def _until(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.01)
    return bool(predicate())


def test_backfill_only_once_recovery_and_revision_delete(tmp_path):
    path = tmp_path / "a" / "native_idle_timetable.json"
    table = IdleTimetable(path)
    now = time.time()
    assert table.backfill_once(lambda: [("old", "key-old", now - 10, 25)]) == 1
    first = table._entries["old"].copy()
    table.update("new", "key-new", now, 100)
    assert table.backfill_once(lambda: pytest.fail("historical rescan")) == 0
    restarted = IdleTimetable(path)
    restarted.load()
    assert restarted._entries["old"] == first
    old_revision = restarted._entries["new"]["revision"]
    restarted.update("new", "key-new", now + 1, 100)
    assert not restarted.remove("new", old_revision)
    assert restarted.remove("old", first["revision"])
    assert "old" not in IdleTimetable(path)._read()["entries"]
    assert restarted.backfill_once(lambda: pytest.fail("rescan after deletion")) == 0
    assert restarted.remove("new")
    assert IdleTimetable(path)._read()["entries"] == {}
    # Malformed file is never treated as an empty, eligible initial scan.
    path.write_text("{not-json")
    with pytest.raises(json.JSONDecodeError):
        restarted.backfill_once(lambda: pytest.fail("corruption rescan"))
    with pytest.raises(json.JSONDecodeError):
        restarted.update("must-not-overwrite", "key", now, 25)
    assert path.read_text() == "{not-json"


def test_two_workers_wait_for_completion_and_earlier_deadline(tmp_path):
    table = IdleTimetable(tmp_path / "table.json")
    gate = threading.Event()
    lock = threading.Lock()
    started = []
    def run(sid, item, cancellation):
        with lock:
            started.append(sid)
        if sid.startswith("burst"):
            gate.wait(3)
    try:
        table.start(run)
        table.update("late", "late-key", time.time(), 30)
        table.update("early", "early-key", time.time() - 10, 10)
        assert _until(lambda: "early" in started)
        assert "late" not in started
        for n in range(24):
            table.update(f"burst{n}", f"key{n}", time.time() - 2, 1)
        assert _until(lambda: len([x for x in started if x.startswith("burst")]) == 2)
        assert _until(lambda: table._worker_count == 2 and table.waits > 0)
        time.sleep(.03)  # settle the last condition notification from the final update
        waits = table.waits
        time.sleep(.1)
        assert table.waits <= waits + 1 and table._worker_count == 2
        assert len([x for x in started if x.startswith("burst")]) == 2
        gate.set()
        assert _until(lambda: len([x for x in started if x.startswith("burst")]) == 24)
        assert _until(lambda: not any(k.startswith("burst") for k in table._entries))
        assert "late" in table._entries
    finally:
        gate.set()
        table.stop()
    fresh = IdleTimetable(table.path)
    fresh.load()
    assert set(fresh._entries) == {"late"}


def test_foreground_cancels_inflight_without_deleting_new_version(tmp_path):
    table = IdleTimetable(tmp_path / "table.json")
    entered, leave = threading.Event(), threading.Event()
    events = []
    def run(sid, item, cancellation):
        entered.set()
        leave.wait(3)
        events.append(cancellation.is_set())
    try:
        table.start(run)
        table.update("sid", "key", time.time() - 3, 1)
        assert entered.wait(2)
        table.cancel("sid")  # foreground starts; revoke claimed generation
        new = table.update("sid", "key", time.time(), 100)
        leave.set()
        assert _until(lambda: bool(events) and table._worker_count == 0)
        assert events == [True]
        assert IdleTimetable(table.path)._read()["entries"]["sid"]["revision"] == new
    finally:
        leave.set()
        table.stop()


def test_timetables_are_profile_owned_a_b_a(tmp_path):
    from agent.idle_timetable import for_home
    a, b = tmp_path / "a", tmp_path / "b"
    first = for_home(a)
    other = for_home(b)
    first.update("a-session", "key-a", time.time(), 20)
    other.update("b-session", "key-b", time.time(), 20)
    assert for_home(a) is first and set(first._entries) == {"a-session"}
    assert set(other._entries) == {"b-session"}


def test_skipped_and_failed_checks_are_one_shot(tmp_path):
    table = IdleTimetable(tmp_path / "table.json")
    calls = []
    def check(sid, _item, _cancelled):
        calls.append(sid)
        if sid == "failure":
            raise RuntimeError("provider down")
        # "low" is an eligibility skip, not a rescheduling request.
    try:
        table.start(check)
        for sid in ("low", "failure"):
            table.update(sid, sid, time.time() - 5, 1)
        assert _until(lambda: len(calls) == 2 and not table._entries)
        time.sleep(.1)
        assert sorted(calls) == ["failure", "low"]
    finally:
        table.stop()
    assert IdleTimetable(table.path)._read()["entries"] == {}


def test_failed_retirement_waits_for_new_revision_not_overdue_spin(tmp_path, monkeypatch):
    table = IdleTimetable(tmp_path / "table.json")
    table.update("sid", "key", time.time() - 5, 1)
    calls = []
    original_write = table._write
    def fail_retire(data):
        if "sid" not in data["entries"]:
            raise OSError("disk full")
        return original_write(data)
    monkeypatch.setattr(table, "_write", fail_retire)
    try:
        table.start(lambda sid, item, event: calls.append(item["revision"]))
        assert _until(lambda: table._worker_count == 0 and len(calls) == 1)
        waits = table.waits
        time.sleep(.1)
        assert len(calls) == 1 and table.waits <= waits + 1
        assert "sid" in IdleTimetable(table.path)._read()["entries"]
        table.update("sid", "key", time.time() - 5, 1)
        assert _until(lambda: len(calls) == 2)
        assert calls[0] != calls[1]
    finally:
        table.stop()


def test_shutdown_cancellation_keeps_uncommitted_claim_for_restart(tmp_path):
    table = IdleTimetable(tmp_path / "table.json")
    entered = threading.Event()
    table.update("sid", "key", time.time() - 5, 1)
    def interrupted(_sid, _item, event):
        entered.set()
        event.wait(2)
        return False
    table.start(interrupted)
    assert entered.wait(2)
    table.stop()
    assert _until(lambda: table._worker_count == 0)
    assert "sid" in IdleTimetable(table.path)._read()["entries"]
    restarted = IdleTimetable(table.path)
    calls = []
    try:
        restarted.start(lambda sid, item, event: calls.append(sid))
        assert _until(lambda: calls == ["sid"] and "sid" not in restarted._entries)
    finally:
        restarted.stop()


def test_worker_start_failure_retains_row_without_dispatch_loop(tmp_path, monkeypatch):
    table = IdleTimetable(tmp_path / "table.json")
    table.update("sid", "key", time.time() - 5, 1)
    calls = []
    original_start = threading.Thread.start
    def fail_worker(self):
        if self.name == "native-idle-worker":
            raise RuntimeError("thread unavailable")
        return original_start(self)
    monkeypatch.setattr(threading.Thread, "start", fail_worker)
    try:
        table.start(lambda sid, item, event: calls.append(sid))
        assert _until(lambda: ("sid", table._entries["sid"]["revision"]) in table._suppressed)
        waits = table.waits
        time.sleep(.1)
        assert table._thread is not None and table._thread.is_alive() and table._worker_count == 0
        assert not calls and table.waits <= waits + 1
        assert "sid" in IdleTimetable(table.path)._read()["entries"]
    finally:
        table.stop()


def test_non_gateway_timer_cannot_replace_shared_gateway_scheduler(tmp_path, monkeypatch):
    from agent import native_maintenance as maintenance
    from agent import idle_timetable, periodic_scheduler
    table = IdleTimetable(tmp_path / "table.json")
    gateway = lambda sid, item, event: None
    handles = []
    monkeypatch.setattr(idle_timetable, "for_home", lambda *_: table)
    monkeypatch.setattr(maintenance, "eligible", lambda _: True)
    monkeypatch.setattr(periodic_scheduler, "schedule", lambda fn, delay: handles.append(fn) or SimpleNamespace(cancel=lambda: None))
    agent = SimpleNamespace(compression_native_idle_after_seconds=25, _session_db=SimpleNamespace(db_path=tmp_path / "state.db"),
                            session_id="cli-session")
    try:
        table.start(gateway)
        maintenance.arm_idle(agent, {"completed": True})
        assert table._run is gateway and len(handles) == 1
        assert "cli-session" not in table._entries
    finally:
        table.stop()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_deadline_is_rejected(tmp_path, bad):
    table = IdleTimetable(tmp_path / "table.json")
    with pytest.raises(ValueError):
        table.update("sid", "key", bad, 1)
    with pytest.raises(ValueError):
        table.update("sid", "key", time.time(), bad)
    table.path.write_text(json.dumps({"version": 1, "initialized": True,
                                      "entries": {"sid": {"revision": "r", "session_key": "key",
                                                          "idle_at": bad, "delay": 1}}}))
    with pytest.raises(ValueError):
        table.load()
