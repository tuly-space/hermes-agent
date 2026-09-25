"""Profile-local, file-backed one-shot idle timetable; no per-session timers.

The JSON file is the authority. All mutations re-read under an interprocess lock and
replace+fsync before publishing the in-memory heap. A corrupt file fails closed: it
must never be overwritten by an empty snapshot or cause another migration scan.
"""
from __future__ import annotations

import heapq
import json
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_FILENAME = "native_idle_timetable.json"
_MANAGERS: dict[Path, "IdleTimetable"] = {}
_REGISTRY_LOCK = threading.Lock()


def for_home(home: Path | None = None) -> "IdleTimetable":
    home = Path(home or get_hermes_home()).resolve()
    with _REGISTRY_LOCK:
        if home not in _MANAGERS:
            _MANAGERS[home] = IdleTimetable(home / _FILENAME)
        return _MANAGERS[home]


def stop_for_home(home: Path) -> None:
    """Stop an existing owner, without initializing a disabled profile."""
    with _REGISTRY_LOCK:
        table = _MANAGERS.get(Path(home).resolve())
    if table is not None:
        table.stop()


class IdleTimetable:
    def __init__(self, path: Path, *, workers: int = 2):
        self.path = Path(path)
        self.workers = workers
        self._condition = threading.Condition()
        self._entries: dict[str, dict] = {}
        self._heap: list[tuple[float, str, str]] = []
        self._active: dict[str, threading.Event] = {}
        self._active_agents: dict[str, object] = {}
        self._suppressed: set[tuple[str, str]] = set()
        self._worker_count = 0
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._run: Callable[[str, dict, threading.Event], bool | None] | None = None
        self._initialized = False
        self._loaded = False
        self.waits = 0  # diagnostic: includes indefinite waits, never a polling tick

    def _locked_file(self):
        # Dedicated lock inode survives os.replace of the data file.
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = open(str(self.path) + ".lock", "a+b")
        fcntl.flock(lock, fcntl.LOCK_EX)
        return lock

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "initialized": False, "entries": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("version") != 1
                or not isinstance(data.get("initialized"), bool)
                or not isinstance(data.get("entries"), dict)):
            raise ValueError("invalid idle timetable schema")
        for sid, item in data["entries"].items():
            if (not isinstance(sid, str) or not isinstance(item, dict)
                    or not isinstance(item.get("revision"), str)
                    or not isinstance(item.get("session_key"), str)
                    or not isinstance(item.get("idle_at"), (int, float))
                    or not math.isfinite(item["idle_at"])
                    or not isinstance(item.get("delay"), (int, float))
                    or not math.isfinite(item["delay"]) or item["delay"] <= 0
                    or not math.isfinite(item["idle_at"] + item["delay"])):
                raise ValueError("invalid idle timetable entry")
        return data

    def _write(self, data: dict) -> None:
        import tempfile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".native-idle-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _publish(self, data: dict) -> None:
        self._entries = dict(data["entries"])
        self._initialized = data["initialized"]
        self._suppressed.intersection_update((sid, v["revision"]) for sid, v in self._entries.items())
        self._heap = [(v["idle_at"] + v["delay"], sid, v["revision"])
                      for sid, v in self._entries.items()
                      if sid not in self._active and (sid, v["revision"]) not in self._suppressed]
        heapq.heapify(self._heap)
        self._loaded = True
        self._condition.notify_all()

    def _mutate(self, change) -> None:
        with self._condition:
            with self._locked_file() as lock:
                try:
                    data = self._read()
                    if change(data):
                        self._write(data)
                    self._publish(data)
                finally:
                    lock.close()

    def load(self) -> None:
        self._mutate(lambda _: False)

    def backfill_once(self, discover: Callable[[], list[tuple[str, str, float, float]]]) -> int:
        """Scan once under a migration lock, keeping ordinary file updates short."""
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.path) + ".migration.lock", "a+b") as migration:
            fcntl.flock(migration, fcntl.LOCK_EX)
            with self._condition:
                with self._locked_file() as lock:
                    try:
                        data = self._read()
                        if data["initialized"]:
                            self._publish(data)
                            return 0
                    finally:
                        lock.close()
            # No data-file lock during the potentially large SessionStore scan.
            # Foreground turns may update their own rows; the final read merges them.
            candidates = discover()
            with self._condition:
                with self._locked_file() as lock:
                    try:
                        data = self._read()
                        if data["initialized"]:
                            self._publish(data)
                            return 0
                        added = 0
                        for sid, key, idle_at, delay in candidates:
                            if (not math.isfinite(idle_at) or not math.isfinite(delay)
                                    or delay <= 0 or not math.isfinite(idle_at + delay)):
                                raise ValueError("invalid migration idle deadline")
                            if sid not in data["entries"]:
                                data["entries"][sid] = dict(session_key=key, idle_at=idle_at,
                                                              delay=delay, revision=uuid.uuid4().hex)
                                added += 1
                        data["initialized"] = True
                        self._write(data)
                        self._publish(data)
                        return added
                    finally:
                        lock.close()

    def update(self, sid: str, key: str, idle_at: float, delay: float) -> str:
        if (not sid or not isinstance(idle_at, (int, float)) or not math.isfinite(idle_at)
                or not isinstance(delay, (int, float)) or not math.isfinite(delay)
                or delay <= 0 or not math.isfinite(idle_at + delay)):
            raise ValueError("idle update requires session and finite positive deadline")
        revision = uuid.uuid4().hex
        def change(data):
            data["entries"][sid] = dict(session_key=key, idle_at=idle_at,
                                         delay=delay, revision=revision)
            return True
        self._mutate(change)
        return revision

    def remove(self, sid: str, revision: str | None = None) -> bool:
        removed = False
        def change(data):
            nonlocal removed
            current = data["entries"].get(sid)
            removed = bool(current and (revision is None or current["revision"] == revision))
            if removed:
                del data["entries"][sid]
            return removed
        self._mutate(change)
        return removed

    def cancel(self, sid: str) -> None:
        with self._condition:
            event = self._active.get(sid)
            if event is not None:
                event.set()
            agent = self._active_agents.get(sid)
        if agent is not None:
            from agent.native_maintenance import cancel_idle
            cancel_idle(agent, timetable=False)
        self.remove(sid)

    def start(self, run: Callable[[str, dict, threading.Event], bool | None]) -> None:
        with self._condition:
            if self._thread and self._thread.is_alive():
                if self._run is not run:
                    logger.warning("Idle timetable already running; refusing to replace its owner")
                return
            self._run = run
            if not self._loaded:
                self.load()
            self._stopped = False
            self._thread = threading.Thread(target=self._loop, name="native-idle-timetable", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            for event in self._active.values():
                event.set()
            agents = list(self._active_agents.values())
            self._condition.notify_all()
        for agent in agents:
            from agent.native_maintenance import cancel_idle
            cancel_idle(agent, timetable=False)
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        with self._condition:
            while not self._stopped:
                # When the pool is full do NOT peek at an overdue head and wait(0).
                if self._worker_count >= self.workers or not self._heap or self._run is None:
                    self.waits += 1
                    self._condition.wait()
                    continue
                deadline, sid, revision = self._heap[0]
                remaining = deadline - time.time()
                if remaining > 0:
                    self.waits += 1
                    self._condition.wait(remaining)
                    continue
                heapq.heappop(self._heap)
                item = self._entries.get(sid)
                if not item or item["revision"] != revision or sid in self._active:
                    continue
                cancellation = threading.Event()
                self._active[sid] = cancellation
                self._worker_count += 1
                try:
                    threading.Thread(target=self._work, args=(sid, dict(item), cancellation),
                                     name="native-idle-worker", daemon=True).start()
                except Exception:
                    # A failed spawn is not work performed. Retain the durable row for
                    # restart but never dispatch this overdue revision again in this run.
                    logger.warning("Idle worker could not start for %s", sid, exc_info=True)
                    self._active.pop(sid, None)
                    self._worker_count -= 1
                    self._suppressed.add((sid, revision))

    def _work(self, sid: str, item: dict, cancellation: threading.Event) -> None:
        attempted = False
        committed = False
        try:
            if not cancellation.is_set() and self._run is not None:
                attempted = True
                committed = bool(self._run(sid, item, cancellation))
        except Exception:
            logger.warning("Idle maintenance failed for %s", sid, exc_info=True)
        finally:
            # A shutdown-cancelled pass with no durable commit is still pending
            # work. Foreground cancellation instead deletes/replaces the row.
            if attempted and (committed or not (self._stopped and cancellation.is_set())):
                try:
                    self.remove(sid, item["revision"])
                except Exception:
                    logger.warning("Idle timetable could not retire %s; restart will recheck", sid, exc_info=True)
                    with self._condition:
                        self._suppressed.add((sid, item["revision"]))
            with self._condition:
                self._active.pop(sid, None)
                self._worker_count -= 1
                # The updated version (if any) was excluded from the heap during in-flight
                # publication; put it back on worker completion.
                current = self._entries.get(sid)
                if current and (sid, current["revision"]) not in self._suppressed and not self._stopped:
                    heapq.heappush(self._heap, (current["idle_at"] + current["delay"],
                                                sid, current["revision"]))
                self._condition.notify_all()
