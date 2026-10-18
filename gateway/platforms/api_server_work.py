"""Owner-checked work slots behind the API server's in-flight counters.

The adapter's busy count (``_pending_agent_requests`` + ``_inflight_agent_runs`` +
``_orphaned_api_workers``, and their per-profile ``Counter`` twins) used to be bare ``+= 1`` /
``-= 1`` pairs. Any path that incremented and then failed to reach its release (an exception
between the increment and the ``try``, a done callback that never ran) left the gateway reporting
BUSY until a restart, and a fleet update refuses to restart a BUSY gateway.

Every unit is now a *slot* that records the asyncio task (or other ``done()``-able handle) that
owns its release. A slot whose owner has finished while the slot is still open is a lost release:
readers subtract it at once (so status is right on the very next read, from any thread) and the
owning loop is asked to release it for real. Releases are exactly-once, so a late legitimate
release after a reclaim is a no-op and nothing is double-counted. The counters stay the fast path
and the public shape; the slots only ever lower a count whose owner provably cannot release it.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# kind -> (host-total attribute, per-profile Counter attribute)
_COUNTERS = {
    "pending": ("_pending_agent_requests", "_pending_by_profile"),
    "inflight": ("_inflight_agent_runs", "_inflight_by_profile"),
    "orphan": ("_orphaned_api_workers", "_orphaned_by_profile"),
}


class FlagOwner:
    """A slot owner that is finished once ``finished()`` says so (an executor thread's flag)."""

    def __init__(self, finished: Callable[[], bool]) -> None:
        self._finished = finished

    def done(self) -> bool:
        return bool(self._finished())


def _registry(adapter: Any) -> dict:
    slots = vars(adapter).get("_work_slots")
    if slots is None:
        slots = {}
        adapter._work_slots = slots
    return slots


def _current_task() -> Optional[asyncio.Task]:
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def acquire(adapter: Any, kind: str, profile: str, *, slot: Optional[dict] = None,
            owner: Any = None) -> dict:
    """Count one unit of ``kind`` work for ``profile``; ``owner`` defaults to the current task."""
    reap_lost_work(adapter)
    slot = {} if slot is None else slot
    slot.update(active=True, kind=kind, profile=profile,
                owner=_current_task() if owner is None else owner, loop=_running_loop())
    total, by_profile = _COUNTERS[kind]
    setattr(adapter, total, int(getattr(adapter, total, 0)) + 1)
    getattr(adapter, by_profile)[profile] += 1
    _registry(adapter)[id(slot)] = slot
    return slot


def release(adapter: Any, slot: Optional[dict]) -> bool:
    """Release ``slot`` exactly once; True when this call released it."""
    if not slot or not slot.get("active"):
        return False
    slot["active"] = False
    _registry(adapter).pop(id(slot), None)
    total, by_profile = _COUNTERS[slot["kind"]]
    setattr(adapter, total, max(0, int(getattr(adapter, total, 0)) - 1))
    counter = getattr(adapter, by_profile)
    counter[slot["profile"]] -= 1
    if counter[slot["profile"]] <= 0:
        del counter[slot["profile"]]
    return True


def _owner_finished(owner: Any) -> bool:
    if owner is None:
        return False  # acquired outside any task: nothing to check, trust the explicit release
    try:
        return bool(owner.done())
    except Exception:
        return False  # unreadable owner stays counted: false BUSY is the safe failure


def lost_slots(adapter: Any) -> list[dict]:
    """Open slots whose owner has finished: counted, but nothing is left to release them."""
    return [slot for slot in list(_registry(adapter).values())
            if slot.get("active") and _owner_finished(slot.get("owner"))]


def lost_by_profile(lost: list[dict]) -> Counter:
    return Counter(slot["profile"] for slot in lost)


def schedule_reap(adapter: Any, lost: list[dict]) -> None:
    """Ask each owning loop to release ``lost`` for real (safe from the status socket's thread)."""
    for loop in {slot["loop"] for slot in lost if slot.get("loop") is not None}:
        try:
            loop.call_soon_threadsafe(reap_lost_work, adapter)
        except RuntimeError:
            pass  # loop closed: the adapter is gone with it


def reap_lost_work(adapter: Any) -> int:
    """Release every lost slot (call on the owning loop); returns how many were reclaimed."""
    reclaimed = 0
    for slot in lost_slots(adapter):
        if release(adapter, slot):
            reclaimed += 1
            logger.warning(
                "[api_server] reclaimed a lost %s release for profile %s: its owner finished "
                "without releasing, so the busy count would have stayed high until restart",
                slot["kind"], slot["profile"])
    if reclaimed:
        persist = getattr(adapter, "_persist_active_work", None)
        if callable(persist):
            persist()
    return reclaimed
