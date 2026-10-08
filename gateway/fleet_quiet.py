"""Fleet-update quiet gate: hold every inbound message while a fleet update runs.

Marker: ``<root HERMES home>/fleet-update-quiet.json`` — ONE file shared by every profile and every
gateway process, written and removed by olympus-ops (this module never writes or deletes it).
Fields read: ``expires`` (epoch seconds), ``owner_pid`` (int; 0/missing = owner-less), ``phase``
(``preflight`` | ``passed`` | ``update``), ``since`` (epoch). Other fields (``paused``, ``count``,
``pending``) are ignored here.

Active predicate (``is_active``):

* marker missing                                   -> NOT quiet (release)
* marker unreadable / not a JSON object            -> NOT quiet (fail-open, logged once)
* ``now >= expires``                               -> NOT quiet (expiry = release)
* ``owner_pid`` > 0, phase != ``passed``, PID dead -> NOT quiet (stale: the owner crashed)
* otherwise (incl. owner-less unexpired ``passed``) -> QUIET

The owner PID is a process handle only: it can END a stale preflight/update window early, never
extend one past ``expires``. A missing/non-numeric ``expires`` is clamped to
``since + MAX_UNDATED_S`` so a malformed marker cannot wedge the fleet.

Hold queue: ``<root>/fleet-update-quiet.d/<profile>.jsonl`` (dir 0700, files 0600 — they hold
message text). Append-only JSONL, fsynced per record, FIFO by line order. Duplicate-safe: a record
whose ``(channel, message_id | platform_update_id)`` is already queued is not appended twice
(platform redelivery after reconnect). Bounded: ``MAX_HELD`` records / ``MAX_QUEUE_BYTES``; past the
bound new messages are refused (the caller logs and still sends the one notice).

Replay is crash-safe and at-most-once per record: ``claim_held`` atomically renames the live queue
to ``<profile>.replay-<ts>.jsonl``; the replayer dispatches records in order and advances a fsynced
cursor file after each one; a gateway restart mid-replay resumes from the cursor. Ordering is FIFO
per channel: a record that cannot be dispatched yet (its adapter is offline, or its session is
mid-turn) moves — together with every later record of the same channel — to a newer carry claim, so
one stuck channel never blocks the others. Claim files are deleted once fully consumed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MARKER_NAME = "fleet-update-quiet.json"
QUEUE_DIR_NAME = "fleet-update-quiet.d"
NOTICE_TEXT = "Update in progress, I'll answer after."
MAX_UNDATED_S = 2 * 3600.0
MAX_HELD = 1000
MAX_QUEUE_BYTES = 4 * 1024 * 1024
MAX_RECORD_AGE_S = 24 * 3600.0  # a record that cannot be replayed for a day is dropped (logged)
_lock = threading.RLock()
_corrupt_logged = False


# ── paths ────────────────────────────────────────────────────────────────────────────────────────

def root_home() -> Path:
    """The SHARED root Hermes home (``<root>/profiles/<p>`` -> ``<root>``)."""
    override = os.environ.get("HERMES_FLEET_QUIET_ROOT")
    if override:
        return Path(override)
    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    if home.parent.name == "profiles":
        return home.parent.parent
    return home


def marker_path() -> Path:
    return root_home() / MARKER_NAME


def queue_dir() -> Path:
    return root_home() / QUEUE_DIR_NAME


def _profile_name() -> str:
    """Queue owner = the gateway PROCESS's home (multiplexed gateway -> ``default``). Each record
    carries its own ``source.profile`` so replay re-routes to the right profile's adapter."""
    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return home.name if home.parent.name == "profiles" else "default"


def queue_path(profile: Optional[str] = None) -> Path:
    return queue_dir() / f"{profile or _profile_name()}.jsonl"


def status_path(profile: Optional[str] = None) -> Path:
    return queue_dir() / f"{profile or _profile_name()}.status.json"


def _ensure_dir() -> Path:
    d = queue_dir()
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


# ── marker ───────────────────────────────────────────────────────────────────────────────────────

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except Exception:
        return False


def read_marker() -> Optional[Dict[str, Any]]:
    global _corrupt_logged
    try:
        raw = marker_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception:
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("marker is not a JSON object")
        _corrupt_logged = False
        return data
    except Exception as exc:
        if not _corrupt_logged:
            _corrupt_logged = True
            logger.warning("Fleet quiet marker unreadable (%s) — treating as NOT quiet", exc)
        return None


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def evaluate(data: Optional[Dict[str, Any]], now: Optional[float] = None) -> Tuple[bool, str]:
    """``(active, reason)`` for a parsed marker; pure apart from the owner-PID liveness probe."""
    if not data:
        return False, "no-marker"
    now = time.time() if now is None else now
    expires = _num(data.get("expires"))
    if expires is None:
        since = _num(data.get("since"))
        if since is None:
            return False, "undated"  # neither expires nor since: cannot bound it -> fail open
        expires = since + MAX_UNDATED_S
    if now >= expires:
        return False, "expired"
    owner = data.get("owner_pid") or 0
    try:
        owner = int(owner)
    except (TypeError, ValueError):
        owner = 0
    if owner > 0 and data.get("phase") != "passed" and not _pid_alive(owner):
        return False, "owner-dead"
    return True, "active"


def is_active(data: Optional[Dict[str, Any]] = None, now: Optional[float] = None) -> bool:
    return evaluate(read_marker() if data is None else data, now)[0]


def retry_after_s(data: Optional[Dict[str, Any]] = None, now: Optional[float] = None) -> int:
    """Seconds until the marker expires, clamped to [1, 60] (an HTTP Retry-After hint)."""
    data = read_marker() if data is None else data
    now = time.time() if now is None else now
    exp = _num((data or {}).get("expires"))
    if exp is None:
        return 30
    return int(max(1, min(60, exp - now)))


# ── events ───────────────────────────────────────────────────────────────────────────────────────

def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return None


def serialize_event(event: Any) -> Dict[str, Any]:
    src = event.source
    ts = getattr(event, "timestamp", None)
    mt = getattr(event, "message_type", None)
    return {
        "v": 1, "held_at": time.time(),
        "text": event.text, "message_type": getattr(mt, "value", mt) or "text",
        "source": src.to_dict() if src is not None else None,
        "message_id": getattr(event, "message_id", None),
        "platform_update_id": getattr(event, "platform_update_id", None),
        "media_urls": list(getattr(event, "media_urls", None) or []),
        "media_types": list(getattr(event, "media_types", None) or []),
        "reply_to_message_id": getattr(event, "reply_to_message_id", None),
        "reply_to_text": getattr(event, "reply_to_text", None),
        "reply_to_author_id": getattr(event, "reply_to_author_id", None),
        "reply_to_author_name": getattr(event, "reply_to_author_name", None),
        "reply_to_is_own_message": bool(getattr(event, "reply_to_is_own_message", False)),
        "auto_skill": _jsonable(getattr(event, "auto_skill", None)),
        "channel_prompt": getattr(event, "channel_prompt", None),
        "channel_context": _jsonable(getattr(event, "channel_context", None)),
        "reply_expected": getattr(event, "reply_expected", None),
        "allow_gateway_control": bool(getattr(event, "allow_gateway_control", False)),
        "metadata": {k: v for k, v in (getattr(event, "metadata", None) or {}).items()
                     if _jsonable(v) is not None},
        "timestamp": ts.isoformat() if isinstance(ts, datetime) else None,
    }


_RESTORE_KEYS = ("message_id", "platform_update_id", "media_urls", "media_types",
                 "reply_to_message_id", "reply_to_text", "reply_to_author_id", "reply_to_author_name",
                 "reply_to_is_own_message", "auto_skill", "channel_prompt", "channel_context",
                 "reply_expected", "allow_gateway_control", "metadata")


def deserialize_event(rec: Dict[str, Any]) -> Any:
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import SessionSource
    ev = MessageEvent(text=rec.get("text") or "", message_type=MessageType(rec.get("message_type") or "text"),
                      source=SessionSource.from_dict(rec["source"]))
    for key in _RESTORE_KEYS:
        if key in rec and rec[key] is not None and hasattr(ev, key):
            setattr(ev, key, rec[key])
    if rec.get("timestamp"):
        try:
            ev.timestamp = datetime.fromisoformat(rec["timestamp"])
        except Exception:
            pass
    ev.metadata = dict(ev.metadata or {})
    ev.metadata["fleet_quiet_replay"] = True
    return ev


def channel_key(source: Any) -> str:
    plat = getattr(getattr(source, "platform", None), "value", "?")
    prof = getattr(source, "profile", None) or ""
    return f"{prof}|{plat}:{getattr(source, 'chat_id', '?')}:{getattr(source, 'thread_id', None) or ''}"


def _dedup_key(rec: Dict[str, Any]) -> Optional[str]:
    ident = rec.get("message_id") or rec.get("platform_update_id")
    return f"{rec.get('channel')}#{ident}" if ident not in (None, "") else None


# ── queue ────────────────────────────────────────────────────────────────────────────────────────

def _read(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        out.append(rec)
                except Exception:
                    logger.warning("Fleet quiet queue: skipping corrupt line in %s", path.name)
    except FileNotFoundError:
        pass
    return out


class QueueFull(Exception):
    pass


def hold(event: Any) -> Dict[str, bool]:
    """Durably append ``event``. Returns ``{"first": <first hold for this channel this window>,
    "duplicate": <already queued>}``. Raises :class:`QueueFull` past the bound and ``OSError`` on
    disk failure (caller falls back)."""
    rec = serialize_event(event)
    rec["channel"] = channel_key(event.source)
    path = queue_path()
    with _lock:
        _ensure_dir()
        existing = _read(path)
        for claim in _claim_files():  # a window that re-engaged mid-replay: still the same backlog
            existing.extend(_read(claim))
        key = _dedup_key(rec)
        if key is not None and any(_dedup_key(r) == key for r in existing):
            return {"first": False, "duplicate": True}
        size = path.stat().st_size if path.exists() else 0
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        if len(existing) >= MAX_HELD or size + len(line.encode("utf-8")) > MAX_QUEUE_BYTES:
            raise QueueFull(f"{len(existing)} records / {size} bytes")
        first = rec["channel"] not in {r.get("channel") for r in existing}
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
    return {"first": first, "duplicate": False}


def _claim_files(profile: Optional[str] = None) -> List[Path]:
    d = queue_dir()
    if not d.exists():
        return []
    stem = profile or _profile_name()
    return sorted(p for p in d.glob(f"{stem}.replay-*.jsonl"))


def _cursor_path(claim: Path) -> Path:
    return claim.with_suffix(".cursor")


def read_cursor(claim: Path) -> int:
    try:
        return max(0, int(_cursor_path(claim).read_text(encoding="utf-8").strip() or 0))
    except Exception:
        return 0


def write_cursor(claim: Path, index: int) -> None:
    cp = _cursor_path(claim)
    tmp = cp.with_suffix(".cursor.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(str(int(index)))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, cp)


def new_carry_path() -> Path:
    """A claim file that sorts AFTER every existing claim: records a replay pass could not dispatch
    yet (adapter offline / session mid-turn) move here so one stuck channel never blocks the rest."""
    _ensure_dir()
    return queue_dir() / f"{_profile_name()}.replay-{time.time_ns():020d}.jsonl"


def append_carry(carry: Path, rec: Dict[str, Any]) -> None:
    with _lock:
        key = _dedup_key(rec)
        if key is not None and any(_dedup_key(r) == key for r in _read(carry)):
            return  # crash between carry-append and cursor-advance: do not carry twice
        fd = os.open(carry, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def finish_claim(claim: Path) -> None:
    for p in (claim, _cursor_path(claim)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def claim_held() -> List[Path]:
    """Move the live queue into a new claim file; return every claim (oldest first) still to replay."""
    path = queue_path()
    with _lock:
        if path.exists() and path.stat().st_size > 0:
            _ensure_dir()
            claim = queue_dir() / f"{_profile_name()}.replay-{time.time_ns():020d}.jsonl"
            os.replace(path, claim)
        elif path.exists():
            path.unlink()
        return _claim_files()


def load_held() -> List[Dict[str, Any]]:
    """Every record not yet replayed: unfinished claims (from their cursor) then the live queue."""
    with _lock:
        out: List[Dict[str, Any]] = []
        for claim in _claim_files():
            out.extend(_read(claim)[read_cursor(claim):])
        out.extend(_read(queue_path()))
        return out


def pending_count() -> int:
    return len(load_held())


# ── status (panel integration) ───────────────────────────────────────────────────────────────────

def status(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = read_marker()
    active, reason = evaluate(data)
    out = {"active": active, "reason": reason, "phase": (data or {}).get("phase"),
           "since": (data or {}).get("since"), "expires": (data or {}).get("expires"),
           "owner_pid": (data or {}).get("owner_pid"), "held": pending_count(),
           "profile": _profile_name(), "pid": os.getpid(), "updated_at": time.time()}
    if extra:
        out.update(extra)
    return out


def write_status(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist :func:`status` to ``<root>/fleet-update-quiet.d/<profile>.status.json`` (0600)."""
    snap = status(extra)
    try:
        _ensure_dir()
        sp = status_path()
        tmp = sp.with_suffix(".json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(snap, fh)
        os.replace(tmp, sp)
    except Exception:
        logger.debug("Fleet quiet: status write failed", exc_info=True)
    return snap
