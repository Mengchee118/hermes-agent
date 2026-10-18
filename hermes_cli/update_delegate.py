"""Hand ``hermes update`` to the updater a host declares in ``updates.delegate_command``.

Some hosts update through their own wrapper: it rehearses carried local commits, pins the
release it tested and restarts a shared service once. On such a host a plain ``hermes update``
moves the checkout to the release and drops those commits from the running tree. Setting
``updates.delegate_command`` in the ROOT config.yaml makes every mutating ``hermes update`` run
that wrapper instead. The wrapper runs the real update itself with ``HERMES_UPDATE_DELEGATED=1``
set, which is the only way past the hand-off (recursion escape). Hosts without the key keep the
normal behaviour; read-only forms (``--check``, ``--plan``, ``--list-venv-holders``) always do.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DELEGATED_ENV = "HERMES_UPDATE_DELEGATED"

# Flags the wrapper understands, mapped to its own spelling; ``--yes`` changes nothing for a
# wrapper that never prompts. Any other non-default flag is refused rather than dropped, so a
# caller asking for a branch or channel never silently gets the host's default instead.
_FORWARDED = {"backup": "--backup", "no_backup": "--no-backup"}
_IGNORED = {"yes"}
_READ_ONLY = ("check", "plan", "list_venv_holders")
_NOT_UPDATE_FLAGS = {"command", "func", "gateway"} | set(_READ_ONLY)


def delegate_command(home: Path | None = None) -> str | None:
    """The configured wrapper path from the ROOT config.yaml, or None when unset.

    Raises RuntimeError when that file exists but cannot be read: whether this host delegates
    is then unknown, and guessing "no" would run the very update the setting exists to stop.
    """
    from hermes_cli.config import require_readable_config_before_write
    from hermes_constants import get_default_hermes_root

    root = home if home is not None else get_default_hermes_root()
    config = require_readable_config_before_write(root / "config.yaml")
    updates = config.get("updates") if isinstance(config, dict) else None
    value = updates.get("delegate_command") if isinstance(updates, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _requested_flags(args) -> list[str]:
    return sorted(name for name, value in vars(args).items()
                  if name not in _NOT_UPDATE_FLAGS and value not in (None, False, "", [], 0))


def _refuse(message: str, *, gateway_mode: bool) -> None:
    from hermes_cli.update_cmd_common import _record_stop

    print(message)
    if gateway_mode:
        from hermes_cli.update_cmd_fleet import _write_gateway_update_exit_code

        _write_gateway_update_exit_code(False)
    _record_stop("managed_install", without_receipt="refused")  # managed by the host wrapper
    sys.exit(2)


def handle_delegated_update(args, *, home: Path | None = None) -> None:
    """Return when this run should update normally; otherwise exec the wrapper or exit 2."""
    if os.environ.get(DELEGATED_ENV) == "1" or any(getattr(args, f, False) for f in _READ_ONLY):
        return
    gateway_mode = bool(getattr(args, "gateway", False))
    try:
        command = delegate_command(home)
    except RuntimeError as exc:
        _refuse(f"✗ Could not read whether this host has its own updater: {exc} Nothing was changed.",
                gateway_mode=gateway_mode)
    if command is None:
        return
    wrapper = Path(os.path.expanduser(command))
    nothing = "Nothing was changed."
    if gateway_mode:
        _refuse(f"✗ This host updates through {wrapper}, which must run from a terminal, not a chat. "
                f"{nothing}", gateway_mode=True)
    extra = [name for name in _requested_flags(args) if name not in _FORWARDED and name not in _IGNORED]
    if extra:
        flags = ", ".join("--" + name.replace("_", "-") for name in extra)
        _refuse(f"✗ This host updates through {wrapper}, which does not take {flags}. {nothing}",
                gateway_mode=False)
    if not (wrapper.is_file() and os.access(wrapper, os.X_OK)):
        _refuse(f"✗ This host updates through {wrapper}, but that file is missing or not executable. "
                f"{nothing} Fix the file (or remove updates.delegate_command) and try again.",
                gateway_mode=False)
    forwarded = [flag for name, flag in _FORWARDED.items() if getattr(args, name, False)]
    print(f"→ This host updates through {wrapper}; handing over (a plain `hermes update` here "
          "would drop the local fixes this host carries).", flush=True)
    sys.stderr.flush()
    os.execv(str(wrapper), [str(wrapper), *forwarded])
