"""Regression tests for CLI gateway run exit behavior.

``hermes gateway run`` enters through hermes_cli.gateway, not gateway.run.main().
After graceful teardown it must use the same hard-exit backstop as gateway.run.main()
so Python finalization does not wait on non-daemon worker threads (for example
in-flight cron ThreadPoolExecutor jobs) and delay service-managed restarts.
"""

from __future__ import annotations

import types

import pytest


class _HardExitObserved(BaseException):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


def _prepare(monkeypatch):
    import hermes_cli.gateway as gateway_cli
    import pm.environments
    # A source checkout inside the real home is not a sealed payload; importing
    # gateway.run must not probe the real home's manifest under the I/O guard.
    monkeypatch.setattr(pm.environments, "_payload_manifest", lambda root: None)
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_cli, "_guard_official_docker_root_gateway", lambda: None)
    monkeypatch.setattr(gateway_cli, "_guard_named_profile_under_multiplexer", lambda force=False: None)
    monkeypatch.setattr(gateway_cli, "_attach_to_host_gateway_or_guard", lambda **kwargs: None)
    monkeypatch.setattr(gateway_cli, "_guard_supervised_gateway_conflict", lambda force=False: None)
    monkeypatch.setattr(gateway_cli, "_guard_existing_gateway_process_conflict", lambda replace=False: None)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway_cli.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    monkeypatch.setenv("HERMES_GATEWAY_EXIT_DIAG", "0")

    async def _start_gateway(*args, **kwargs):  # pragma: no cover - never awaited by fake run
        return True

    def _hard_exit(code: int) -> None:
        raise _HardExitObserved(code)

    monkeypatch.setattr(gateway_run, "start_gateway", _start_gateway)
    monkeypatch.setattr(gateway_run, "_exit_after_graceful_shutdown", _hard_exit)
    return gateway_cli


def test_run_gateway_hard_exits_after_clean_return(monkeypatch):
    gateway_cli = _prepare(monkeypatch)

    def _fake_run(coro):
        coro.close()
        return True

    monkeypatch.setattr(gateway_cli.asyncio, "run", _fake_run)

    with pytest.raises(_HardExitObserved) as excinfo:
        gateway_cli.run_gateway()

    assert excinfo.value.code == 0


def test_join_probe_reports_live_default_executor_worker(monkeypatch, caplog, tmp_path):
    """A stuck default-executor worker gets a stack, and the probe never holds up exit."""
    import asyncio
    import threading
    import hermes_cli.gateway as gateway_cli
    import hermes_constants

    _prepare(monkeypatch)
    import gateway.run as gateway_run
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    released = threading.Event()
    records = []
    monkeypatch.setattr(gateway_cli, "_make_exit_diag", lambda: lambda tag, **data: records.append((tag, data)))

    async def _start_gateway(**kwargs):
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, released.wait)
        await asyncio.sleep(0.05)
        return True

    monkeypatch.setattr(gateway_run, "start_gateway", _start_gateway)
    original_timer = threading.Timer

    class ImmediateTimer:
        def __init__(self, delay, callback):
            assert delay == 5.0
            self.callback = callback
            self.daemon = False

        def start(self):
            try:
                self.callback()
            finally:
                released.set()

        def cancel(self):
            pass

    monkeypatch.setattr(threading, "Timer", ImmediateTimer)
    try:
        with pytest.raises(_HardExitObserved) as excinfo:
            gateway_cli.run_gateway()
    finally:
        released.set()
        monkeypatch.setattr(threading, "Timer", original_timer)
    assert excinfo.value.code == 0
    reports = [data for tag, data in records if tag == "asyncio.executor_join_wait"]
    assert reports and reports[0]["thread"].startswith("asyncio_")
    assert any("wait" in call for call in reports[0]["stack"])
    assert "Default executor still joining" in caplog.text
    dump = (tmp_path / "logs" / "gateway.error.log").read_text()
    assert "=== EXECUTOR THREAD DUMP AT LOOP CLOSE ===" in dump
    assert "asyncio_" in dump
    assert "Thread" in dump
    assert "thread name='asyncio_" in caplog.text
    assert "daemon=False" in caplog.text
    assert "ident=" in caplog.text


def test_dump_failure_does_not_interrupt_shutdown(monkeypatch, tmp_path, caplog):
    """An unwritable diagnostics path cannot prevent loop teardown."""
    import hermes_cli.gateway as gateway_cli
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    (tmp_path / "logs").write_text("not a directory")
    gateway_cli._dump_executor_threads_at_loop_close()
    assert "Executor thread dump failed" in caplog.text


def test_run_gateway_hard_exits_after_keyboard_interrupt(monkeypatch):
    """KeyboardInterrupt (console Ctrl+C) must also hard-exit, not return.

    A bare ``return`` would let Python finalization join non-daemon worker
    threads, the same wedge this backstop prevents on the other exit paths.
    """
    gateway_cli = _prepare(monkeypatch)

    def _fake_run(coro):
        coro.close()
        raise KeyboardInterrupt()

    monkeypatch.setattr(gateway_cli.asyncio, "run", _fake_run)

    with pytest.raises(_HardExitObserved) as excinfo:
        gateway_cli.run_gateway()

    assert excinfo.value.code == 0
