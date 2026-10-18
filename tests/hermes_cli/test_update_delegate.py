"""``updates.delegate_command``: on a host that updates through its own wrapper, a bare
``hermes update`` hands over to that wrapper and never moves the checkout itself; the wrapper's
own inner run (``HERMES_UPDATE_DELEGATED=1``) and hosts without the key update normally."""
import argparse
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import main, update_delegate
from hermes_cli.subcommands.update import build_update_parser


class Handover(Exception):
    """Stands in for os.execv replacing the process."""


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL).stdout.strip()


@pytest.fixture
def host(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv(update_delegate.DELEGATED_ENV, raising=False)
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    wrapper = tmp_path / "host_update.sh"
    wrapper.write_text("#!/bin/sh\nexit 0\n")
    wrapper.chmod(0o755)
    calls = []

    def fake_execv(path, argv):
        calls.append((path, argv))
        raise Handover

    monkeypatch.setattr(update_delegate.os, "execv", fake_execv)
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=main.cmd_update)

    def configure(command=str(wrapper)):
        (root / "config.yaml").write_text(f"updates:\n  delegate_command: '{command}'\n")

    return argparse.Namespace(root=root, wrapper=wrapper, calls=calls, parser=parser,
                              configure=configure, tmp=tmp_path)


def args(host, *flags):
    return host.parser.parse_args(["update", *flags])


def test_host_without_the_key_updates_normally(host):
    update_delegate.handle_delegated_update(args(host))
    assert host.calls == []


def test_bare_update_hands_over_to_the_wrapper(host):
    host.configure()
    with pytest.raises(Handover):
        update_delegate.handle_delegated_update(args(host))
    assert host.calls == [(str(host.wrapper), [str(host.wrapper)])]


def test_backup_choice_is_forwarded_and_yes_is_harmless(host):
    host.configure()
    with pytest.raises(Handover):
        update_delegate.handle_delegated_update(args(host, "--backup", "--yes"))
    assert host.calls[0][1] == [str(host.wrapper), "--backup"]


def test_wrappers_own_inner_update_is_not_handed_back(host, monkeypatch):
    """The recursion escape: the wrapper's sanctioned `hermes update` runs the real updater."""
    host.configure()
    monkeypatch.setenv(update_delegate.DELEGATED_ENV, "1")
    update_delegate.handle_delegated_update(args(host, "--channel", "stable", "--no-gateway-restart"))
    assert host.calls == []


@pytest.mark.parametrize("flag", ["--check", "--plan", "--list-venv-holders"])
def test_read_only_forms_never_hand_over(host, flag):
    host.configure()
    update_delegate.handle_delegated_update(args(host, flag))
    assert host.calls == []


@pytest.mark.parametrize("flags", [["--branch", "dev"], ["--channel", "main"], ["--keep-stash"],
                                   ["--no-gateway-restart"]])
def test_flags_the_wrapper_cannot_honour_are_refused_not_dropped(host, flags, capsys):
    host.configure()
    with pytest.raises(SystemExit) as exc:
        update_delegate.handle_delegated_update(args(host, *flags))
    assert exc.value.code == 2
    assert host.calls == []
    assert "Nothing was changed" in capsys.readouterr().out


def test_missing_wrapper_refuses_instead_of_falling_back(host, capsys):
    """A broken wrapper path must never fall through to the plain update it exists to prevent."""
    host.configure(str(host.tmp / "gone.sh"))
    with pytest.raises(SystemExit) as exc:
        update_delegate.handle_delegated_update(args(host))
    assert exc.value.code == 2
    assert "missing or not executable" in capsys.readouterr().out


def test_chat_update_is_refused_and_reported_failed(host):
    host.configure()
    with pytest.raises(SystemExit) as exc:
        update_delegate.handle_delegated_update(args(host, "--gateway"))
    assert exc.value.code == 2
    assert host.calls == []
    assert (host.root / ".update_exit_code").read_text().strip() == "1"


def test_named_profile_reads_the_root_setting(host, monkeypatch):
    """`hermes -p work update` scopes HERMES_HOME to the profile; the host setting is the root's."""
    host.configure()
    profile = host.root / "profiles" / "work"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model: {}\n")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(Path, "home", lambda: host.tmp)
    with pytest.raises(Handover):
        update_delegate.handle_delegated_update(args(host))


def test_bare_update_cannot_move_a_checkout_carrying_local_commits(host, monkeypatch):
    """End to end through `cmd_update`: the carried commit is still HEAD after a bare update."""
    host.configure()
    origin, checkout = host.tmp / "origin", host.tmp / "checkout"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    for key, value in (("user.name", "Fixture"), ("user.email", "f@example.invalid"),
                       ("commit.gpgsign", "false")):
        git(origin, "config", key, value)
    (origin / "a.txt").write_text("upstream")
    git(origin, "add", "a.txt")
    git(origin, "commit", "-m", "upstream")
    git(host.tmp, "clone", str(origin), str(checkout))
    for key, value in (("user.name", "Fixture"), ("user.email", "f@example.invalid"),
                       ("commit.gpgsign", "false")):
        git(checkout, "config", key, value)
    (checkout / "a.txt").write_text("carried fix")
    git(checkout, "commit", "-am", "carried fix")
    carried = git(checkout, "rev-parse", "HEAD")
    (origin / "a.txt").write_text("upstream moved")
    git(origin, "commit", "-am", "upstream moved")
    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    monkeypatch.setattr("hermes_cli.update_owning_install.retarget_to_owning_install", lambda *_: None)
    with pytest.raises(Handover):
        main.cmd_update(args(host, "--yes"))
    assert git(checkout, "rev-parse", "HEAD") == carried
    assert (checkout / "a.txt").read_text() == "carried fix"
    assert os.access(host.wrapper, os.X_OK)
