"""``hermes update --commit``: the run installs the commit a wrapper tested, not whatever the
channel points at by the time it runs, and refuses a commit the channel never published."""
import argparse
from copy import deepcopy
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main, source_releases, update_cmd
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_channel import set_install_channel

pytestmark = pytest.mark.real_release_channels


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL).stdout.strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    home, origin, checkout = (tmp_path / name for name in ("home", "origin", "checkout"))
    home.mkdir()
    origin.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Channel Fixture")
    git(origin, "config", "user.email", "fixture@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    commits = []
    for label in ("installed", "tested-release", "newer-release"):
        (origin / "content.txt").write_text(label)
        git(origin, "add", "content.txt")
        git(origin, "commit", "-m", label)
        commits.append(git(origin, "rev-parse", "HEAD"))
    # A commit the channel never published: on a side branch, not in main's history.
    git(origin, "checkout", "-b", "side", commits[0])
    (origin / "content.txt").write_text("side")
    git(origin, "commit", "-am", "side")
    side = git(origin, "rev-parse", "HEAD")
    git(origin, "checkout", "main")
    git(tmp_path, "clone", str(origin), str(checkout))
    git(checkout, "checkout", "--detach", commits[0])
    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: checkout)
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=main.cmd_update)
    opts = update_cmd._UpdateOptions(pre_update_version=None, gw_input_fn=None,
        assume_yes=True, keep_stash=False, switch_branch=False, discard_local_changes=False)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda **_: (False, ["git"], False))
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update",
                        lambda request: completed.append(deepcopy(request)))
    return SimpleNamespace(home=home, origin=origin, root=checkout, commits=commits, side=side,
                           parser=parser, completed=completed)


def publish(monkeypatch, commit, name="preview-pinned"):
    """The channel archive names ``commit`` as its current release."""
    def record(state="active"):
        return {"schema": 1, "name": name, "repository": "NousResearch/hermes-agent",
                "policy": "preview", "state": state, "identity": {}, "nextSequence": 2,
                "head": {"buildId": "build-fixture", "sequence": 1}}
    result = SimpleNamespace(requested=record(), terminal=record(), manifest={
        "schema": 1, "request": {"buildId": "build-fixture", "channel": name, "sequence": 1,
        "repository": "NousResearch/hermes-agent", "commit": commit, "sourceVersion": "1.2.3",
        "version": "0.0.1", "identity": {}, "bundleEnv": {},
        "publicBase": "https://hermes-assets.nousresearch.com"}, "packages": []})
    monkeypatch.setattr(source_releases, "_resolve_channel",
                        lambda _name, _repository: deepcopy(result), raising=False)


def run(source, *flags):
    update_cmd._cmd_update_impl(source.parser.parse_args(["update", "--yes", *flags]), False)


def test_release_published_after_the_test_does_not_change_what_lands(source, monkeypatch):
    """The wrapper tested release 1; release 2 is published before Update runs. The pinned run
    installs release 1 exactly, and the completion it hands on names release 1."""
    set_install_channel("preview-pinned", source.root)
    publish(monkeypatch, source.commits[2])
    run(source, "--channel", "preview-pinned", "--commit", source.commits[1])
    assert git(source.root, "rev-parse", "HEAD") == source.commits[1]
    assert (source.root / "content.txt").read_text() == "tested-release"
    assert source.completed[0]["expected_sha"] == source.commits[1]


def test_unpinned_run_still_takes_the_newest_release(source, monkeypatch):
    """Control for the test above: without --commit the same archive lands release 2."""
    set_install_channel("preview-pinned", source.root)
    publish(monkeypatch, source.commits[2])
    run(source, "--channel", "preview-pinned")
    assert git(source.root, "rev-parse", "HEAD") == source.commits[2]


def test_pin_equal_to_the_current_release_lands_it(source, monkeypatch):
    publish(monkeypatch, source.commits[1])
    run(source, "--channel", "preview-pinned", "--commit", source.commits[1])
    assert git(source.root, "rev-parse", "HEAD") == source.commits[1]


def test_main_channel_pin_holds_while_the_branch_advances(source):
    """origin/main is already past the tested commit; the pinned run stops at the tested one."""
    run(source, "--channel", "main", "--commit", source.commits[1])
    assert git(source.root, "rev-parse", "HEAD") == source.commits[1]
    assert git(source.root, "rev-parse", "origin/main") == source.commits[2]


@pytest.mark.parametrize("channel", ["preview-pinned", "main"])
def test_commit_outside_the_channel_is_refused_before_anything_moves(source, monkeypatch, channel):
    publish(monkeypatch, source.commits[2])
    before = git(source.root, "rev-parse", "HEAD")
    with pytest.raises(SystemExit) as exc:
        run(source, "--channel", channel, "--commit", source.side)
    assert exc.value.code == 1
    assert git(source.root, "rev-parse", "HEAD") == before
    assert source.completed == []


def test_abbreviated_commit_is_rejected_by_the_parser(source, capsys):
    with pytest.raises(SystemExit):
        source.parser.parse_args(["update", "--commit", source.commits[1][:12]])
    assert "full commit id" in capsys.readouterr().err
