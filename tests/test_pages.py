import os

import pytest

from bike_network import config, pages, viz

pytestmark = pytest.mark.skipif(pages.GIT is None, reason="git is not installed")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    pages.git(root, "init", "-q", "-b", "main")
    pages.git(root, "config", "user.name", "Test")
    pages.git(root, "config", "user.email", "test@example.com")
    (root / "README.md").write_text("source\n")
    pages.git(root, "add", "README.md")
    pages.git(root, "commit", "-q", "-m", "source")
    return root


def site(path, data="{}"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "index.html").write_text(viz.page())
    (path / "cost.js").write_text((config.ROOT / "viz" / "cost.js").read_text())
    (path / "data.json").write_text(data)
    return path


def test_site_becomes_the_whole_branch_and_nothing_else_moves(repo, tmp_path):
    out = site(tmp_path / "viz", data='{"n": 1}')
    head = pages.git(repo, "rev-parse", "HEAD")
    commit = pages.commit_site(out, repo, "publish")
    files = pages.git(repo, "ls-tree", "-r", "--name-only", pages.BRANCH).split("\n")
    assert files == [".nojekyll", "cost.js", "data.json", "index.html"]
    assert pages.git(repo, "show", f"{pages.BRANCH}:data.json") == '{"n": 1}'
    assert pages.git(repo, "rev-parse", pages.BRANCH) == commit
    # main, its index and its working tree are untouched
    assert pages.git(repo, "rev-parse", "HEAD") == head
    assert pages.git(repo, "symbolic-ref", "--short", "HEAD") == "main"
    assert pages.git(repo, "status", "--porcelain") == ""


def test_a_second_publish_replaces_the_first(repo, tmp_path):
    out = site(tmp_path / "viz")
    (out / "old.txt").write_text("gone next time")
    pages.commit_site(out, repo, "first")
    (out / "old.txt").unlink()
    (out / "data.json").write_text('{"n": 2}')
    commit = pages.commit_site(out, repo, "second")
    assert "old.txt" not in pages.git(repo, "ls-tree", "-r", "--name-only", commit)
    assert pages.git(repo, "show", f"{commit}:data.json") == '{"n": 2}'
    # no parent: the branch never holds an old payload
    assert pages.git(repo, "rev-list", "--count", commit) == "1"


def test_stale_export_is_refused(tmp_path):
    build = tmp_path / "build"
    out = site(build / "viz")
    assert pages.stale(out, build) == []
    (build / "corridors.json").write_text("{}")
    t = (out / "data.json").stat().st_mtime
    os.utime(build / "corridors.json", (t + 10, t + 10))
    assert pages.stale(out, build) == ["corridors.json is newer than data.json"]
    os.utime(build / "corridors.json", (t - 10, t - 10))
    (out / "index.html").write_text("an older viewer")
    assert pages.stale(out, build) == ["index.html is not the checkout's viz/index.html"]
    (out / "data.json").unlink()
    assert pages.stale(out, build) == [f"{out / 'data.json'} is missing"]


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/nikhilsaggi/bike-network.git",
        "https://github.com/nikhilsaggi/bike-network",
        "git@github.com:nikhilsaggi/bike-network.git",
    ],
)
def test_pages_url(remote):
    assert pages.pages_url(remote) == "https://nikhilsaggi.github.io/bike-network/"


def test_pages_url_is_none_off_github():
    assert pages.pages_url("https://gitlab.com/someone/bike-network.git") is None
