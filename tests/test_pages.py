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


def site(path, data=f'{{"format": {viz.FORMAT}}}'):
    path.mkdir(parents=True, exist_ok=True)
    (path / "index.html").write_text(pages.page())
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


def test_viewer_reads_the_format_the_export_writes():
    assert pages.page_format(pages.page()) == viz.FORMAT


def test_export_in_another_format_is_refused(tmp_path):
    build = tmp_path / "build"
    out = site(build / "viz", data='{"n": 1}')
    assert pages.stale(out, build) == [
        f"data.json is format 0 and the viewer reads format {viz.FORMAT}"
    ]


@pytest.fixture
def remote(repo, tmp_path):
    """A bare remote whose gh-pages holds an older viewer over some data."""
    bare = tmp_path / "remote.git"
    pages.git(tmp_path, "init", "-q", "--bare", str(bare))
    pages.git(repo, "remote", "add", "origin", str(bare))
    old = site(tmp_path / "old", data=f'{{"format": {viz.FORMAT}, "n": 7}}')
    (old / "index.html").write_text("an older viewer")
    commit = pages.commit_site(old, repo, "Publish the map built from abc1234")
    pages.git(repo, "push", "-q", "origin", f"{commit}:refs/heads/{pages.BRANCH}")
    return bare


def test_refresh_puts_the_viewer_over_the_published_data(repo, remote):
    commit = pages.refresh(repo)
    assert pages.git(remote, "rev-parse", pages.BRANCH) == commit
    assert pages.git(repo, "show", f"{commit}:index.html") == pages.page().strip()
    assert pages.git(repo, "show", f"{commit}:data.json") == f'{{"format": {viz.FORMAT}, "n": 7}}'
    assert pages.git(repo, "rev-list", "--count", commit) == "1"
    first = pages.git(repo, "log", "-1", "--format=%B", commit)
    assert "Data: Publish the map built from abc1234 (" in first
    # nothing new: nothing published
    assert pages.refresh(repo) is None
    # a later page-only publish still names the build the data came from
    (repo / "README.md").write_text("changed\n")
    pages.git(repo, "commit", "-qam", "change")
    data_line = next(line for line in first.splitlines() if line.startswith("Data: "))
    old = site(repo.parent / "older", data=f'{{"format": {viz.FORMAT}, "n": 7}}')
    (old / "index.html").write_text("yet another viewer")
    stale_commit = pages.git(
        repo, "commit-tree", pages.site_tree(old, repo), "-m", f"Publish the viewer\n\n{data_line}"
    )
    pages.git(repo, "push", "-qf", "origin", f"{stale_commit}:refs/heads/{pages.BRANCH}")
    again = pages.refresh(repo)
    assert data_line in pages.git(repo, "log", "-1", "--format=%B", again).splitlines()


def test_refresh_refuses_data_in_another_format(repo, remote, tmp_path):
    other = site(tmp_path / "other", data='{"n": 1}')
    commit = pages.commit_site(other, repo, "Publish the map built from old")
    pages.git(repo, "push", "-qf", "origin", f"{commit}:refs/heads/{pages.BRANCH}")
    with pytest.raises(SystemExit, match="format 0"):
        pages.refresh(repo)
    assert pages.git(remote, "rev-parse", pages.BRANCH) == commit
