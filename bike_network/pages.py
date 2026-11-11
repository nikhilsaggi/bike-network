"""Publish the exported viewer to a GitHub Pages branch.

The viewer in ``data/build/viz`` is already a static site: the page, the cost
model it recomputes in the browser, and the data, all loaded by relative path.
It is built locally, because the pipeline downloads gigabytes no CI run
should, so publishing commits that folder as the whole of a ``gh-pages``
branch and pushes it there. Each publish is one commit with no parent: the
branch is build output whose history is ``main``'s, and replacing it keeps old
payloads from piling up on the remote.

A change to the page alone does not need the data rebuilt. ``python -m
bike_network.pages refresh`` (run by .github/workflows/pages.yml on every
push to ``main`` that touches the viewer) republishes the checkout's page and
cost model over the ``data.json`` already on the branch, as long as the page
reads that data's format. This module imports nothing outside the standard
library, so that runs without the pipeline's dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config

BRANCH = "gh-pages"
GIT = shutil.which("git")
# Build outputs the payload is made from: newer than data.json means the
# export no longer shows them.
INPUTS = (
    "segments.parquet",
    "edges.parquet",
    "facilities.parquet",
    "traffic.parquet",
    "flows.parquet",
    "routes.parquet",
    "corridors.json",
    "corridors.parquet",
    "corridor_edges.parquet",
    "plan_effect.json",
    "plan_flows.parquet",
    "plan_examples.parquet",
    "plan_stations.parquet",
)


def git(repo: Path, *args: str, env: dict[str, str] | None = None, stdin: str = "") -> str:
    """Run git in ``repo`` and return its output."""
    if GIT is None:
        msg = "git is not installed"
        raise RuntimeError(msg)
    res = subprocess.run(  # noqa: S603 -- git with arguments built here
        [GIT, *args], cwd=repo, env=env, input=stdin, capture_output=True, text=True, check=True
    )
    return res.stdout.strip()


def page() -> str:
    """Return the viewer page as a whole document.

    ``viz/index.html`` is written as a page body (the artifact host adds the
    document around it); a standalone copy gets that skeleton here.
    """
    body = (config.ROOT / "viz" / "index.html").read_text()
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        '<meta name="description" content="Every NYC street costed by cycling stress, a year of '
        'Citi Bike trips routed over it, and where new protected lanes would help most.">'
        f"</head><body>\n{body}\n</body></html>\n"
    )


def page_format(html: str) -> int:
    """Return the ``data.json`` format a viewer page reads (its ``DATA_FORMAT``)."""
    m = re.search(r"\bconst DATA_FORMAT = (\d+);", html)
    if m is None:
        msg = "the viewer page does not declare DATA_FORMAT"
        raise ValueError(msg)
    return int(m[1])


def data_format(data: Path) -> int:
    """Return the format a ``data.json`` was written in; 0 for one from before formats."""
    return int(json.loads(data.read_text()).get("format", 0))


def stale(site: Path, build: Path) -> list[str]:
    """Return why ``site`` would not show this checkout and build, or nothing if it would."""
    data = site / "data.json"
    if not data.exists():
        return [f"{data} is missing"]
    problems = []
    if not (site / "index.html").exists() or (site / "index.html").read_text() != page():
        problems.append("index.html is not the checkout's viz/index.html")
    elif (have := data_format(data)) != (want := page_format(page())):
        problems.append(f"data.json is format {have} and the viewer reads format {want}")
    source = (config.ROOT / "viz" / "cost.js").read_text()
    if not (site / "cost.js").exists() or (site / "cost.js").read_text() != source:
        problems.append("cost.js is not the checkout's viz/cost.js")
    built = data.stat().st_mtime
    problems += [
        f"{name} is newer than data.json"
        for name in INPUTS
        if (build / name).exists() and (build / name).stat().st_mtime > built
    ]
    return problems


def site_tree(site: Path, repo: Path) -> str:
    """Write every file under ``site``, plus ``.nojekyll``, as a git tree and return it.

    It goes through a scratch index, so the working tree, the real index and
    the branch checked out are never touched.
    """
    lines = [
        f"100644 {git(repo, 'hash-object', '-w', '--no-filters', '--', str(p))}\t"
        f"{p.relative_to(site).as_posix()}"
        for p in sorted(site.rglob("*"))
        if p.is_file()
    ]
    # Pages runs a branch through Jekyll unless told not to; nothing here needs building.
    lines.append(f"100644 {git(repo, 'hash-object', '-w', '--stdin')}\t.nojekyll")
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git(repo, "update-index", "--add", "--index-info", env=env, stdin="\n".join(lines) + "\n")
        return git(repo, "write-tree", env=env)


def commit_site(site: Path, repo: Path, message: str, branch: str = BRANCH) -> str:
    """Commit every file under ``site`` as the whole of ``branch`` and return the commit."""
    commit = git(repo, "commit-tree", site_tree(site, repo), "-m", message)
    git(repo, "update-ref", f"refs/heads/{branch}", commit)
    return commit


def pages_url(remote_url: str) -> str | None:
    """Return the Pages address of a GitHub remote, or None for any other host."""
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", remote_url)
    return f"https://{m[1].lower()}.github.io/{m[2]}/" if m else None


def refresh(repo: Path, remote: str = "origin", *, push: bool = True) -> str | None:
    """Republish ``repo``'s viewer over the data on ``remote``'s branch.

    Returns the new commit, or None when the branch already serves this page.
    Raises SystemExit, publishing nothing, when the page reads another format
    than the published data: that change needs ``viz`` and ``publish`` run
    where the build is.
    """
    ref = f"refs/remotes/{remote}/{BRANCH}"
    git(repo, "fetch", "--quiet", remote, f"+refs/heads/{BRANCH}:{ref}")
    old = git(repo, "rev-parse", ref)
    html = page()
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp)
        with (site / "data.json").open("wb") as out:
            subprocess.run(  # noqa: S603 -- git with arguments built here
                [GIT, "cat-file", "blob", f"{old}:data.json"], cwd=repo, stdout=out, check=True
            )
        have, want = data_format(site / "data.json"), page_format(html)
        if have != want:
            msg = (
                f"Not publishing: the published data.json is format {have} and this viewer reads "
                f"format {want}. Run `python -m bike_network viz` and "
                "`python -m bike_network publish` where the build is."
            )
            raise SystemExit(msg)
        (site / "index.html").write_text(html)
        (site / "cost.js").write_text((config.ROOT / "viz" / "cost.js").read_text())
        tree = site_tree(site, repo)
    if tree == git(repo, "rev-parse", f"{old}^{{tree}}"):
        return None
    # Keep saying where the data came from, however many page-only publishes follow.
    body = git(repo, "log", "-1", "--format=%B", old)
    data = re.search(r"^Data: (.+)$", body, re.MULTILINE)
    origin = data[1] if data else f"{body.splitlines()[0]} ({old[:7]})"
    head = git(repo, "rev-parse", "--short", "HEAD")
    message = f"Publish the viewer from {head} over the published data\n\nData: {origin}"
    commit = git(repo, "commit-tree", tree, "-m", message)
    git(repo, "update-ref", f"refs/heads/{BRANCH}", commit)
    if push:
        # Lease on the commit read: a full publish landing meanwhile is not overwritten.
        git(
            repo,
            "push",
            f"--force-with-lease=refs/heads/{BRANCH}:{old}",
            remote,
            f"{commit}:refs/heads/{BRANCH}",
        )
    return commit


def main() -> None:
    """Run ``refresh`` from the command line."""
    parser = argparse.ArgumentParser(prog="python -m bike_network.pages")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("refresh", help="republish the viewer over the published data")
    cmd.add_argument("--no-push", action="store_true", help="commit the branch but do not push it")
    cmd.add_argument("--remote", default="origin", help="remote to read and push (default origin)")
    args = parser.parse_args()
    commit = refresh(config.ROOT, args.remote, push=not args.no_push)
    if commit is None:
        print(f"{BRANCH} already serves this viewer")
    else:
        print(
            f"{BRANCH} is now {commit[:7]}" + ("" if args.no_push else f", pushed to {args.remote}")
        )


if __name__ == "__main__":
    main()
