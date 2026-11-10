"""Publish the exported viewer to a GitHub Pages branch.

The viewer in ``data/build/viz`` is already a static site: the page, the cost
model it recomputes in the browser, and the data, all loaded by relative path.
It is built locally, because the pipeline downloads gigabytes no CI run
should, so publishing commits that folder as the whole of a ``gh-pages``
branch and pushes it there. Each publish is one commit with no parent: the
branch is build output whose history is ``main``'s, and replacing it keeps old
payloads from piling up on the remote.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config, viz

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


def stale(site: Path, build: Path) -> list[str]:
    """Return why ``site`` would not show this checkout and build, or nothing if it would."""
    data = site / "data.json"
    if not data.exists():
        return [f"{data} is missing"]
    problems = []
    if not (site / "index.html").exists() or (site / "index.html").read_text() != viz.page():
        problems.append("index.html is not the checkout's viz/index.html")
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


def commit_site(site: Path, repo: Path, message: str, branch: str = BRANCH) -> str:
    """Commit every file under ``site`` as the whole of ``branch`` and return the commit.

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
        tree = git(repo, "write-tree", env=env)
    commit = git(repo, "commit-tree", tree, "-m", message)
    git(repo, "update-ref", f"refs/heads/{branch}", commit)
    return commit


def pages_url(remote_url: str) -> str | None:
    """Return the Pages address of a GitHub remote, or None for any other host."""
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", remote_url)
    return f"https://{m[1].lower()}.github.io/{m[2]}/" if m else None
