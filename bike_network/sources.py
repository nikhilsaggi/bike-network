"""Download the public inputs into ``data/raw``, once.

Each fetch writes to a temporary name and renames on success, so an
interrupted download never leaves a file that looks complete.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from pathlib import Path

USER_AGENT = "bike-network/0.1"


def _download(url: str, dest: Path, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 -- fixed https URLs from config
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:  # noqa: S310
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a zip, refusing members that would land outside ``dest``."""
    root = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not target.is_relative_to(root):
                msg = f"zip member escapes extraction dir: {member}"
                raise ValueError(msg)
        zf.extractall(dest)


def ensure_lion(release: str = config.LION_RELEASE) -> Path:
    """Return the path to LION's file geodatabase, downloading it if absent."""
    folder = config.RAW_DIR / f"lion_{release}"
    gdb = folder / "lion" / "lion.gdb"
    if gdb.exists():
        return gdb
    archive = folder / f"nyclion_{release}.zip"
    if not archive.exists():
        print(f"  Downloading LION {release}...")
        _download(config.LION_URL.format(release=release), archive)
    _safe_extract(archive, folder)
    if not gdb.exists():
        msg = f"{archive} did not contain lion/lion.gdb"
        raise FileNotFoundError(msg)
    return gdb


def ensure_bike_routes() -> Path:
    """Return the path to DOT's bike route attributes, downloading if absent."""
    path = config.RAW_DIR / "bike_routes" / "bike_routes.json"
    if path.exists():
        return path
    print("  Downloading DOT bike routes...")
    _download(config.BIKE_ROUTES_URL, path)
    rows = json.loads(path.read_text())
    if len(rows) >= config.BIKE_ROUTES_LIMIT:
        path.unlink()
        msg = f"bike routes response filled the {config.BIKE_ROUTES_LIMIT} row limit"
        raise RuntimeError(msg)
    return path
