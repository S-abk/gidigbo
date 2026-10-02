"""Get the UFCStats source CSVs and make sure the processed dataset is ready.

The app and CI call `ensure_data(commit)` so a fresh clone (e.g. a Streamlit Community
Cloud deploy) works without a manual `git clone` of the data repo:

- The four CSVs the pipeline reads are downloaded from the source repo at a *pinned*
  commit -- by default the one recorded in models/model_metadata.json, i.e. exactly the
  data the shipped model was trained and tested on. The refresh workflow asks for the
  latest commit instead.
- A SOURCE_COMMIT marker next to the raw CSVs and in data/processed/ records which
  upstream commit they came from, so stale processed data is rebuilt automatically
  when the model (and therefore the expected commit) changes.
- A raw directory that is a git checkout (the README's manual `git clone`) is never
  overwritten; it is used as-is.

CLI:
  python -m src.fetch_data                         # pinned commit from model metadata (else latest)
  python -m src.fetch_data --latest                # newest upstream commit
  python -m src.fetch_data --print-latest-commit   # just print it (used by the refresh workflow)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path

from src.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR

SOURCE_REPO = "Greco1899/scrape_ufc_stats"
RAW_FILES = ["ufc_event_details.csv", "ufc_fight_results.csv", "ufc_fight_stats.csv", "ufc_fighter_tott.csv"]
PROCESSED_FILES = ["fights.parquet", "appearances.parquet", "fighters.parquet"]
MARKER = "SOURCE_COMMIT"
USER_AGENT = "ufc-predictor-mvp/0.1 (+https://github.com/S-abk/gidigbo)"


def _log(msg: str) -> None:
    print(f"[fetch_data] {msg}", flush=True)


def _get(url: str, timeout: float = 60, attempts: int = 3) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"  # avoids the 60/hour anonymous limit in CI
    last = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # network blips: retry
            last = exc
    raise RuntimeError(f"could not download {url}: {last}")


def latest_commit() -> str:
    return json.loads(_get(f"https://api.github.com/repos/{SOURCE_REPO}/commits/main"))["sha"]


def raw_source_commit() -> str | None:
    """Upstream commit of the raw CSVs: git HEAD for a clone, else the download marker."""
    if (RAW_DIR / ".git").exists():
        try:
            return subprocess.check_output(["git", "-C", str(RAW_DIR), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            return None
    marker = RAW_DIR / MARKER
    return marker.read_text().strip() if marker.exists() else None


def processed_source_commit() -> str | None:
    marker = PROCESSED_DIR / MARKER
    return marker.read_text().strip() if marker.exists() else None


def pinned_commit() -> str | None:
    """The upstream commit the shipped model was trained on (None if unknown)."""
    meta = MODELS_DIR / "model_metadata.json"
    if not meta.exists():
        return None
    return (json.loads(meta.read_text()).get("data_source") or {}).get("commit")


def download_raw(commit: str) -> None:
    """Download the CSVs at `commit` into RAW_DIR (atomically, with a header sanity check)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Remove the marker first: if the download dies partway, a stale marker on top of a mix
    # of old and new files would otherwise be accepted later as a complete old download.
    (RAW_DIR / MARKER).unlink(missing_ok=True)
    for name in RAW_FILES:
        data = _get(f"https://raw.githubusercontent.com/{SOURCE_REPO}/{commit}/{name}")
        header = data[:200].decode("utf-8", errors="replace")
        if "EVENT" not in header and "FIGHTER" not in header:
            raise RuntimeError(f"{name} at {commit[:10]} doesn't look like the expected CSV")
        tmp = RAW_DIR / f".{name}.part"
        tmp.write_bytes(data)
        tmp.replace(RAW_DIR / name)
        _log(f"downloaded {name} ({len(data) / 1e6:.1f} MB)")
    (RAW_DIR / MARKER).write_text(commit + "\n")


def ensure_raw(commit: str | None) -> str | None:
    """Make RAW_DIR hold the source CSVs, at `commit` when it is known. Returns the commit."""
    have = raw_source_commit()
    complete = all((RAW_DIR / f).exists() for f in RAW_FILES)
    if (RAW_DIR / ".git").exists():
        if commit and have != commit:
            _log(f"raw data is a git checkout at {str(have)[:10]}, not {commit[:10]}; using it as-is")
        return have
    if complete and (commit is None or have == commit):
        return have
    commit = commit or latest_commit()
    _log(f"downloading source CSVs at {commit[:10]}")
    download_raw(commit)
    return commit


def ensure_data(commit: str | None = None) -> str | None:
    """Raw CSVs + processed parquet files present and built from the same upstream commit."""
    from src.build_dataset import build_all  # heavy import, only when a build is needed

    commit = commit or pinned_commit()
    processed_ok = all((PROCESSED_DIR / f).exists() for f in PROCESSED_FILES)
    if processed_ok and (commit is None or processed_source_commit() == commit):
        return processed_source_commit()
    raw_commit = ensure_raw(commit)
    _log(f"building the processed dataset from {str(raw_commit)[:10]}")
    build_all(save=True)  # writes data/processed/SOURCE_COMMIT
    return raw_commit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--latest", action="store_true", help="use the newest upstream commit")
    ap.add_argument("--print-latest-commit", action="store_true", help="print the newest upstream commit and exit")
    args = ap.parse_args()
    if args.print_latest_commit:
        print(latest_commit())
        return
    commit = latest_commit() if args.latest else pinned_commit()
    ensure_raw(commit)
    _log(f"raw data ready at {str(raw_source_commit())[:10]} in {RAW_DIR}")


if __name__ == "__main__":
    main()
