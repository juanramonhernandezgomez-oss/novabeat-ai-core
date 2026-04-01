from __future__ import annotations

import subprocess
from pathlib import Path


MERGE_MARKERS = ("<<<" + "<<<<", "===" + "====", ">>>" + ">>>>")


def test_repository_has_no_merge_conflict_markers() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    tracked_files = subprocess.check_output(
        ["git", "ls-files"],
        cwd=repo_root,
        text=True,
    ).splitlines()

    offenders: list[str] = []
    for rel_path in tracked_files:
        path = repo_root / rel_path
        if path.suffix in {".png", ".jpg", ".jpeg", ".gif", ".wav", ".mp3"}:
            continue

        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in content for marker in MERGE_MARKERS):
            offenders.append(rel_path)

    assert not offenders, f"Merge conflict markers found in: {', '.join(offenders)}"
