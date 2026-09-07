"""Check that the repository contains only the intended public research surface."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "data",
    "docs",
    "jobs",
    "notebooks",
    "pipelines",
    "reports",
    "results",
}
EXCLUDED_SUFFIXES = {
    ".csv",
    ".docx",
    ".feather",
    ".ipynb",
    ".parquet",
    ".pdf",
    ".pptx",
    ".zip",
}


def _restricted_phrases() -> tuple[str, ...]:
    parts = (
        "stablecoin" + "-ews-thesis",
        "data" + "bricks",
        "coingecko_pro" + "_api_key",
        "fred" + "_api_key",
        "stablecoin_ews_v4" + "_live_pilot",
        "authorize g" + "1 one run",
        "authorize g" + "2 one run",
        "authorize g" + "4",
        "/content/" + "drive",
        "live-permit" + "-journals",
    )
    return tuple(value.casefold() for value in parts)


def _tracked_files() -> list[Path]:
    process = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if process.returncode == 0 and process.stdout:
        return [
            ROOT / os.fsdecode(name) for name in process.stdout.split(b"\0") if name
        ]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(
            part in EXCLUDED_DIRECTORIES for part in path.relative_to(ROOT).parts
        )
    ]


def main() -> int:
    issues: list[str] = []
    files = _tracked_files()
    markdown = [path for path in files if path.suffix.casefold() == ".md"]
    if [path.name for path in markdown] != ["README.md"]:
        issues.append("README.md must be the only Markdown file")
    email_pattern = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
    for path in files:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            issues.append(f"symbolic link: {relative}")
        if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
            issues.append(f"excluded directory: {relative}")
        if path.suffix.casefold() in EXCLUDED_SUFFIXES:
            issues.append(f"excluded file type: {relative}")
        if path.resolve() == SELF or path.suffix.casefold() not in {
            ".cff",
            ".py",
            ".toml",
            ".txt",
            ".yml",
            ".yaml",
            ".md",
            "",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        folded = text.casefold()
        for phrase in _restricted_phrases():
            if phrase in folded:
                issues.append(f"restricted internal reference in {relative}")
        if email_pattern.search(text):
            issues.append(f"email address in {relative}")
    git_directory = ROOT / ".git"
    if git_directory.exists():
        count = subprocess.run(
            ["git", "-C", str(ROOT), "rev-list", "--count", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if count.returncode == 0 and int(count.stdout.strip()) > 1:
            issues.append("repository history contains more than one commit")
    if issues:
        for issue in sorted(set(issues)):
            print(f"FAIL: {issue}")
        return 1
    print(f"Publication check passed for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
