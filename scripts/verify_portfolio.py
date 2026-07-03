#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
# ─── How to run ───
# python scripts/verify_portfolio.py
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
MAX_TRACKED_FILE_BYTES: Final = 50 * 1024 * 1024
REQUIRED_PATHS: Final = (
    "README.md",
    "requirements.txt",
    "docs/public-safety.md",
    "docs/verification-report.md",
)
README_MARKERS: Final = ("##", "재현", "공개")


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_required_paths() -> CheckResult:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    if missing:
        return CheckResult("required paths", False, ", ".join(missing))
    return CheckResult("required paths", True, "all required files exist")


def check_readme_markers() -> CheckResult:
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = [marker for marker in README_MARKERS if marker not in readme_text]
    if missing:
        return CheckResult("README markers", False, ", ".join(missing))
    return CheckResult("README markers", True, "README includes review markers")


def tracked_files() -> list[Path]:
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    except subprocess.CalledProcessError:
        return []
    return [ROOT / line for line in completed.stdout.splitlines() if line]


def check_tracked_file_sizes() -> CheckResult:
    oversized = [
        path.relative_to(ROOT).as_posix()
        for path in tracked_files()
        if path.exists() and path.stat().st_size > MAX_TRACKED_FILE_BYTES
    ]
    if oversized:
        return CheckResult("tracked file size", False, ", ".join(oversized))
    return CheckResult("tracked file size", True, "no tracked file exceeds 50MB")


def main() -> int:
    results = (
        check_required_paths(),
        check_readme_markers(),
        check_tracked_file_sizes(),
    )
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
