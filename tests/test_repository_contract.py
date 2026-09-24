"""Repository-level completeness contracts.

These tests protect the public project scaffolding that users rely on but
that pure unit tests do not exercise: required files, README links, release
history, CI artifact generation, CodeQL, and package/runtime version parity.
"""
from __future__ import annotations

import re
from pathlib import Path

from einsum_oracle import __version__

ROOT = Path(__file__).resolve().parent.parent


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_required_project_files_exist():
    for relative in (
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "MANIFEST.in",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        ".github/workflows/codeql.yml",
    ):
        assert (ROOT / relative).is_file(), f"missing required project file: {relative}"


def test_readme_links_license_and_release_history():
    readme = _read("README.md")

    assert "[MIT](LICENSE)" in readme
    assert "[CHANGELOG.md](CHANGELOG.md)" in readme


def test_changelog_documents_current_version():
    changelog = _read("CHANGELOG.md")

    assert f"## v{__version__}" in changelog
    assert "## v0.1.0" in changelog


def test_ci_builds_distribution_artifacts_and_checksums():
    ci = _read(".github/workflows/ci.yml")

    assert "python -m build" in ci
    assert "sha256sum * > SHA256SUMS.txt" in ci
    assert "actions/upload-artifact@v4" in ci
    assert "path: dist/" in ci


def test_codeql_scans_python_on_push_and_schedule():
    codeql = _read(".github/workflows/codeql.yml")

    assert "github/codeql-action/init@v3" in codeql
    assert "github/codeql-action/analyze@v3" in codeql
    assert "languages: python" in codeql
    assert "schedule:" in codeql


def test_package_version_matches_current_changelog_entry():
    pyproject = _read("pyproject.toml")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)

    assert match, 'missing project version in pyproject.toml'
    assert match.group(1) == __version__
    assert f"## v{match.group(1)}" in _read("CHANGELOG.md")
