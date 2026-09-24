"""Regression tests for current setuptools license metadata.

Setuptools 77+ deprecates the old ``project.license`` table and license
classifiers in favor of an SPDX string plus explicit license files. Builds
used to emit deprecation warnings; keep the package metadata modern so a
future setuptools release does not turn that warning into a hard failure.
"""
from __future__ import annotations

from pathlib import Path


def _pyproject_text() -> str:
    return (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()


def test_license_metadata_uses_spdx_string_not_deprecated_table():
    text = _pyproject_text()

    assert 'license = "MIT"' in text
    assert 'license = { text = "MIT" }' not in text
    assert 'license-files = ["LICENSE"]' in text


def test_deprecated_license_classifier_is_not_present():
    text = _pyproject_text()

    assert "License :: OSI Approved :: MIT License" not in text


def test_setuptools_floor_supports_spdx_license_metadata():
    text = _pyproject_text()

    assert 'requires = ["setuptools>=77", "wheel"]' in text
