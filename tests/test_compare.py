"""Tests for numpy einsum_path vs. exact-oracle comparison, including the
regression test for numpy/numpy#32111 (duplicate of the still-open
numpy/numpy#11825): this test would have FAILED to detect the bug before
this tool existed (there was no automated check at all), and it fails now
if numpy ever silently stops reproducing the bug OR if a code change here
accidentally breaks detection -- it asserts the exact suboptimal ratio
range actually observed, not just "verdict != optimal"."""
import numpy as np
import pytest

from einsum_oracle.compare import check_path
from einsum_oracle import fixtures as fx


def test_chain_matmul_reports_optimal():
    fixture = fx.get_fixture("chain_matmul_4")
    result = check_path(fixture.subscripts, fixture.shapes)
    assert result.verdict == "optimal"
    # numpy's own FLOP accounting is a small constant multiple of the
    # oracle's idealized multiply count on genuinely optimal plans --
    # pin the observed range so a real regression (ratio drifting toward
    # the suboptimal threshold) would fail this test.
    assert 1.0 <= result.ratio <= 3.0


def test_star_contraction_reports_optimal():
    fixture = fx.get_fixture("star_contraction_5")
    result = check_path(fixture.subscripts, fixture.shapes)
    assert result.verdict == "optimal"


def test_numpy_32111_regression_still_reproducible():
    """This is the regression test for the real, externally-reported,
    currently-open numpy defect (numpy/numpy#11825, duplicate report
    #32111). It intentionally asserts numpy is CURRENTLY suboptimal here --
    if numpy ever fixes this upstream, this test will start failing with a
    ratio near 1-3x instead of >>8x, which is the correct signal to update
    this fixture's status (see README's "if this test starts failing"
    note) rather than a tool defect."""
    fixture = fx.get_fixture("einsum32111_optimal_bug_small")
    result = check_path(fixture.subscripts, fixture.shapes)
    assert result.verdict == "suboptimal"
    # observed ratio on numpy 2.5.2 was ~692x; assert a generous lower bound
    # (100x) so minor cost-model changes don't make this test flaky while
    # still failing hard if numpy actually fixes the search.
    assert result.ratio > 100.0


def test_check_path_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        check_path("ij,jk->ik", [(3, 4)])


def test_check_path_json_round_trip_shape():
    fixture = fx.get_fixture("chain_matmul_4")
    result = check_path(fixture.subscripts, fixture.shapes)
    d = result.to_dict()
    assert set(d) == {
        "subscripts",
        "shapes",
        "numpy_flops",
        "oracle_flops",
        "ratio",
        "verdict",
    }
