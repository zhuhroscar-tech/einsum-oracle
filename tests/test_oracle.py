"""Tests for the exact DP contraction-order oracle."""
import pytest

from einsum_oracle.oracle import (
    TooManyOperandsError,
    optimal_contraction,
    parse_subscripts,
)


def test_parse_subscripts_basic():
    labels, output = parse_subscripts("ij,jk->ik")
    assert labels == [("i", "j"), ("j", "k")]
    assert output == ("i", "k")


def test_parse_subscripts_requires_explicit_output():
    with pytest.raises(ValueError, match="explicit"):
        parse_subscripts("ij,jk")


def test_parse_subscripts_rejects_ellipsis():
    with pytest.raises(ValueError, match="ellipsis"):
        parse_subscripts("i...,i...->i...")


def test_two_operand_matmul_is_trivial():
    plan = optimal_contraction("ij,jk->ik", [(3, 4), (4, 5)])
    assert plan.total_flops == 3 * 4 * 5
    assert len(plan.steps) == 1


def test_conflicting_label_sizes_rejected():
    with pytest.raises(ValueError, match="conflicting sizes"):
        optimal_contraction("ij,jk->ik", [(3, 4), (5, 6)])


def test_label_shape_length_mismatch_rejected():
    with pytest.raises(ValueError, match="do not match shape"):
        optimal_contraction("ijk,jk->ik", [(3, 4), (4, 5)])


def test_shape_count_mismatch_rejected():
    with pytest.raises(ValueError, match="expected 2 shapes"):
        optimal_contraction("ij,jk->ik", [(3, 4)])


def test_too_many_operands_raises():
    subscripts = ",".join(f"x{i}" for i in range(13)) + "->x0"
    shapes = [(2,)] * 13
    with pytest.raises(TooManyOperandsError):
        optimal_contraction(subscripts, shapes)


def _brute_force_min_flops(operand_labels, shapes, output_labels):
    """Independent reference for the DP under test: naive (non-memoized)
    recursive exhaustive search over every way to bipartition the full
    operand set into two non-empty groups, recursing into each side. This
    explores the SAME full binary-tree search space as the bitmask DP
    (unlike a left-to-right permutation fold, which only reaches "caterpillar"
    trees and would understate the true optimum for any contraction where
    a non-adjacent/balanced grouping is cheaper) but is written as a
    completely separate, non-memoized, index-set-based implementation with
    no shared code path -- a real implementation bug in the DP's subset
    bookkeeping, future-label logic, or flop accounting would very likely
    disagree with this independent recursion, even though both explore the
    same combinatorial space.

    Key fact used here: the labels that must survive contracting any subset
    S of the operands are exactly (output_labels UNION labels appearing
    outside S in the ORIGINAL full problem) intersected with S's own
    labels -- this holds regardless of what tree shape is chosen to
    contract S internally, so it can be precomputed per-subset without
    threading state through the recursion.
    """
    n = len(operand_labels)
    dim = {}
    for labs, shp in zip(operand_labels, shapes):
        for l, s in zip(labs, shp):
            dim[l] = s
    label_sets = [frozenset(labs) for labs in operand_labels]
    output_set = frozenset(output_labels)
    full = frozenset(range(n))

    def labels_of(index_set):
        s = set()
        for i in index_set:
            s |= label_sets[i]
        return s

    def surviving_labels(index_set):
        outside = full - index_set
        needed = output_set | labels_of(outside)
        return labels_of(index_set) & needed

    def solve(index_set):
        if len(index_set) == 1:
            return 0
        items = sorted(index_set)
        m = len(items)
        best = None
        for mask in range(1, 1 << (m - 1)):
            left = frozenset(items[i] for i in range(m) if mask & (1 << i))
            right = index_set - left
            cost1 = solve(left)
            cost2 = solve(right)
            combined = surviving_labels(left) | surviving_labels(right)
            flops = 1
            for l in combined:
                flops *= dim[l]
            total = cost1 + cost2 + flops
            if best is None or total < best:
                best = total
        return best

    return solve(full)


@pytest.mark.parametrize(
    "subscripts,shapes",
    [
        ("ab,bc,cd,de->ae", ((8, 30), (30, 4), (4, 25), (25, 6))),
        ("ij,jk,kl->il", ((10, 2), (2, 20), (20, 3))),
        ("ab,bc,bd,be,bf->acdef", ((3, 20), (20, 4), (20, 5), (20, 6), (20, 7))),
    ],
)
def test_dp_matches_independent_brute_force(subscripts, shapes):
    operand_labels, output_labels = parse_subscripts(subscripts)
    expected = _brute_force_min_flops(operand_labels, shapes, output_labels)
    plan = optimal_contraction(subscripts, shapes)
    assert plan.total_flops == expected


def test_reconstructed_steps_cover_every_operand_exactly_once():
    plan = optimal_contraction(
        "ab,bc,cd,de->ae", ((8, 30), (30, 4), (4, 25), (25, 6))
    )
    n = len(plan.shapes)
    seen = set()
    for left, right in plan.steps:
        assert left.isdisjoint(right)
    # the final step's union must be all operand indices
    last_left, last_right = plan.steps[-1]
    assert (last_left | last_right) == frozenset(range(n))
