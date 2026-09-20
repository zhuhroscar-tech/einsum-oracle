"""Exact optimal-contraction-order oracle, independent of numpy/opt_einsum.

Computes the TRUE minimum total FLOP count for evaluating an einsum
expression, via exhaustive bitmask dynamic programming over all subsets of
operands (the standard exact algorithm for optimal matrix-chain / tensor-
contraction ordering: O(3^n) subset-pair enumeration, exact for n up to
about 12-14 operands in practice).

This exists to let a path-optimizer's claimed "optimal" plan be checked
against ground truth, catching cases where the optimizer's search space or
heuristic misses the true minimum (see numpy/numpy#11825 / #32111, an open
bug where numpy's own `optimize='optimal'` search space excludes better
orderings for certain shapes -- verified independently reproducible against
several numpy versions as of 2026-09, see README).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence, Tuple

Shape = Tuple[int, ...]
Labels = Tuple[str, ...]


class TooManyOperandsError(ValueError):
    """Raised when the exact DP would be computationally infeasible."""


@dataclass(frozen=True)
class ContractionPlan:
    """Result of the exact oracle: total FLOPs and one optimal binary
    contraction order, expressed as a sequence of (left_indices, right_indices)
    subset merges over the original operand indices."""

    total_flops: int
    steps: Tuple[Tuple[FrozenSet[int], FrozenSet[int]], ...]
    subscripts: str
    shapes: Tuple[Shape, ...]


def parse_subscripts(subscripts: str) -> Tuple[List[Labels], Labels]:
    """Split 'ij,jk->ik' into ([('i','j'), ('j','k')], ('i','k')).

    Only explicit (non-implicit-output, non-ellipsis) subscript strings are
    supported -- this oracle targets exact verification of specific shapes,
    not full einsum-syntax coverage.
    """
    if "->" not in subscripts:
        raise ValueError(
            "einsum-oracle requires an explicit '->output' subscript "
            "(implicit-output inference is out of scope for this oracle)"
        )
    if "." in subscripts:
        raise ValueError("ellipsis ('...') subscripts are not supported")
    lhs, rhs = subscripts.split("->")
    operand_labels = [tuple(s.strip()) for s in lhs.split(",")]
    output_labels = tuple(rhs.strip())

    # Match numpy.einsum's own explicit-output validation: every output
    # label must actually appear in at least one input, and no output
    # label may repeat. Without this, optimal_contraction() below silently
    # produces a "plan" for an expression numpy itself would reject
    # (an absent output label was treated as contributing zero FLOPs and
    # vanishing from the result entirely; a repeated output label such as
    # '->ii' hit the popcount==1 trivial base case and returned cost 0 with
    # no validation at all) -- a structurally wrong, undetectable-looking
    # answer rather than an honest error.
    input_label_set = {label for labels in operand_labels for label in labels}
    seen_output_labels: set = set()
    for label in output_labels:
        if label not in input_label_set:
            raise ValueError(
                f"output subscript label {label!r} never appears in any "
                f"input operand of {subscripts!r} (numpy.einsum rejects "
                f"this too)"
            )
        if label in seen_output_labels:
            raise ValueError(
                f"output subscript label {label!r} appears more than once "
                f"in {subscripts!r} (numpy.einsum rejects a repeated "
                f"output label too)"
            )
        seen_output_labels.add(label)

    return operand_labels, output_labels


def _dim_map(operand_labels: Sequence[Labels], shapes: Sequence[Shape]) -> Dict[str, int]:
    dim: Dict[str, int] = {}
    for labels, shape in zip(operand_labels, shapes):
        if len(labels) != len(shape):
            raise ValueError(
                f"labels {labels!r} do not match shape {shape!r} in length"
            )
        for label, size in zip(labels, shape):
            if label in dim and dim[label] != size:
                raise ValueError(
                    f"label {label!r} has conflicting sizes {dim[label]} "
                    f"and {size} across operands"
                )
            dim[label] = size
    return dim


def optimal_contraction(
    subscripts: str, shapes: Sequence[Shape], max_operands: int = 12
) -> ContractionPlan:
    """Exact bitmask-DP solution for the true minimum-FLOP contraction order.

    Cost model: for each pairwise merge, FLOPs = product of the sizes of
    every distinct label appearing in either operand (the standard
    multiply-accumulate cost model used by numpy's own einsum_path
    docstring and by opt_einsum). This matches numpy's "Optimized FLOP
    count" convention closely enough for direct ratio comparison (see
    tests/test_oracle.py for a cross-check against numpy's own reported
    FLOP counts on cases where numpy IS optimal).

    Raises TooManyOperandsError above `max_operands` (default 12): the DP is
    O(3^n), so it stays fast (well under a second) through n=12 but becomes
    impractical beyond that on typical hardware -- this is a correctness
    oracle for spot-checking specific expressions, not a production planner.
    """
    operand_labels, output_labels = parse_subscripts(subscripts)
    n = len(operand_labels)
    if n == 0:
        raise ValueError("at least one operand is required")
    if len(shapes) != n:
        raise ValueError(f"expected {n} shapes for {n} operands, got {len(shapes)}")
    if n > max_operands:
        raise TooManyOperandsError(
            f"{n} operands exceeds max_operands={max_operands}; exact DP is "
            f"O(3^n) and becomes impractically slow beyond this size"
        )

    dim = _dim_map(operand_labels, shapes)
    full_mask = (1 << n) - 1
    label_sets = [frozenset(labels) for labels in operand_labels]
    output_set = frozenset(output_labels)

    def labels_in_mask(mask: int) -> FrozenSet[str]:
        s: set = set()
        for i in range(n):
            if mask & (1 << i):
                s |= label_sets[i]
        return frozenset(s)

    # memo[mask] = (best_cost, best_result_labels, best_split_submask_or_None)
    memo: Dict[int, Tuple[int, FrozenSet[str], int]] = {}

    def solve(mask: int) -> Tuple[int, FrozenSet[str]]:
        if mask in memo:
            cost, labels, _ = memo[mask]
            return cost, labels
        popcount = bin(mask).count("1")
        if popcount == 1:
            i = mask.bit_length() - 1
            memo[mask] = (0, label_sets[i], -1)
            return 0, label_sets[i]

        outside_mask = full_mask ^ mask
        future_needed = output_set | labels_in_mask(outside_mask)

        best_cost = None
        best_labels: FrozenSet[str] = frozenset()
        best_split = -1
        # Enumerate every way to split `mask` into two non-empty disjoint
        # submasks (each unordered pair counted once via sub < other).
        sub = (mask - 1) & mask
        while sub > 0:
            other = mask ^ sub
            if sub < other:
                cost1, labels1 = solve(sub)
                cost2, labels2 = solve(other)
                combined = labels1 | labels2
                result_labels = frozenset(l for l in combined if l in future_needed)
                flops = 1
                for label in combined:
                    flops *= dim[label]
                total = cost1 + cost2 + flops
                if best_cost is None or total < best_cost:
                    best_cost = total
                    best_labels = result_labels
                    best_split = sub
            sub = (sub - 1) & mask

        assert best_cost is not None
        memo[mask] = (best_cost, best_labels, best_split)
        return best_cost, best_labels

    total_cost, _ = solve(full_mask)

    # Reconstruct one optimal step sequence by walking the memo splits.
    steps: List[Tuple[FrozenSet[int], FrozenSet[int]]] = []

    def index_set(mask: int) -> FrozenSet[int]:
        return frozenset(i for i in range(n) if mask & (1 << i))

    def reconstruct(mask: int) -> None:
        if bin(mask).count("1") == 1:
            return
        _, _, split = memo[mask]
        other = mask ^ split
        reconstruct(split)
        reconstruct(other)
        steps.append((index_set(split), index_set(other)))

    reconstruct(full_mask)

    return ContractionPlan(
        total_flops=total_cost,
        steps=tuple(steps),
        subscripts=subscripts,
        shapes=tuple(shapes),
    )
