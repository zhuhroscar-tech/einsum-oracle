"""Named einsum expression + shape fixtures for regression testing.

`einsum32111_optimal_bug` reproduces the exact case from the numpy issue
tracker (numpy/numpy#32111, closed as a duplicate of the still-open
numpy/numpy#11825): numpy's own `optimize='optimal'` search space excludes
a dramatically better contraction order for this shape family. Verified
independently reproducible on numpy 2.5.2 (2026-09-17) -- see README for
the exact command used and the resulting numpy-vs-oracle ratio observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .oracle import Shape


@dataclass(frozen=True)
class Fixture:
    name: str
    subscripts: str
    shapes: Tuple[Shape, ...]
    note: str


_FIXTURES: Dict[str, Fixture] = {}


def _register(fixture: Fixture) -> None:
    _FIXTURES[fixture.name] = fixture


_register(
    Fixture(
        name="chain_matmul_4",
        subscripts="ab,bc,cd,de->ae",
        shapes=((8, 30), (30, 4), (4, 25), (25, 6)),
        note="classic matrix-chain-multiplication warm-up case; numpy is optimal here.",
    )
)

_register(
    Fixture(
        name="einsum32111_optimal_bug_small",
        subscripts="ijk,klmn,nop,jmp->ilo",
        # Same shape family and label structure as numpy/numpy#32111's
        # repro, scaled down (r=15, n=50 instead of r=150, n=5000) so the
        # exact DP oracle and the numpy path check both run in well under a
        # second in CI, while still triggering the same missed-optimal-path
        # behavior (the defect is structural, not a large-N-only effect --
        # see README for the full-scale numbers from the original issue).
        shapes=((15, 15, 2), (2, 50, 50, 2), (2, 15, 15), (15, 50, 15)),
        note=(
            "small-scale reproduction of numpy/numpy#32111 (dup of #11825): "
            "numpy optimize='optimal' misses a far cheaper contraction order "
            "for this 4-operand shape family. Verified against numpy 2.5.2."
        ),
    )
)

_register(
    Fixture(
        name="einsum32111_optimal_bug_original_scale",
        subscripts="ijk,klmn,nop,jmp->ilo",
        # The exact shapes from the original issue report. Not run by
        # default CI (the *plan* is instant, but building the DP's future-
        # label bookkeeping is still fine at n=4 operands -- this fixture is
        # kept for documentation/manual reproduction of the literal
        # headline numbers from the GitHub issue, run via
        # `einsum-oracle check --fixture einsum32111_optimal_bug_original_scale`).
        shapes=((150, 150, 2), (2, 5000, 5000, 2), (2, 150, 150), (150, 5000, 150)),
        note="original numpy/numpy#32111 issue scale (r=150, n=5000).",
    )
)

_register(
    Fixture(
        name="star_contraction_5",
        subscripts="ab,bc,bd,be,bf->acdef",
        shapes=((3, 20), (20, 4), (20, 5), (20, 6), (20, 7)),
        note="5-operand star contraction sharing one index; exercises the DP's O(3^n) path at a still-tiny n.",
    )
)


def list_fixtures(include_large: bool = False):
    """Return fixture names for default use. `original_scale` fixtures are
    excluded by default (they allocate real, sizable arrays via
    `np.empty` -- ~800MB for `einsum32111_optimal_bug_original_scale` --
    which is unnecessary memory pressure for routine CI/default `check`
    runs since the small-scale fixture already reproduces the identical
    defect pattern). Pass include_large=True or reference the fixture by
    exact name to run it anyway.
    """
    if include_large:
        return sorted(_FIXTURES)
    return sorted(name for name in _FIXTURES if "original_scale" not in name)


def list_all_fixtures():
    return sorted(_FIXTURES)


def get_fixture(name: str) -> Fixture:
    if name not in _FIXTURES:
        raise KeyError(f"unknown fixture {name!r}; available: {list_fixtures()}")
    return _FIXTURES[name]
