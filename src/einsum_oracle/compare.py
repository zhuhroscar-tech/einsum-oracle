"""Compare numpy's `einsum_path` FLOP estimate against the exact DP oracle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .oracle import Shape, optimal_contraction

# numpy's "Optimized FLOP count" line in the einsum_path description; we
# parse it directly rather than reimplementing numpy's own internal cost
# accounting, so a change to that accounting is visible in our tests too.
_FLOP_LINE_PREFIX = "Optimized FLOP count"


@dataclass(frozen=True)
class CheckResult:
    subscripts: str
    shapes: Tuple[Shape, ...]
    numpy_flops: float
    oracle_flops: int
    ratio: float
    verdict: str  # "optimal" | "suboptimal" | "unknown"
    numpy_path: object
    numpy_description: str

    def to_dict(self) -> dict:
        return {
            "subscripts": self.subscripts,
            "shapes": list(self.shapes),
            "numpy_flops": self.numpy_flops,
            "oracle_flops": self.oracle_flops,
            "ratio": self.ratio,
            "verdict": self.verdict,
        }


def _parse_numpy_flops(description: str) -> float:
    for line in description.splitlines():
        if _FLOP_LINE_PREFIX in line:
            return float(line.split(":")[1].strip())
    raise ValueError(
        f"could not find {_FLOP_LINE_PREFIX!r} in numpy's einsum_path output"
    )


def check_path(
    subscripts: str,
    shapes: Sequence[Shape],
    optimize: str = "optimal",
    suboptimal_ratio_threshold: float = 8.0,
) -> CheckResult:
    """Run numpy.einsum_path and compare its reported FLOP count against the
    exact DP oracle for the same expression and shapes.

    `suboptimal_ratio_threshold`: a ratio (numpy_flops / oracle_flops) above
    this is flagged "suboptimal". Empirically, numpy's own "Optimized FLOP
    count" uses a multiply-accumulate convention that runs a small constant
    factor (observed ~1.6x-2.0x on every genuinely-optimal case in
    tests/test_oracle.py) above this oracle's idealized multiply count --
    that is baseline noise from a cost-model convention difference, not a
    search defect. A real missed-optimal-path defect (as in
    numpy/numpy#11825 / #32111) produces ratios of 10^3-10^5x, so an 8x
    threshold cleanly separates convention noise from an actual miss with
    wide margin in both directions.
    """

    arrays = [np.empty(shape) for shape in shapes]
    path, description = np.einsum_path(subscripts, *arrays, optimize=optimize)
    numpy_flops = _parse_numpy_flops(description)

    plan = optimal_contraction(subscripts, shapes)
    oracle_flops = plan.total_flops

    if oracle_flops == 0:
        ratio = float("inf") if numpy_flops > 0 else 1.0
    else:
        ratio = numpy_flops / oracle_flops

    verdict = "suboptimal" if ratio > suboptimal_ratio_threshold else "optimal"

    return CheckResult(
        subscripts=subscripts,
        shapes=tuple(shapes),
        numpy_flops=numpy_flops,
        oracle_flops=oracle_flops,
        ratio=ratio,
        verdict=verdict,
        numpy_path=path,
        numpy_description=description,
    )
