"""einsum-oracle CLI: check numpy einsum_path plans against an exact DP oracle."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import __version__
from . import fixtures as fx
from .compare import check_path
from .style import Style, print_fields, resolve_style, section, status_headline


def cmd_check(args: argparse.Namespace) -> int:
    style = resolve_style(args.no_color)
    names = fx.list_fixtures() if args.fixture == "all" else [args.fixture]
    results = []
    for name in names:
        try:
            fixture = fx.get_fixture(name)
        except KeyError as exc:
            results.append((None, None, str(exc), name))
            continue
        try:
            result = check_path(
                fixture.subscripts,
                fixture.shapes,
                optimize=args.optimize,
                suboptimal_ratio_threshold=args.threshold,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced per-fixture, not fatal
            results.append((fixture, None, str(exc), name))
            continue
        results.append((fixture, result, None, name))

    any_suboptimal = any(
        r is not None and r.verdict == "suboptimal" for _, r, _, _ in results
    )
    any_error = any(err is not None for _, _, err, _ in results)

    if args.json:
        payload = []
        for fixture, result, err, name in results:
            row = {"fixture": name}
            if err is not None:
                row["error"] = err
            else:
                row.update(result.to_dict())
            payload.append(row)
        print(json.dumps(payload, indent=2))
    else:
        for fixture, result, err, name in results:
            if err is not None:
                print(status_headline(style, "warn", f"{name}: ERROR"))
                print_fields([("error", err)])
                print()
                continue
            level = "fail" if result.verdict == "suboptimal" else "ok"
            print(status_headline(style, level, f"{fixture.name}  [{result.verdict}]"))
            print_fields(
                [
                    ("subscripts", result.subscripts),
                    ("numpy FLOPs", f"{result.numpy_flops:.3e}"),
                    ("oracle FLOPs", f"{result.oracle_flops:.3e}"),
                    ("ratio (numpy/oracle)", f"{result.ratio:.2f}x"),
                    ("note", fixture.note),
                ]
            )
            print()
    return 1 if (any_suboptimal or any_error) else 0


def cmd_list_fixtures(args: argparse.Namespace) -> int:
    section("fixtures")
    for name in fx.list_all_fixtures():
        fixture = fx.get_fixture(name)
        default_marker = "" if name in fx.list_fixtures() else "  (not included in default 'check --fixture all'; large arrays)"
        print(f"  {name}{default_marker}")
        print(f"      {fixture.subscripts}  shapes={list(fixture.shapes)}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="einsum-oracle",
        description=(
            "Check numpy.einsum_path's chosen contraction order against an "
            "independent exact dynamic-programming oracle for the true "
            "minimum FLOP count."
        ),
    )
    parser.add_argument("--version", action="version", version=f"einsum-oracle {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser(
        "check", help="compare numpy's einsum_path plan against the exact oracle"
    )
    check_p.add_argument("--fixture", default="all", help="fixture name, or 'all' (default)")
    check_p.add_argument(
        "--optimize", default="optimal",
        help="value passed to numpy.einsum_path's optimize= argument (default: optimal)",
    )
    check_p.add_argument(
        "--threshold", type=float, default=8.0,
        help="ratio above which a plan is flagged suboptimal (default: 8.0)",
    )
    check_p.add_argument("--json", action="store_true")
    check_p.add_argument("--no-color", action="store_true")
    check_p.set_defaults(func=cmd_check)

    list_p = sub.add_parser("list-fixtures", help="list available fixtures")
    list_p.set_defaults(func=cmd_list_fixtures)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
