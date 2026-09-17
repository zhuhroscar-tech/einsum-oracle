"""CLI-level tests: argument parsing, exit codes, JSON well-formedness."""
import json

import pytest

from einsum_oracle.cli import main


def test_check_all_fixtures_json_is_well_formed(capsys):
    exit_code = main(["check", "--json", "--no-color"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    assert len(payload) >= 3
    # the known-bug fixture should be in there and flagged suboptimal
    names = {row["fixture"] for row in payload}
    assert "einsum32111_optimal_bug_small" in names
    bug_row = next(r for r in payload if r["fixture"] == "einsum32111_optimal_bug_small")
    assert bug_row["verdict"] == "suboptimal"
    # overall exit code must be nonzero since at least one fixture is flagged
    assert exit_code == 1


def test_check_single_optimal_fixture_exits_zero(capsys):
    exit_code = main(["check", "--fixture", "chain_matmul_4", "--no-color"])
    captured = capsys.readouterr()
    assert "optimal" in captured.out
    assert exit_code == 0


def test_list_fixtures_runs(capsys):
    exit_code = main(["list-fixtures"])
    captured = capsys.readouterr()
    assert "einsum32111_optimal_bug_small" in captured.out
    assert "einsum32111_optimal_bug_original_scale" in captured.out
    assert "not included in default" in captured.out
    assert exit_code == 0


def test_unknown_fixture_reports_error_not_crash(capsys):
    exit_code = main(["check", "--fixture", "does-not-exist", "--json", "--no-color"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload[0]["fixture"] == "does-not-exist"
    assert "error" in payload[0]
    assert exit_code == 1


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "einsum-oracle" in captured.out
