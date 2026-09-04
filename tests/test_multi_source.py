"""Round-trip tests for the two-stage ECR multi-source file format.

Verifies that ``sample_to_multi_source_files`` → ``_load_multi_source``
reconstructs a sample dict that is structurally identical to the original,
and that the solver path (``sample_to_instance`` → ``solve_dep``) yields the
same optimal objective.
"""

from __future__ import annotations

import math
import os

import pytest

from generator import build_smoke_export, sample_to_multi_source_files, sample_to_instance
from fsm_stackelberg.utils.utils import _load_multi_source


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def _norm_val(v):
    return round(v, 8) if isinstance(v, float) else v


def _norm_list(records):
    """Sort a list so order-independent comparison is possible."""
    if records and isinstance(records[0], dict):
        return sorted(
            tuple(sorted((k, _norm_val(val)) for k, val in r.items())) for r in records
        )
    return sorted(_norm_val(x) for x in records)


def assert_deep_eq(a, b, path="root"):
    if isinstance(a, dict) and isinstance(b, dict):
        ak, bk = set(a), set(b)
        assert ak == bk, f"{path}: keys differ only_a={ak - bk} only_b={bk - ak}"
        for k in a:
            assert_deep_eq(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: len {len(a)} vs {len(b)}"
        na, nb = _norm_list(a), _norm_list(b)
        assert na == nb, f"{path}: list content differs"
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        assert math.isclose(float(a), float(b), rel_tol=1e-9), f"{path}: {a} vs {b}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def smoke_sample():
    sample, _ = build_smoke_export(T=4, n_scenarios=5, seed=42)
    return sample


@pytest.fixture()
def multi_source_dir(smoke_sample, tmp_path):
    out = tmp_path / "instance"
    sample_to_multi_source_files(smoke_sample, out)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
EXPECTED_FILES = [
    "nodes.csv",
    "arcs.csv",
    "lambda_hl.csv",
    "vessel_calls.csv",
    "supply_demand.csv",
    "scenarios.csv",
    "scenario_probability.csv",
    "inventory.csv",
    "transport_modes.csv",
    "first_stage.json",
    "metadata.json",
]


def test_all_expected_files_written(multi_source_dir):
    written = set(os.listdir(multi_source_dir))
    missing = set(EXPECTED_FILES) - written
    assert not missing, f"missing files: {missing}"


def test_round_trip_dict_identity(smoke_sample, multi_source_dir):
    """Loaded dict must be structurally identical to the original sample."""
    loaded = _load_multi_source(str(multi_source_dir))
    assert loaded is not None, "loader returned None for a multi-source dir"
    assert_deep_eq(smoke_sample, loaded)


def test_loader_returns_none_without_metadata(tmp_path):
    """A directory without metadata.json must signal fallback (return None)."""
    assert _load_multi_source(str(tmp_path)) is None


def test_loader_returns_none_for_legacy_metadata(tmp_path):
    """A metadata.json without the ``sets`` contract must also fall back."""
    import json

    (tmp_path / "metadata.json").write_text(
        json.dumps({"problem": "tslp_ecr_demand", "objective": 1.0})
    )
    assert _load_multi_source(str(tmp_path)) is None


def test_solver_round_trip(smoke_sample, multi_source_dir):
    """The solver must reach the same objective from the reconstructed dict."""
    from tslp_ecr_demand.dep import solve_dep

    _, optimal = build_smoke_export(T=4, n_scenarios=5, seed=42)
    loaded = _load_multi_source(str(multi_source_dir))
    assert loaded is not None

    inst, scenarios, probs = sample_to_instance(loaded)
    result = solve_dep(inst, scenarios, probs)
    assert result.status == optimal["status"]
    assert math.isclose(result.objective, optimal["objective"], rel_tol=1e-6)
