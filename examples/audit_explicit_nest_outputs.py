"""Read-only audit of completed compact explicit-NEST outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]["reconstruction"]
    assert manifest["reconstruction_method"] == "explicit_nest"
    assert not manifest["saved_nests"]
    rec = pd.read_csv(directory / "reconstruction.csv")
    assert rec.Year.is_unique and rec.Year.is_monotonic_increasing
    if config["retain_longest_annual_segment"]:
        assert rec.Year.diff().dropna().eq(1).all()
        assert np.isfinite(rec["median"]).all()
    assert len(rec) == manifest["annual_output_year_count"]
    assert int(rec.Year.min()) == manifest["annual_output_start"]
    assert int(rec.Year.max()) == manifest["annual_output_end"]
    nests = pd.read_csv(directory / "nest_summary.csv")
    coverage = nests.loc[nests.row_type.eq("coverage_nest")]
    resolution = nests.loc[nests.row_type.eq("resolution_subnest")]
    assert coverage.pids.is_unique and coverage.nest_id.is_unique
    assert len(coverage) == manifest["candidate_nest_count"]
    assert int(coverage.accepted.sum()) == manifest["accepted_nest_count"]
    accepted = nests.loc[nests.accepted.eq(True)]
    if config["nest"]["require_internal_ce_re"] and config["auto_tune"]:
        for metric in ("ce", "re"):
            threshold = config[f"min_{metric}"]
            if threshold is None:
                threshold = config["skill_floor"]
            values = accepted[f"internal_median_{metric}"]
            if config.get("internal_ce_re_comparison", "ge") == "gt":
                assert values.gt(threshold).all()
            else:
                assert values.ge(threshold).all()
    screening = pd.read_csv(directory / "proxy_screening.csv")
    selected = screening.loc[screening.selected.eq(True)].set_index("pid")
    assert len(selected) == manifest["selected_proxy_count"]
    weights = pd.read_csv(directory / "proxy_weights.csv")
    native = weights.loc[weights.nest_layer.eq("resolution_subnest")]
    native_pids = set()
    for _, row in resolution.loc[resolution.accepted.eq(True)].iterrows():
        fitted = native.loc[
            native.nest_id.eq(row.nest_id)
            & native.resolution_years.eq(row.resolution_years), "pid"
        ]
        # The earliest development output predates explicit fitted-pid audit
        # columns. Its retained PCA weight rows still identify actual members.
        declared = set(str(row.native_pids).split("|"))
        pids = declared.intersection(set(fitted))
        assert pids and pids.issubset(set(fitted))
        assert pids.issubset(set(selected.index))
        if "fitted_native_pids" in row:
            assert pids == set(str(row.fitted_native_pids).split("|"))
            assert row.fitted_native_proxy_count == len(pids)
        native_pids.update(pids)
    interpolation = pd.read_csv(directory / "proxy_interpolation_audit.csv")
    filled = interpolation.loc[interpolation.interpolated_year_count.gt(0)]
    assert not interpolation.endpoint_extrapolation.any()
    assert set(filled.archive).issubset(set(config["interpolation_archives"]))
    assert not set(filled.pid).intersection(native_pids)
    assert int(interpolation.interpolated_year_count.sum()) == manifest["interpolated_year_value_count"]
    assert not list(directory.glob("*.xlsx")) and not list(directory.glob("*.npy"))
    fit = pd.read_csv(directory / "observation_fit.csv").iloc[0]
    return {
        "directory": str(directory),
        "selected_proxies": len(selected), "coverage_nests": len(coverage),
        "accepted_coverage_nests": int(coverage.accepted.sum()),
        "accepted_resolution_subnests": int(resolution.accepted.sum()),
        "native_resolution_proxy_count_in_pca": len(native_pids),
        "native_resolution_pids_in_pca": sorted(native_pids),
        "annual_start": int(rec.Year.min()), "annual_end": int(rec.Year.max()),
        "r": float(fit.r), "sd_ratio": float(fit.sd_ratio),
        "lowpass_10y_r": float(fit.lowpass_10y_r),
        "lowpass_10y_sd_ratio": float(fit.lowpass_10y_sd_ratio),
        "interpolated_year_values": int(interpolation.interpolated_year_count.sum()),
        "output_bytes": sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
        "audit_passed": True,
    }


def compare(first: Path, second: Path) -> dict:
    for name in ("reconstruction", "nest_summary", "model_selection", "proxy_weights"):
        old = pd.read_csv(first / f"{name}.csv")
        new = pd.read_csv(second / f"{name}.csv")
        pd.testing.assert_frame_equal(old, new, check_exact=False, atol=1e-10, rtol=1e-10)
    old = pd.read_csv(first / "reconstruction.csv")
    new = pd.read_csv(second / "reconstruction.csv")
    return {
        "first": str(first), "second": str(second),
        "reconstruction_and_model_audits_equal": True,
        "maximum_annual_prediction_difference": float((old["median"] - new["median"]).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    result = {"audits": [audit(path) for path in args.directories]}
    if args.compare:
        result["comparison"] = compare(*args.compare)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
