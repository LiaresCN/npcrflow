"""PDO ablation of proxy resolution and regression model, with no interpolation."""

from dataclasses import replace

import pandas as pd

from run_pdo_from_raw_dod2k import PROJECT, PROXIES, TARGET, build_config
from npcrflow import run_pipeline
from npcrflow.diagnostics import observation_fit_table


ROOT = PROJECT / "results" / "pdo_v030_low_frequency_benchmark"


EXPERIMENTS = (
    # Resolution ablation under the same Ridge estimator.
    ("resolution", "native_to_1p5y", "ridge", 1.5, False),
    ("resolution", "native_to_5y", "ridge", 5.0, False),
    ("resolution", "native_to_10y", "ridge", 10.0, False),
    ("resolution", "all_native", "ridge", None, False),
    ("resolution", "hybrid_10y_lowpass", "ridge", None, True),
    # Regression alternatives use the same complete native selected network.
    ("regression", "ridge", "ridge", None, False),
    ("regression", "pls", "pls", None, False),
    ("regression", "elasticnet", "elasticnet", None, False),
    ("regression", "random_forest", "random_forest", None, False),
)


def _metric(row: pd.Series, name: str) -> float:
    value = row.get(name, float("nan"))
    return float(value) if pd.notna(value) else float("nan")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    fold_rows: list[pd.DataFrame] = []
    for family, name, regression, maximum_resolution, multiresolution in EXPERIMENTS:
        config = build_config(
            output_directory=ROOT / name,
            amplitude_method="none",
            multiresolution_enabled=multiresolution,
        )
        config = replace(
            config,
            proxy_filter=replace(
                config.proxy_filter,
                maximum_resolution_years=maximum_resolution,
            ),
            reconstruction=replace(
                config.reconstruction,
                regression=regression,
                n_bootstrap=0,
            ),
            output=replace(
                config.output,
                save_proxy_map=False,
                save_observation_plot=False,
            ),
        )
        result = run_pipeline(PROXIES, TARGET, config)
        apparent = observation_fit_table(
            result.target,
            result.reconstruction.reconstruction,
            config.reconstruction,
        ).iloc[0]
        outer = result.reconstruction.validation_summary.iloc[0]
        selected = result.screening[result.screening["selected"]]
        roles = selected["reconstruction_role"].value_counts()
        rows.append(
            {
                "family": family,
                "experiment": name,
                "regression": regression,
                "maximum_resolution_years": maximum_resolution,
                "multiresolution": multiresolution,
                "selected_proxy_count": len(selected),
                "pcr_proxy_count": int(roles.get("pcr_native", 0)),
                "lowpass_proxy_count": int(roles.get("lowpass_native_window", 0)),
                "apparent_r": _metric(apparent, "r"),
                "apparent_sd_ratio": _metric(apparent, "sd_ratio"),
                "apparent_lowpass_10y_r": _metric(apparent, "lowpass_10y_r"),
                "apparent_lowpass_10y_sd_ratio": _metric(apparent, "lowpass_10y_sd_ratio"),
                "apparent_lowpass_20y_r": _metric(apparent, "lowpass_20y_r"),
                "apparent_lowpass_20y_sd_ratio": _metric(apparent, "lowpass_20y_sd_ratio"),
                "outer_median_r": _metric(outer, "median_r"),
                "outer_median_sd_ratio": _metric(outer, "median_sd_ratio"),
                "outer_median_lowpass_10y_r": _metric(outer, "median_lowpass_10y_r"),
                "outer_median_lowpass_10y_sd_ratio": _metric(
                    outer, "median_lowpass_10y_sd_ratio"
                ),
                "outer_median_lowpass_20y_r": _metric(outer, "median_lowpass_20y_r"),
                "outer_median_lowpass_20y_sd_ratio": _metric(
                    outer, "median_lowpass_20y_sd_ratio"
                ),
                "outer_median_re": _metric(outer, "median_re"),
                "outer_median_ce": _metric(outer, "median_ce"),
            }
        )
        folds = result.reconstruction.validation_folds.copy()
        folds.insert(0, "experiment", name)
        folds.insert(0, "family", family)
        fold_rows.append(folds)
        print(pd.DataFrame([rows[-1]]).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "benchmark_summary.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(ROOT / "benchmark_folds.csv", index=False)


if __name__ == "__main__":
    main()
