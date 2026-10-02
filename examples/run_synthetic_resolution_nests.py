"""Paired native-resolution NEST check: ten annual trees plus one stalagmite.

This controlled check isolates fitting/fusion after a declared proxy network is
selected. The final 100 target years are never supplied to a fit or tuning.
Only compact scores and NEST audits are saved; native observations stay sparse.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from npcrflow.config import AmplitudeCalibrationConfig, PCAConfig, ReconstructionConfig, ScreeningConfig
from npcrflow.nesting import combine_fitted_nests, fit_explicit_nests
from npcrflow.records import ProxyRecord
from npcrflow.validation import multiscale_reconstruction_metrics, reconstruction_metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results/synthetic_resolution_nests"
    root.mkdir(parents=True, exist_ok=True)
    scores = []
    audits = []
    for seed in (20261002, 20261003, 20261004):
        rng = np.random.default_rng(seed)
        years = np.arange(1800, 2000)
        step = np.arange(len(years))
        annual = 0.8 * np.sin(2 * np.pi * step / 4) + 0.4 * np.sin(2 * np.pi * step / 7)
        slow = np.sin(2 * np.pi * step / 25) + 0.2 * np.sin(2 * np.pi * step / 65)
        truth = pd.Series(annual + slow, index=years, name="truth")
        train = years[:100]
        validation = years[100:]
        records = {
            f"tree_{number}": ProxyRecord(
                f"tree_{number}", years,
                (1 + number / 20) * annual + rng.normal(0, 0.12, len(years)),
                archive="Wood",
            )
            for number in range(10)
        }
        config = ReconstructionConfig(
            interpolation="none", regression="ridge", ridge_alphas=(0.0, 0.1),
            validation_block_years=30, n_bootstrap=0,
            full_network_outer_validation=False, rescreen_outer_folds=False,
            amplitude=AmplitudeCalibrationConfig(method="none"),
        )
        pca = PCAConfig(max_components=2, min_pairwise_overlap=6)

        for width in (0, 3, 5, 10):
            network = dict(records)
            if width:
                centers = np.arange(years[0] + (width - 1) / 2, years[-1] + 1, width)
                values = [
                    truth.loc[(truth.index >= center - width / 2) & (truth.index < center + width / 2)].mean()
                    for center in centers
                ]
                network["stalagmite"] = ProxyRecord(
                    "stalagmite", centers,
                    np.asarray(values) + rng.normal(0, 0.05, len(values)),
                    archive="Speleothem",
                )
            selected = pd.DataFrame({"pid": list(network), "selected": True})
            fitted, audit = fit_explicit_nests(
                network, list(network), selected, truth.reindex(train), train,
                ScreeningConfig(), pca, config,
            )
            combined, _ = combine_fitted_nests(fitted, config)
            metrics = reconstruction_metrics(
                truth.reindex(validation), combined["median"].reindex(validation),
                calibration_mean=float(truth.reindex(train).mean()),
            )
            metrics.update(multiscale_reconstruction_metrics(
                truth.reindex(validation), combined["median"].reindex(validation), (10, 20),
                calibration_mean=float(truth.reindex(train).mean()),
            ))
            row = {"seed": seed, "native_resolution_years": width,
                   "annual_proxy_count": 10,
                   "native_proxy_count": int(width > 0), **metrics}
            scores.append(row)
            audits.append(audit.assign(seed=seed, experiment_resolution_years=width))
            print(pd.DataFrame([row])[["seed", "native_resolution_years", "r", "sd_ratio", "lowpass_10y_r", "lowpass_10y_sd_ratio"]].to_string(index=False), flush=True)
    table = pd.DataFrame(scores)
    table.to_csv(root / "heldout_scores.csv", index=False)
    summary = table.groupby("native_resolution_years")[["r", "sd_ratio", "lowpass_10y_r", "lowpass_10y_sd_ratio"]].median()
    summary.to_csv(root / "heldout_summary.csv")
    pd.concat(audits, ignore_index=True).to_csv(root / "nest_summary.csv", index=False)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
