"""Compact diagnostics comparing reconstructions with observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ReconstructionConfig
from .validation import (
    multiscale_reconstruction_metrics,
    reconstruction_metrics,
    spectral_reconstruction_metrics,
)


def observation_fit_table(
    target: pd.Series,
    reconstruction: pd.DataFrame,
    config: ReconstructionConfig,
) -> pd.DataFrame:
    """Return the apparent calibration fit against the observed target.

    This is deliberately labelled ``apparent`` because the same observations
    were used to fit the final model. Independent skill is reported separately
    in ``validation_folds.csv`` and ``validation_summary.csv``.
    """

    predicted = reconstruction.set_index("Year")["median"]
    observed = target.copy()
    if config.calibration_period is not None:
        start, end = config.calibration_period
        observed = observed.loc[(observed.index >= start) & (observed.index <= end)]
    joined = pd.concat(
        [observed.rename("observed"), predicted.rename("predicted")], axis=1, join="inner"
    ).dropna()
    metrics = reconstruction_metrics(
        joined["observed"],
        joined["predicted"],
        calibration_mean=float(joined["observed"].mean()) if not joined.empty else None,
    )
    metrics.update(
        multiscale_reconstruction_metrics(
            joined["observed"],
            joined["predicted"],
            config.evaluation_lowpass_periods,
            calibration_mean=float(joined["observed"].mean()) if not joined.empty else None,
        )
    )
    metrics.update(
        spectral_reconstruction_metrics(
            joined["observed"],
            joined["predicted"],
            config.evaluation_period_bands,
        )
    )
    if "median_raw" in reconstruction:
        raw_metrics = reconstruction_metrics(
            joined["observed"],
            reconstruction.set_index("Year")["median_raw"].reindex(joined.index),
            calibration_mean=float(joined["observed"].mean()) if not joined.empty else None,
        )
        metrics.update({f"raw_{name}": value for name, value in raw_metrics.items()})
    return pd.DataFrame(
        [
            {
                "evaluation": "apparent_calibration_fit",
                "independent_validation": False,
                "start": int(joined.index.min()) if not joined.empty else np.nan,
                "end": int(joined.index.max()) if not joined.empty else np.nan,
                **metrics,
            }
        ]
    )


def plot_observation_diagnostics(
    target: pd.Series,
    reconstruction: pd.DataFrame,
    validation_folds: pd.DataFrame,
    config: ReconstructionConfig,
    path: str | Path,
) -> Path:
    """Plot one observation comparison and the independent fold scores."""

    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predicted = reconstruction.set_index("Year")["median"]
    observed = target.copy()
    if config.calibration_period is not None:
        start, end = config.calibration_period
        observed = observed.loc[(observed.index >= start) & (observed.index <= end)]
    joined = pd.concat(
        [observed.rename("Observed"), predicted.rename("Reconstructed")], axis=1, join="inner"
    ).dropna()

    font = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }
    with plt.rc_context(font):
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        axes[0].plot(joined.index, joined["Observed"], color="black", linewidth=1.4, label="Observed")
        axes[0].plot(
            joined.index,
            joined["Reconstructed"],
            color="#2b6cb0",
            linewidth=1.2,
            label="Final-model fit",
        )
        axes[0].set_title("Observation comparison (apparent calibration fit)")
        axes[0].set_ylabel("Target index")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False, ncol=2)

        folds = validation_folds.copy()
        if not folds.empty:
            x = np.arange(1, len(folds) + 1)
            width = 0.24
            for offset, metric, color in (
                (-width, "r", "#2b6cb0"),
                (0.0, "re", "#2f855a"),
                (width, "ce", "#c05621"),
            ):
                axes[1].bar(x + offset, folds[metric], width=width, label=metric.upper(), color=color)
            labels = [
                f"{int(row.validation_start)}–{int(row.validation_end)}"
                for row in folds.itertuples()
            ]
            axes[1].set_xticks(x, labels, rotation=25, ha="right")
        else:
            axes[1].text(0.5, 0.5, "No outer-fold validation available", ha="center", va="center")
        axes[1].axhline(0.0, color="black", linewidth=0.9)
        axes[1].axhline(
            config.strong_skill_threshold,
            color="0.4",
            linewidth=0.9,
            linestyle="--",
            label=f"strong target = {config.strong_skill_threshold:g}",
        )
        axes[1].set_title("Outer-fold predictions compared with withheld observations")
        axes[1].set_ylabel("Score")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(frameon=False, ncol=4)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return path
