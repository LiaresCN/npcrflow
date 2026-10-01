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
    were used to fit the final model. Contiguous-period correlation
    sensitivities are reported separately in ``validation_folds.csv`` and
    ``validation_summary.csv``; they do not accept or reject the final result.
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


def primary_reconstruction_summary(
    target: pd.Series,
    reconstruction: pd.DataFrame,
    model_selection: pd.DataFrame,
    config: ReconstructionConfig,
) -> pd.DataFrame:
    """Combine the two declared sources of main-result evidence.

    The main reconstruction is described by its correlation with observations
    and the CE/RE obtained during internal NPCR model construction.  External
    segment and proxy-network experiments are intentionally excluded because
    they are supplementary robustness sensitivities.
    """

    apparent = observation_fit_table(target, reconstruction, config).iloc[0]
    selected = model_selection.copy()
    if not selected.empty and "selected" in selected:
        selected = selected.loc[selected["selected"].astype(bool)]
    internal = None
    if not selected.empty and "selection_stage" in selected:
        amplitude = selected.loc[selected["selection_stage"] == "amplitude"]
        internal = amplitude.iloc[-1] if not amplitude.empty else selected.iloc[-1]
    elif not selected.empty:
        internal = selected.iloc[-1]
    return pd.DataFrame(
        [{
            "evaluation": "primary_reconstruction_evidence",
            "validity_basis": "observation_correlation_and_internal_npcr_ce_re",
            "main_r": float(apparent.get("r", np.nan)),
            "main_p_effective": float(apparent.get("p_effective", np.nan)),
            "main_n_eff": float(apparent.get("n_eff", np.nan)),
            "main_sd_ratio": float(apparent.get("sd_ratio", np.nan)),
            "internal_median_ce": (
                float(internal.get("median_ce", np.nan)) if internal is not None else np.nan
            ),
            "internal_median_re": (
                float(internal.get("median_re", np.nan)) if internal is not None else np.nan
            ),
            "internal_ce_threshold": (
                float(internal.get("internal_ce_threshold", np.nan))
                if internal is not None else config.min_ce
            ),
            "internal_re_threshold": (
                float(internal.get("internal_re_threshold", np.nan))
                if internal is not None else config.min_re
            ),
            "internal_npcr_threshold_passed": (
                bool(
                    internal.get(
                        "passes_internal_ce_re",
                        internal.get("passes_skill_floor", False),
                    )
                )
                if internal is not None else pd.NA
            ),
            "external_sensitivities_decide_main_result": False,
        }]
    )


def plot_observation_diagnostics(
    target: pd.Series,
    reconstruction: pd.DataFrame,
    validation_folds: pd.DataFrame,
    config: ReconstructionConfig,
    path: str | Path,
    *,
    show_external_sensitivities: bool = False,
) -> Path:
    """Plot apparent fit and correlation-only contiguous sensitivities."""

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
        if not show_external_sensitivities:
            axes[1].set_visible(False)
            figure.set_size_inches(10, 4.2)
        elif not folds.empty:
            fold_keys = list(
                dict.fromkeys(
                    (
                        str(row.holdout_position),
                        int(row.validation_start),
                        int(row.validation_end),
                    )
                    for row in folds.itertuples()
                )
            )
            x = np.arange(len(fold_keys), dtype=float)
            modes = list(dict.fromkeys(folds["validation_mode"].astype(str)))
            width = 0.75 / max(1, len(modes))
            colors = ("#2b6cb0", "#c05621", "#2f855a", "#805ad5")
            for mode_number, mode in enumerate(modes):
                group = folds.loc[folds["validation_mode"].astype(str) == mode]
                lookup = {
                    (
                        str(row.holdout_position),
                        int(row.validation_start),
                        int(row.validation_end),
                    ): float(row.r)
                    for row in group.itertuples()
                }
                offset = (mode_number - (len(modes) - 1) / 2.0) * width
                axes[1].bar(
                    x + offset,
                    [lookup.get(key, np.nan) for key in fold_keys],
                    width=width,
                    label=mode.replace("_", " "),
                    color=colors[mode_number % len(colors)],
                )
            labels = [
                f"{position}\n{start}–{end}" for position, start, end in fold_keys
            ]
            axes[1].set_xticks(x, labels)
        else:
            axes[1].text(0.5, 0.5, "No outer-fold validation available", ha="center", va="center")
        if show_external_sensitivities:
            axes[1].axhline(0.0, color="black", linewidth=0.9)
            axes[1].set_title("Contiguous holdout correlations (sensitivity only)")
            axes[1].set_ylabel("Correlation (r)")
            axes[1].grid(axis="y", alpha=0.25)
            axes[1].legend(frameon=False, ncol=max(1, len(modes)) if not folds.empty else 1)
        figure.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(figure)
    return path
