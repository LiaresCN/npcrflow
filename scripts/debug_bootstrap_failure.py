"""Print the first NPCR bootstrap failure for development diagnostics."""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "examples"))

from run_pdo_from_raw_dod2k import PROXIES, TARGET, build_config
from npcrflow.data import load_proxy_frame
from npcrflow.deduplicate import deduplicate_frame
from npcrflow.model import fit_native_pcr
from npcrflow.pipeline import filter_records_with_report, prepare_target
from npcrflow.reconstruction import _calibration_years, _moving_block_sample, build_proxy_matrix


def main() -> None:
    config = build_config(amplitude_method="none", multiresolution_enabled=False)
    frame, _, _ = deduplicate_frame(pd.read_pickle(PROXIES))
    records, _ = filter_records_with_report(load_proxy_frame(frame), config.proxy_filter)
    target = prepare_target(TARGET, config)
    screening = pd.read_csv(PROJECT / "results/pdo_v021_all_native_pcr_no_amplitude/proxy_screening.csv")
    selected = screening.loc[screening["selected"], "pid"].astype(str).tolist()
    matrix = build_proxy_matrix(
        records,
        selected,
        config.screening,
        screening,
        config.reconstruction.reconstruction_period,
        "none",
    )
    years = _calibration_years(target, config.reconstruction)
    years = years[np.isin(years, matrix.index)]
    model = fit_native_pcr(matrix, target, years, config.pca, config.reconstruction)
    rng = np.random.default_rng(config.reconstruction.random_seed)
    for repeat in range(5):
        sampled = _moving_block_sample(years, config.reconstruction.bootstrap_block_years, rng)
        try:
            fit_native_pcr(
                matrix,
                target,
                sampled,
                config.pca,
                config.reconstruction,
                forced_n_components=model.n_components,
                forced_alpha=model.alpha,
            )
            print(f"repeat {repeat}: success")
        except Exception:
            print(f"repeat {repeat}: failed")
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
