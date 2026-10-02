"""Check score-cache speed and equivalence against a declared Git revision.

This small, deterministic synthetic engineering benchmark does not measure
proxy screening or entire NEST workflows. No data or package files are saved.
Run from the repository with its Git history available.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd

import npcrflow.model as current
from npcrflow.config import AmplitudeCalibrationConfig, PCAConfig, ReconstructionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="2615aa1")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    reference_source = subprocess.check_output(
        ["git", "show", f"{args.reference}:src/npcrflow/model.py"],
        cwd=repository, text=True,
    )
    reference = ModuleType("npcrflow._benchmark_reference")
    reference.__package__ = "npcrflow"
    sys.modules[reference.__name__] = reference
    exec(compile(reference_source, f"{args.reference}/model.py", "exec"), reference.__dict__)
    rng = np.random.default_rng(71)
    years = np.arange(1000, 2001)
    latent = np.sin((years - 1000) / 8) + 0.3 * np.cos((years - 1000) / 19)
    matrix = pd.DataFrame({
        f"p{i}": latent + rng.normal(0, 0.3, len(years)) for i in range(40)
    }, index=years)
    matrix.values[rng.random(matrix.shape) < 0.12] = np.nan
    target = pd.Series(latent + rng.normal(0, 0.1, len(years)), index=years)
    pca = PCAConfig(selection="blocked_cv", max_components=5)
    config = ReconstructionConfig(
        regression="ridge", ridge_alphas=(0.0, 0.01, 0.1, 1.0, 10.0, 100.0),
        amplitude=AmplitudeCalibrationConfig(method="auto"),
        validation_block_years=20, n_bootstrap=0,
    )
    models, timing, counts = {}, {}, {}
    for label, module in (("uncached", reference), ("cached", current)):
        timing[label], counts[label] = [], []
        for _ in range(args.repeats):
            with patch.object(module, "_score_rows", wraps=module._score_rows) as calls:
                start = perf_counter()
                models[label] = module.fit_native_pcr(matrix, target, years[-101:], pca, config)
                timing[label].append(perf_counter() - start)
                counts[label].append(calls.call_count)
    pd.testing.assert_frame_equal(
        models["uncached"].selection_table, models["cached"].selection_table,
        check_exact=False, atol=1e-12, rtol=1e-12,
    )
    old, new = models["uncached"].predict(matrix), models["cached"].predict(matrix)
    np.testing.assert_allclose(old, new, atol=1e-12, rtol=1e-12, equal_nan=True)
    print(json.dumps({
        "reference": args.reference,
        "seconds": timing, "score_calls": counts,
        "median_speedup": float(np.median(timing["uncached"]) / np.median(timing["cached"])),
        "maximum_prediction_difference": float((old - new).abs().max()),
        "selection_and_ce_re_equal": True,
    }, indent=2))


if __name__ == "__main__":
    main()
