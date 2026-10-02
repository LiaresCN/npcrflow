"""Exercise the release wheel directly without installing into mybase."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    sys.path.insert(0, str(wheel))
    import npcrflow
    from npcrflow import PipelineConfig, ProxyFilterConfig, TargetConfig, ScreeningConfig, OutputConfig, run_pipeline

    assert npcrflow.__file__.startswith(str(wheel))
    assert npcrflow.__version__ == "1.0.0"
    config = PipelineConfig()
    assert config.pca.selection == "kaiser" and config.pca.max_components == 8
    assert config.reconstruction.regression == "ols"
    assert config.reconstruction.n_bootstrap == 500
    assert config.reconstruction.bootstrap_method == "random_holdout"
    assert config.reconstruction.min_ce == .1 and config.reconstruction.min_re == .1
    assert config.reconstruction.internal_ce_re_comparison == "gt"
    assert not config.reconstruction.full_network_outer_validation
    rng = np.random.default_rng(31)
    years = np.arange(1900, 2001)
    latent = np.sin(np.arange(101) / 7) + .3 * np.cos(np.arange(101) / 3)
    rows = []
    for pid, times, archive in [("tree1", years, "Wood"), ("tree2", years, "Wood"),
                                 ("stone", years[1::3], "Speleothem")]:
        values = pd.Series(latent, index=years).reindex(times).to_numpy() + rng.normal(0, .04, len(times))
        rows.append({"datasetId": pid, "year": times, "paleoData_values": values,
                     "yearUnits": "CE", "archiveType": archive, "paleoData_proxy": "d18O",
                     "geo_meanLat": 25., "geo_meanLon": 120.})
    with tempfile.TemporaryDirectory(prefix="npcr-v100-wheel-") as temporary:
        root = Path(temporary)
        pd.DataFrame(rows).to_pickle(root / "proxy.pkl")
        pd.DataFrame({"Year": years, "data": latent}).to_csv(root / "target.csv", index=False)
        cfg = replace(config, proxy_filter=ProxyFilterConfig(proxies=("d18O",)),
                      target=TargetConfig(value_column="data"),
                      screening=ScreeningConfig(period=(1900, 2000), p_threshold=1., proxy_multiple_testing="none"),
                      output=OutputConfig(directory=root / "results", save_proxy_map=False,
                                          save_observation_plot=False))
        result = run_pipeline(root / "proxy.pkl", root / "target.csv", cfg)
        rec = result.reconstruction.reconstruction
        nests = result.reconstruction.nest_summary
        manifest = json.loads((root / "results" / "manifest.json").read_text())
        assert manifest["package_version"] == "1.0.0"
        assert manifest["uncertainty"]["method"] == "random_holdout"
        assert manifest["uncertainty"]["successful_nest_runs"] >= 500
        assert (rec.q05 <= rec.q25).all() and (rec.q25 <= rec["median"]).all()
        assert (rec["median"] <= rec.q75).all() and (rec.q75 <= rec.q95).all()
        assert (rec.q95 - rec.q05).gt(0).any()
        native = nests[nests.row_type.eq("resolution_subnest") & nests.accepted.eq(True)]
        assert not native.empty and native.fitted_native_pids.str.contains("stone").any()
        print(json.dumps({"wheel_import": npcrflow.__file__, "version": npcrflow.__version__,
                          "full_default_500_smoke_passed": True, "annual_years": len(rec),
                          "accepted_coverage_nests": manifest["accepted_nest_count"],
                          "accepted_native_layers": len(native)}))


if __name__ == "__main__":
    main()
