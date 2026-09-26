"""Command-line interface for reproducible server runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import (
    AmplitudeCalibrationConfig,
    ProxyWeightConfig,
    MultiresolutionConfig,
    OutputConfig,
    PCAConfig,
    PipelineConfig,
    ProxyFilterConfig,
    ReconstructionConfig,
    ScreeningConfig,
    SensitivityConfig,
    TargetConfig,
)
from .deduplicate import deduplicate_proxy_database
from .pipeline import run_pipeline


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> PipelineConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = _tuples(raw)
    output = dict(raw.get("output", {}))
    if "directory" in output:
        output["directory"] = Path(output["directory"])
    reconstruction = dict(raw.get("reconstruction", {}))
    reconstruction["amplitude"] = AmplitudeCalibrationConfig(
        **dict(reconstruction.get("amplitude", {}))
    )
    reconstruction["proxy_weights"] = ProxyWeightConfig(
        **dict(reconstruction.get("proxy_weights", {}))
    )
    reconstruction["multiresolution"] = MultiresolutionConfig(
        **dict(reconstruction.get("multiresolution", {}))
    )
    return PipelineConfig(
        proxy_filter=ProxyFilterConfig(**raw.get("proxy_filter", {})),
        target=TargetConfig(**raw.get("target", {})),
        screening=ScreeningConfig(**raw.get("screening", {})),
        pca=PCAConfig(**raw.get("pca", {})),
        reconstruction=ReconstructionConfig(**reconstruction),
        sensitivity=SensitivityConfig(**raw.get("sensitivity", {})),
        output=OutputConfig(**output),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="npcrflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    deduplicate = subparsers.add_parser("deduplicate", help="remove exact time/value duplicates")
    deduplicate.add_argument("source", type=Path)
    deduplicate.add_argument("output", type=Path)
    deduplicate.add_argument("--report", type=Path)
    deduplicate.add_argument("--groups", type=Path)

    run = subparsers.add_parser("run", help="screen and reconstruct from raw files")
    run.add_argument("proxy_source", type=Path)
    run.add_argument("observation_source", type=Path)
    run.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "deduplicate":
        cleaned, report, groups = deduplicate_proxy_database(
            args.source,
            args.output,
            report_csv=args.report,
            groups_csv=args.groups,
        )
        print(json.dumps({"kept": len(cleaned), "removed": len(report), "groups": len(groups)}))
        return
    config = load_config(args.config)
    result = run_pipeline(args.proxy_source, args.observation_source, config)
    summary = result.reconstruction.validation_summary.iloc[0].to_dict()
    summary["selected_proxy_count"] = int(result.screening["selected"].sum())
    summary["outputs"] = {name: str(path) for name, path in result.output_paths.items()}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
