"""Compare a compact npcrflow reconstruction with a reference series."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("new", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--new-year", default="Year")
    parser.add_argument("--new-value", default="median")
    parser.add_argument("--reference-year", default="Year")
    parser.add_argument("--reference-value", default="Median")
    args = parser.parse_args()

    new = read_table(args.new)[[args.new_year, args.new_value]].rename(
        columns={args.new_year: "year", args.new_value: "new"}
    )
    reference = read_table(args.reference)[
        [args.reference_year, args.reference_value]
    ].rename(columns={args.reference_year: "year", args.reference_value: "reference"})
    joined = new.merge(reference, on="year").dropna()
    if joined.empty:
        raise RuntimeError("the two reconstructions have no finite common years")
    residual = joined["new"] - joined["reference"]
    scale = joined["new"].std(ddof=1) / joined["reference"].std(ddof=1)
    summary = pd.DataFrame(
        [
            {
                "new_source": str(args.new.resolve()),
                "reference_source": str(args.reference.resolve()),
                "overlap_start": int(joined["year"].min()),
                "overlap_end": int(joined["year"].max()),
                "n": len(joined),
                "r": float(joined["new"].corr(joined["reference"])),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mean_difference": float(residual.mean()),
                "standard_deviation_ratio": float(scale),
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
