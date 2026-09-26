#!/usr/bin/env python
"""Create an auditable exact-series de-duplicated Dod2k database."""

from __future__ import annotations

import argparse
from pathlib import Path

from npcrflow.deduplicate import deduplicate_proxy_database


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--groups", type=Path)
    args = parser.parse_args()
    cleaned, report, groups = deduplicate_proxy_database(
        args.source,
        args.output,
        report_csv=args.report,
        groups_csv=args.groups,
    )
    print(f"kept={len(cleaned)} removed={len(report)} groups={len(groups)}")
    print(args.output)


if __name__ == "__main__":
    main()

