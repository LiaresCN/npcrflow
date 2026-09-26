"""Conservative, auditable de-duplication of raw proxy databases."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import parse_array


def _canonical_series(row: pd.Series) -> tuple[str, int, float, float]:
    time = pd.to_numeric(pd.Series(parse_array(row["year"])), errors="coerce").to_numpy(float)
    value = pd.to_numeric(
        pd.Series(parse_array(row["paleoData_values"])), errors="coerce"
    ).to_numpy(float)
    if time.size != value.size:
        raise ValueError(f"{row.get('datasetId', row.name)}: time/value lengths differ")
    finite = np.isfinite(time) & np.isfinite(value)
    series = pd.DataFrame({"time": time[finite], "value": value[finite]})
    series = series.groupby("time", as_index=False)["value"].mean().sort_values("time")
    canonical = np.column_stack(
        [
            np.round(series["time"].to_numpy(float), 6),
            np.round(series["value"].to_numpy(float), 10),
        ]
    ).astype("<f8")
    digest = hashlib.sha256(canonical.tobytes()).hexdigest()
    start = float(canonical[:, 0].min()) if len(canonical) else np.nan
    end = float(canonical[:, 0].max()) if len(canonical) else np.nan
    return digest, len(canonical), start, end


def _canonical_vectors(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted finite time/value vectors for conservative near matching."""

    time = pd.to_numeric(pd.Series(parse_array(row["year"])), errors="coerce").to_numpy(float)
    value = pd.to_numeric(
        pd.Series(parse_array(row["paleoData_values"])), errors="coerce"
    ).to_numpy(float)
    finite = np.isfinite(time) & np.isfinite(value)
    series = pd.DataFrame({"time": time[finite], "value": value[finite]})
    series = series.groupby("time", as_index=False)["value"].mean().sort_values("time")
    return series["time"].to_numpy(float), series["value"].to_numpy(float)


def _near_group_key(row: pd.Series, time: np.ndarray) -> tuple[Any, ...]:
    rounded_time = np.round(time, 6).astype("<f8")
    time_hash = hashlib.sha256(rounded_time.tobytes()).hexdigest()

    def text(name: str) -> str:
        return str(row.get(name, "")).strip().lower()

    def coordinate(name: str) -> float | None:
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        return round(float(value), 4) if np.isfinite(value) else None

    return (
        time_hash,
        len(time),
        text("archiveType"),
        text("paleoData_proxy"),
        text("paleoData_variableName"),
        coordinate("geo_meanLat"),
        coordinate("geo_meanLon"),
    )


def _numerically_equivalent(first: np.ndarray, second: np.ndarray) -> bool:
    """Identify database copies that differ only by numeric storage rounding."""

    if first.shape != second.shape or first.size == 0:
        return False
    scale = max(float(np.std(first)), float(np.std(second)), 1.0)
    difference = first - second
    return bool(
        np.max(np.abs(difference)) <= 1e-4 * scale
        and np.sqrt(np.mean(difference**2)) <= 5e-5 * scale
    )


def _quality(row: pd.Series) -> tuple[int, int, str]:
    excluded = {"year", "paleoData_values"}
    completeness = sum(
        not (value is None or (np.isscalar(value) and pd.isna(value)))
        for key, value in row.items()
        if key not in excluded
    )
    url_bonus = int(pd.notna(row.get("originalDataURL")))
    return completeness, url_bonus, str(row.get("datasetId", row.name))


def deduplicate_frame(
    frame: pd.DataFrame,
    *,
    include_near: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop exact and optionally rounding-equivalent database copies.

    The keeper is the row with the most complete metadata; ties are resolved
    deterministically by dataset identifier. Near matching requires the same
    complete timestamp vector, archive, proxy variable, and rounded location;
    values must differ only at numeric-storage precision. No correlation-only
    heuristic is used because that could erase genuinely different records.
    A group table is returned for a full audit.
    """

    required = {"year", "paleoData_values"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"raw proxy table is missing columns: {sorted(missing)}")
    canonical = pd.DataFrame(
        [_canonical_series(row) for _, row in frame.iterrows()],
        index=frame.index,
        columns=["series_hash", "series_n", "series_start", "series_end"],
    )
    keep_indices: list[Any] = []
    report_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    duplicate_number = 0
    for digest, info in canonical.groupby("series_hash", sort=True):
        indices = list(info.index)
        if len(indices) == 1:
            keep_indices.append(indices[0])
            continue
        duplicate_number += 1
        ranked = sorted(indices, key=lambda index: (-_quality(frame.loc[index])[0], -_quality(frame.loc[index])[1], _quality(frame.loc[index])[2]))
        keeper = ranked[0]
        keep_indices.append(keeper)
        group_id = f"exact_{duplicate_number:04d}"
        group_rows.append(
            {
                "group_id": group_id,
                "group_type": "exact_time_value_duplicate",
                "series_hash": digest,
                "group_size": len(indices),
                "series_n": int(info.iloc[0]["series_n"]),
                "series_start": float(info.iloc[0]["series_start"]),
                "series_end": float(info.iloc[0]["series_end"]),
                "keeper_dataset_id": str(frame.loc[keeper].get("datasetId", keeper)),
                "member_dataset_ids": "|".join(str(frame.loc[index].get("datasetId", index)) for index in indices),
            }
        )
        for index in ranked[1:]:
            report_rows.append(
                {
                    "group_id": group_id,
                    "reason": "exact_time_value_duplicate",
                    "kept_source_index": keeper,
                    "kept_dataset_id": str(frame.loc[keeper].get("datasetId", keeper)),
                    "dropped_source_index": index,
                    "dropped_dataset_id": str(frame.loc[index].get("datasetId", index)),
                    "series_hash": digest,
                    "series_n": int(info.loc[index, "series_n"]),
                    "series_start": float(info.loc[index, "series_start"]),
                    "series_end": float(info.loc[index, "series_end"]),
                }
            )
    keep_indices = sorted(keep_indices, key=lambda index: list(frame.index).index(index))

    if include_near:
        candidates: dict[tuple[Any, ...], list[Any]] = {}
        vectors: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        for index in keep_indices:
            time, value = _canonical_vectors(frame.loc[index])
            vectors[index] = (time, value)
            candidates.setdefault(_near_group_key(frame.loc[index], time), []).append(index)
        near_drop: set[Any] = set()
        for indices in candidates.values():
            clusters: list[list[Any]] = []
            for index in indices:
                for cluster in clusters:
                    representative = cluster[0]
                    if _numerically_equivalent(
                        vectors[index][1], vectors[representative][1]
                    ):
                        cluster.append(index)
                        break
                else:
                    clusters.append([index])
            for indices_in_cluster in clusters:
                if len(indices_in_cluster) < 2:
                    continue
                duplicate_number += 1
                ranked = sorted(
                    indices_in_cluster,
                    key=lambda index: (
                        -_quality(frame.loc[index])[0],
                        -_quality(frame.loc[index])[1],
                        _quality(frame.loc[index])[2],
                    ),
                )
                keeper = ranked[0]
                group_id = f"near_{duplicate_number:04d}"
                time, _ = vectors[keeper]
                group_rows.append(
                    {
                        "group_id": group_id,
                        "group_type": "rounding_equivalent_time_value_duplicate",
                        "series_hash": "",
                        "group_size": len(ranked),
                        "series_n": len(time),
                        "series_start": float(time.min()),
                        "series_end": float(time.max()),
                        "keeper_dataset_id": str(frame.loc[keeper].get("datasetId", keeper)),
                        "member_dataset_ids": "|".join(
                            str(frame.loc[index].get("datasetId", index)) for index in ranked
                        ),
                    }
                )
                for index in ranked[1:]:
                    near_drop.add(index)
                    report_rows.append(
                        {
                            "group_id": group_id,
                            "reason": "rounding_equivalent_time_value_duplicate",
                            "kept_source_index": keeper,
                            "kept_dataset_id": str(frame.loc[keeper].get("datasetId", keeper)),
                            "dropped_source_index": index,
                            "dropped_dataset_id": str(frame.loc[index].get("datasetId", index)),
                            "series_hash": "",
                            "series_n": len(time),
                            "series_start": float(time.min()),
                            "series_end": float(time.max()),
                        }
                    )
        keep_indices = [index for index in keep_indices if index not in near_drop]

    cleaned = frame.loc[keep_indices].copy().reset_index(drop=True)
    report = pd.DataFrame(report_rows)
    groups = pd.DataFrame(group_rows)
    return cleaned, report, groups


def deduplicate_proxy_database(
    source: str | Path,
    output: str | Path,
    *,
    report_csv: str | Path | None = None,
    groups_csv: str | Path | None = None,
    include_near: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a new de-duplicated pickle without modifying ``source``."""

    source = Path(source)
    output = Path(output)
    frame = pd.read_pickle(source)
    cleaned, report, groups = deduplicate_frame(frame, include_near=include_near)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_pickle(output)
    if report_csv is not None:
        Path(report_csv).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_csv, index=False)
    if groups_csv is not None:
        Path(groups_csv).parent.mkdir(parents=True, exist_ok=True)
        groups.to_csv(groups_csv, index=False)
    return cleaned, report, groups
