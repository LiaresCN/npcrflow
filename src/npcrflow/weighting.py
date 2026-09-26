"""Training-only proxy error and spatial-redundancy weights."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import ProxyWeightConfig


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = phi2 - phi1
    dlambda = np.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    value = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0))))


def _numeric_error(metadata: Mapping[str, object], fields: tuple[str, ...]) -> float:
    lowered = {str(key).strip().lower(): value for key, value in metadata.items()}
    for field in fields:
        value = pd.to_numeric(
            pd.Series([lowered.get(field.strip().lower())]), errors="coerce"
        ).iloc[0]
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return np.nan


def compute_proxy_weights(
    matrix: pd.DataFrame,
    fit_years: Iterable[int],
    config: ProxyWeightConfig,
) -> pd.DataFrame:
    """Return per-proxy PCA weights without consulting the climate target."""

    columns = [str(column) for column in matrix.columns]
    metadata = matrix.attrs.get("proxy_metadata", {})
    explicit = {str(pid): float(value) for pid, value in config.explicit_error_sd}
    years = [int(year) for year in fit_years if year in matrix.index]
    fit = matrix.loc[years, columns] if years else matrix.iloc[0:0][columns]
    rows: list[dict[str, object]] = []
    for pid in columns:
        item = dict(metadata.get(pid, {}))
        error_sd = explicit.get(pid, _numeric_error(item.get("metadata", {}), config.error_sd_fields))
        rows.append(
            {
                "pid": pid,
                "lat": item.get("lat", np.nan),
                "lon": item.get("lon", np.nan),
                "archive": str(item.get("archive", "unknown")),
                "proxy": str(item.get("proxy", "unknown")),
                "site": str(item.get("site", "")),
                "source": str(item.get("source", "")),
                "error_sd": error_sd,
                "calibration_sd": float(fit[pid].std(ddof=1)),
            }
        )
    table = pd.DataFrame(rows).set_index("pid")
    table["error_weight"] = 1.0
    finite_error = pd.to_numeric(table["error_sd"], errors="coerce")
    valid_error = np.isfinite(finite_error) & (finite_error > 0)
    if valid_error.any():
        signal_variance = table.loc[valid_error, "calibration_sd"].pow(2)
        error_variance = finite_error.loc[valid_error].pow(2)
        reliability = signal_variance / (signal_variance + error_variance)
        table.loc[valid_error, "error_weight"] = reliability.fillna(1.0)

    parents = list(range(len(columns)))

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parents[root_second] = root_first

    pair_links = 0
    if config.enabled and config.redundancy_radius_km > 0:
        for first in range(len(columns)):
            row_first = table.iloc[first]
            for second in range(first + 1, len(columns)):
                row_second = table.iloc[second]
                if config.same_archive_only and row_first["archive"].strip().lower() != row_second["archive"].strip().lower():
                    continue
                if config.same_proxy_only and row_first["proxy"].strip().lower() != row_second["proxy"].strip().lower():
                    continue
                same_site = bool(row_first["site"]) and row_first["site"].strip().lower() == row_second["site"].strip().lower()
                same_source = bool(row_first["source"]) and row_first["source"].strip().lower() == row_second["source"].strip().lower()
                coordinates = np.asarray(
                    [row_first["lat"], row_first["lon"], row_second["lat"], row_second["lon"]],
                    dtype=float,
                )
                spatial_neighbor = bool(
                    np.all(np.isfinite(coordinates))
                    and _haversine_km(*coordinates) <= config.redundancy_radius_km
                )
                if not (spatial_neighbor or same_site or same_source):
                    continue
                pair = fit[[columns[first], columns[second]]].dropna()
                if len(pair) < config.redundancy_min_overlap:
                    continue
                correlation = float(pair.corr().iloc[0, 1])
                if np.isfinite(correlation) and abs(correlation) >= config.redundancy_correlation_threshold:
                    union(first, second)
                    pair_links += 1

    groups: dict[int, list[int]] = {}
    for number in range(len(columns)):
        groups.setdefault(find(number), []).append(number)
    group_order = {root: number + 1 for number, root in enumerate(sorted(groups))}
    group_size = {root: len(members) for root, members in groups.items()}
    table["redundancy_group"] = [group_order[find(number)] for number in range(len(columns))]
    table["redundancy_group_size"] = [group_size[find(number)] for number in range(len(columns))]
    table["redundancy_weight"] = 1.0 / table["redundancy_group_size"].astype(float)
    combined = table["error_weight"] * table["redundancy_weight"]
    if config.enabled:
        combined = combined.clip(lower=config.minimum_weight, upper=1.0)
    else:
        combined = pd.Series(1.0, index=table.index)
    table["weight"] = combined
    table["sqrt_weight"] = np.sqrt(combined)
    table["pair_links"] = pair_links
    table["enabled"] = bool(config.enabled)
    return table.reset_index()
