"""Readers and temporal aggregation for raw proxy and observation files."""

from __future__ import annotations

import ast
import calendar
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .records import ProxyCollection, ProxyRecord


MONTH_COLUMNS = {name.lower(): number for number, name in enumerate(calendar.month_abbr) if name}
MONTH_COLUMNS.update({name.lower(): number for number, name in enumerate(calendar.month_name) if name})


def parse_array(value: Any) -> np.ndarray:
    """Parse array cells, including one legacy Dod2k row stored as a string."""

    if isinstance(value, str) and value.lstrip().startswith(("[", "(")):
        value = ast.literal_eval(value)
    return np.asarray(value).reshape(-1)


def _number(value: Any) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(result) if pd.notna(result) else np.nan


def _convert_year_units(time: np.ndarray, unit: Any) -> np.ndarray:
    label = str(unit).strip().lower()
    if label in {"bp", "yr bp", "years bp", "cal bp", "cal yr bp"}:
        return 1950.0 - time
    return time


def _metadata(row: pd.Series, excluded: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded:
            continue
        if isinstance(value, np.ndarray):
            continue
        result[str(key)] = value
    return result


def _from_dod2k(frame: pd.DataFrame) -> ProxyCollection:
    records: list[ProxyRecord] = []
    for source_index, row in frame.iterrows():
        pid = str(row.get("datasetId", row.get("dataSetName", source_index))).strip()
        time = pd.to_numeric(pd.Series(parse_array(row["year"])), errors="coerce").to_numpy(float)
        value = pd.to_numeric(
            pd.Series(parse_array(row["paleoData_values"])), errors="coerce"
        ).to_numpy(float)
        if time.size != value.size:
            raise ValueError(f"{pid}: raw year/value lengths differ")
        time = _convert_year_units(time, row.get("yearUnits", "CE"))
        records.append(
            ProxyRecord(
                pid=pid,
                time=time,
                value=value,
                lat=_number(row.get("geo_meanLat")),
                lon=_number(row.get("geo_meanLon")),
                elev=_number(row.get("geo_meanElev")),
                archive=str(row.get("archiveType", "unknown")),
                proxy=str(row.get("paleoData_proxy", row.get("paleoData_variableName", "unknown"))),
                metadata=_metadata(
                    row,
                    {"year", "paleoData_values", "geo_meanLat", "geo_meanLon", "geo_meanElev"},
                )
                | {"source_index": source_index},
            )
        )
    return ProxyCollection(records)


def _from_cfr(frame: pd.DataFrame) -> ProxyCollection:
    records: list[ProxyRecord] = []
    for source_index, row in frame.iterrows():
        records.append(
            ProxyRecord(
                pid=str(row["pid"]).strip(),
                time=pd.to_numeric(pd.Series(parse_array(row["time"])), errors="coerce").to_numpy(float),
                value=pd.to_numeric(pd.Series(parse_array(row["value"])), errors="coerce").to_numpy(float),
                lat=_number(row.get("lat")),
                lon=_number(row.get("lon")),
                elev=_number(row.get("elev")),
                archive=str(row.get("archiveType", str(row.get("ptype", "unknown")).split(".")[0])),
                proxy=str(row.get("ptype", "unknown")),
                metadata=_metadata(row, {"time", "value", "lat", "lon", "elev"})
                | {"source_index": source_index},
            )
        )
    return ProxyCollection(records)


def load_proxy_frame(frame: pd.DataFrame) -> ProxyCollection:
    """Convert a raw Dod2k or CFR-style DataFrame to proxy records."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("proxy pickle must contain a pandas DataFrame")
    if {"year", "paleoData_values", "datasetId"}.issubset(frame.columns):
        return _from_dod2k(frame)
    if {"pid", "time", "value"}.issubset(frame.columns):
        return _from_cfr(frame)
    raise ValueError(f"unrecognized proxy schema with columns: {frame.columns.tolist()}")


def load_proxy_database(path: str | Path) -> ProxyCollection:
    """Load either the raw 21-column Dod2k table or a CFR-style pickle."""

    path = Path(path)
    if path.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError("proxy databases must be pandas pickle files")
    return load_proxy_frame(pd.read_pickle(path))


def load_proxy_workbook(path: str | Path) -> ProxyCollection:
    """Load a legacy workbook whose sheets each contain Year and Value."""

    sheets = pd.read_excel(path, sheet_name=None)
    records: list[ProxyRecord] = []
    for name, frame in sheets.items():
        if frame.shape[1] < 2:
            continue
        year_col = "Year" if "Year" in frame else frame.columns[0]
        value_col = "Value" if "Value" in frame else frame.columns[1]
        records.append(
            ProxyRecord(
                pid=str(name),
                time=pd.to_numeric(frame[year_col], errors="coerce").to_numpy(float),
                value=pd.to_numeric(frame[value_col], errors="coerce").to_numpy(float),
                metadata={"source_sheet": str(name)},
            )
        )
    return ProxyCollection(records)


def _season_year(year: np.ndarray, month: np.ndarray, months: Sequence[int], anchor: str) -> np.ndarray:
    result = year.copy()
    if anchor == "calendar" or len(months) == 12:
        return result
    if anchor == "end":
        last = months[-1]
        result = result + (month > last)
    elif anchor == "start":
        first = months[0]
        result = result - (month < first)
    else:
        raise ValueError(f"unknown season-year anchor: {anchor}")
    return result


def annualize_record(
    record: ProxyRecord,
    months: Sequence[int] = tuple(range(1, 13)),
    season_year: Literal["end", "start", "calendar"] = "end",
    minimum_month_fraction: float = 0.75,
) -> pd.Series:
    """Aggregate a record to annual/seasonal values without interpolation.

    For already annual or lower-resolution records, repeated observations in
    the same integer year are averaged and the requested season is not
    invented.  Subannual fractional years are converted to calendar months in
    the same convention used by CFR before applying a seasonal mean.
    """

    months = tuple(int(month) for month in months)
    if record.resolution >= 0.75 or record.time.size < 2:
        years = np.floor(record.time + 1e-7).astype(int)
        series = pd.Series(record.value, index=years).groupby(level=0).mean().sort_index()
        series.index.name = "Year"
        series.name = record.pid
        return series

    years = np.floor(record.time).astype(int)
    fractions = record.time - np.floor(record.time)
    month = np.clip(np.floor(fractions * 12.0 + 1e-7).astype(int) + 1, 1, 12)
    keep = np.isin(month, months)
    if not np.any(keep):
        return pd.Series(dtype=float, name=record.pid, index=pd.Index([], name="Year"))
    target_year = _season_year(years[keep], month[keep], months, season_year)
    frame = pd.DataFrame(
        {"Year": target_year, "month": month[keep], "value": record.value[keep]}
    )
    grouped = frame.groupby("Year")
    values = grouped["value"].mean()
    month_count = grouped["month"].nunique()
    required = max(1, int(np.ceil(len(set(months)) * minimum_month_fraction)))
    values = values[month_count >= required].sort_index()
    values.index = values.index.astype(int)
    values.index.name = "Year"
    values.name = record.pid
    return values


def _monthly_target(
    frame: pd.DataFrame,
    year_col: str,
    months: Sequence[int],
    season_year: str,
    minimum_month_fraction: float,
) -> pd.Series:
    columns: list[tuple[str, int]] = []
    for column in frame.columns:
        label = str(column).strip().lower()
        if label in MONTH_COLUMNS:
            columns.append((column, MONTH_COLUMNS[label]))
    if not columns:
        raise ValueError("no monthly columns were found")
    long = frame[[year_col] + [column for column, _ in columns]].melt(
        id_vars=year_col, var_name="month_name", value_name="value"
    )
    lookup = {str(column): number for column, number in columns}
    long["month"] = long["month_name"].astype(str).map(lookup)
    long["Year"] = pd.to_numeric(long[year_col], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["Year", "month", "value"])
    selected = long[long["month"].isin(months)].copy()
    selected["Year"] = _season_year(
        selected["Year"].to_numpy(int),
        selected["month"].to_numpy(int),
        tuple(months),
        season_year,
    )
    grouped = selected.groupby("Year")
    result = grouped["value"].mean()
    required = max(1, int(np.ceil(len(set(months)) * minimum_month_fraction)))
    result = result[grouped["month"].nunique() >= required]
    result.index = result.index.astype(int)
    return result.sort_index()


def load_observations(
    source: str | Path | pd.Series | pd.DataFrame,
    *,
    year_col: str = "Year",
    value_col: str | None = None,
    months: Sequence[int] = tuple(range(1, 13)),
    season_year: Literal["end", "start", "calendar"] = "end",
    minimum_month_fraction: float = 0.75,
) -> pd.Series:
    """Load annual or monthly observations and return a numeric year Series."""

    if isinstance(source, pd.Series):
        result = source.copy()
        result.index = pd.to_numeric(result.index, errors="coerce")
        result = pd.to_numeric(result, errors="coerce")
        result = result[result.index.notna() & result.notna()]
        result.index = result.index.astype(int)
        return result.groupby(level=0).mean().sort_index()
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError(f"unsupported observation format: {path.suffix}")
    if year_col not in frame:
        year_col = str(frame.columns[0])
    monthly_names = {str(column).strip().lower() for column in frame.columns}
    if monthly_names.intersection(MONTH_COLUMNS):
        result = _monthly_target(frame, year_col, months, season_year, minimum_month_fraction)
    else:
        if value_col is None:
            candidates = [column for column in frame.columns if column != year_col]
            if len(candidates) != 1:
                raise ValueError("value_col is required when the observation table has multiple values")
            value_col = str(candidates[0])
        values = pd.DataFrame(
            {
                "Year": pd.to_numeric(frame[year_col], errors="coerce"),
                "value": pd.to_numeric(frame[value_col], errors="coerce"),
            }
        ).dropna()
        values["Year"] = np.floor(values["Year"]).astype(int)
        result = values.groupby("Year")["value"].mean().sort_index()
    result.index.name = "Year"
    result.name = value_col or "target"
    return result.astype(float)
