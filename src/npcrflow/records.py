"""In-memory proxy records with no mandatory temporal interpolation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProxyRecord:
    pid: str
    time: np.ndarray
    value: np.ndarray
    lat: float = np.nan
    lon: float = np.nan
    elev: float = np.nan
    archive: str = "unknown"
    proxy: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float).reshape(-1)
        value = np.asarray(self.value, dtype=float).reshape(-1)
        if time.size != value.size:
            raise ValueError(f"{self.pid}: time/value lengths differ ({time.size} != {value.size})")
        finite = np.isfinite(time) & np.isfinite(value)
        frame = pd.DataFrame({"time": time[finite], "value": value[finite]})
        frame = frame.groupby("time", as_index=False, sort=True)["value"].mean()
        if frame.empty:
            raise ValueError(f"{self.pid}: no finite observations")
        object.__setattr__(self, "time", frame["time"].to_numpy(dtype=float))
        object.__setattr__(self, "value", frame["value"].to_numpy(dtype=float))

    @property
    def resolution(self) -> float:
        if self.time.size < 2:
            return np.nan
        differences = np.diff(np.unique(self.time))
        differences = differences[differences > 0]
        return float(np.median(differences)) if differences.size else np.nan

    @property
    def start(self) -> float:
        return float(self.time.min())

    @property
    def end(self) -> float:
        return float(self.time.max())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"time": self.time, "value": self.value})

    def with_series(self, time: np.ndarray, value: np.ndarray) -> "ProxyRecord":
        return ProxyRecord(
            pid=self.pid,
            time=time,
            value=value,
            lat=self.lat,
            lon=self.lon,
            elev=self.elev,
            archive=self.archive,
            proxy=self.proxy,
            metadata=self.metadata,
        )


class ProxyCollection(Mapping[str, ProxyRecord]):
    def __init__(self, records: Iterable[ProxyRecord]):
        self._records: dict[str, ProxyRecord] = {}
        for record in records:
            if record.pid in self._records:
                raise ValueError(f"duplicate proxy identifier: {record.pid}")
            self._records[record.pid] = record

    def __getitem__(self, key: str) -> ProxyRecord:
        return self._records[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def subset(self, pids: Iterable[str]) -> "ProxyCollection":
        return ProxyCollection(self._records[pid] for pid in pids)

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pid": record.pid,
                    "archive": record.archive,
                    "proxy": record.proxy,
                    "lat": record.lat,
                    "lon": record.lon,
                    "elev": record.elev,
                    "start": record.start,
                    "end": record.end,
                    "resolution": record.resolution,
                    "n": record.time.size,
                }
                for record in self.values()
            ]
        )

