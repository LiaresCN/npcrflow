"""Shared half-open native windows; never interpolate proxy observations."""

import numpy as np
import pandas as pd


def window_table(start: int, end: int, width: int) -> pd.DataFrame:
    starts = np.arange(start, end + 1, width, dtype=int)
    ends = np.minimum(starts + width - 1, end)
    return pd.DataFrame({
        "window_id": starts, "window_start": starts, "window_end": ends,
        "window_center": (starts + ends) / 2.0,
        "window_year_count": ends - starts + 1,
    }).set_index("window_id")


def observed_window_means(time, values, windows: pd.DataFrame) -> pd.Series:
    """Average actual values in [start - .5, end + .5), once per window."""
    if windows.empty:
        return pd.Series(index=windows.index, dtype=float)
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    positions = np.searchsorted(
        windows.window_start.to_numpy(float) - 0.5, time, side="right",
    ) - 1
    safe = np.clip(positions, 0, len(windows) - 1)
    upper = windows.window_end.to_numpy(float)[safe] + 0.5
    inside = ((positions >= 0) & (time < upper)
              & np.isfinite(time) & np.isfinite(values))
    observed = pd.Series(values[inside], index=windows.index.to_numpy()[positions[inside]])
    return observed.groupby(level=0).mean().reindex(windows.index)
