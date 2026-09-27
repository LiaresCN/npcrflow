"""Modular, leakage-aware proxy climate reconstruction."""

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
from .data import load_observations, load_proxy_database, load_proxy_frame, load_proxy_workbook
from .deduplicate import deduplicate_proxy_database
from .pipeline import PipelineResult, run_pipeline
from .records import ProxyCollection, ProxyRecord
from .screening import correlation_with_effective_dof, screen_proxies

__all__ = [
    "AmplitudeCalibrationConfig",
    "ProxyWeightConfig",
    "MultiresolutionConfig",
    "OutputConfig",
    "PCAConfig",
    "PipelineConfig",
    "PipelineResult",
    "ProxyFilterConfig",
    "ProxyCollection",
    "ProxyRecord",
    "ReconstructionConfig",
    "ScreeningConfig",
    "SensitivityConfig",
    "TargetConfig",
    "correlation_with_effective_dof",
    "deduplicate_proxy_database",
    "load_observations",
    "load_proxy_database",
    "load_proxy_frame",
    "load_proxy_workbook",
    "run_pipeline",
    "screen_proxies",
]

__version__ = "0.4.1"
