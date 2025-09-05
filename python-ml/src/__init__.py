"""
MovieRecs ML Pipeline Package

This package provides the core machine learning pipeline components for
the MovieRecs recommendation system, including data acquisition,
preprocessing, validation, and model training capabilities.
"""

__version__ = "0.1.0"

from .acquisition import DataAcquisitionError, DatasetDownloader
from .data_prep import DataPipeline, PipelineConfig, PipelineState
from .feature_utils import FeatureNameMapper, ScalingUtilities
from .preprocessing import FeatureEngineer
from .schema import Movie, MovieCollection, QualityThresholds, ValidationResult
from .validation import BiasMetrics, DataValidator

__all__ = [
    "DatasetDownloader",
    "DataAcquisitionError",
    "DataValidator",
    "BiasMetrics",
    "Movie",
    "MovieCollection",
    "ValidationResult",
    "QualityThresholds",
    "FeatureEngineer",
    "FeatureNameMapper",
    "ScalingUtilities",
    "DataPipeline",
    "PipelineConfig",
    "PipelineState",
]
