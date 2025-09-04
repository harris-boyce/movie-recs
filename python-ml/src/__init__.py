"""
MovieRecs ML Pipeline Package

This package provides the core machine learning pipeline components for 
the MovieRecs recommendation system, including data acquisition, 
preprocessing, validation, and model training capabilities.
"""

__version__ = "0.1.0"

from .acquisition import DatasetDownloader, DataAcquisitionError
from .validation import DataValidator, BiasMetrics
from .schema import Movie, MovieCollection, ValidationResult, QualityThresholds
from .preprocessing import FeatureEngineer
from .feature_utils import FeatureNameMapper, ScalingUtilities
from .data_prep import DataPipeline, PipelineConfig, PipelineState

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
    "PipelineState"
]