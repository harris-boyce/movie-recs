"""
Main data pipeline orchestration for MovieRecs.

This module coordinates all data processing steps: acquisition → validation → preprocessing → export.
Provides progress tracking, error recovery, incremental processing, and a command-line interface.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import yaml
from tqdm import tqdm

from .acquisition import DatasetDownloader, DataAcquisitionError
from .validation import DataValidator, BiasMetrics, ValidationResult
from .preprocessing import FeatureEngineer
from .schema import Movie, create_movie_from_dict


logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration manager for the data pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.config_path = config_path or "configs/data_config.yaml"
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

            # Apply environment variable overrides
            config = self._apply_env_overrides(config)

            return config

        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self._get_default_config()
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data_sources": {
                "primary": "local_fallback",
                "local_fallback": {"path": "data/raw/fallback_movies.json"},
            },
            "download": {"cache_dir": "data/raw", "temp_dir": "data/temp"},
            "processing": {"min_movies": 100, "max_movies": 10000},
            "quality_thresholds": {"completeness_min": 0.9},
            "bias_detection": {
                "enable_genre_analysis": True,
                "enable_demographic_analysis": True,
            },
            "feature_engineering": {
                "use_genre_encoding": True,
                "use_decade_features": True,
                "use_runtime_binning": True,
                "use_rating_features": True,
            },
            "logging": {"level": "INFO", "file": "data/logs/pipeline.log"},
            "performance": {"max_memory_gb": 4, "parallel_workers": 2},
        }

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Override data source if env variable set
        if "MOVIERECS_DATA_SOURCE" in os.environ:
            config["data_sources"]["primary"] = os.environ["MOVIERECS_DATA_SOURCE"]

        # Override output directory
        if "MOVIERECS_OUTPUT_DIR" in os.environ:
            output_dir = os.environ["MOVIERECS_OUTPUT_DIR"]
            config["download"]["cache_dir"] = f"{output_dir}/raw"

        # Override max movies for testing
        if "MOVIERECS_MAX_MOVIES" in os.environ:
            config["processing"]["max_movies"] = int(os.environ["MOVIERECS_MAX_MOVIES"])

        return config

    def _validate_config(self) -> None:
        """Validate configuration values."""
        required_keys = ["data_sources", "processing", "quality_thresholds"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config section: {key}")

        # Validate numeric values
        if (
            self.config["processing"]["max_movies"]
            < self.config["processing"]["min_movies"]
        ):
            raise ValueError("max_movies must be >= min_movies")

        if not 0 <= self.config["quality_thresholds"]["completeness_min"] <= 1:
            raise ValueError("completeness_min must be between 0 and 1")


class PipelineState:
    """Tracks pipeline execution state and progress."""

    def __init__(self):
        """Initialize pipeline state."""
        self.start_time = None
        self.end_time = None
        self.current_step = "initialized"
        self.steps_completed = []
        self.total_movies = 0
        self.valid_movies = 0
        self.exported_formats = []
        self.memory_usage = {}
        self.error_log = []

    def start_step(self, step_name: str) -> None:
        """Start a pipeline step."""
        self.current_step = step_name
        logger.info(f"Starting step: {step_name}")

    def complete_step(self, step_name: str, result: Any = None) -> None:
        """Mark step as completed."""
        self.steps_completed.append(
            {
                "name": step_name,
                "completed_at": datetime.now().isoformat(),
                "result_summary": self._summarize_result(result),
            }
        )
        logger.info(f"Completed step: {step_name}")

    def _summarize_result(self, result: Any) -> str:
        """Create summary of step result."""
        if isinstance(result, ValidationResult):
            return f"Valid: {result.is_valid}, Errors: {result.error_count}, Warnings: {result.warning_count}"
        elif isinstance(result, list):
            return f"Count: {len(result)}"
        elif isinstance(result, dict):
            return f"Keys: {list(result.keys())}"
        else:
            return str(type(result).__name__)

    def add_error(self, error: Exception, step: str) -> None:
        """Add error to log."""
        self.error_log.append(
            {"step": step, "error": str(error), "timestamp": datetime.now().isoformat()}
        )

    def update_memory_usage(self) -> None:
        """Update current memory usage."""
        process = psutil.Process()
        self.memory_usage[datetime.now().isoformat()] = {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            "execution_time_seconds": duration,
            "steps_completed": len(self.steps_completed),
            "total_movies": self.total_movies,
            "valid_movies": self.valid_movies,
            "success_rate": (
                self.valid_movies / self.total_movies if self.total_movies > 0 else 0
            ),
            "memory_peak_mb": (
                max([m["memory_mb"] for m in self.memory_usage.values()])
                if self.memory_usage
                else 0
            ),
            "errors": len(self.error_log),
            "current_step": self.current_step,
        }


class DataPipeline:
    """Main data pipeline orchestrator."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize data pipeline."""
        self.config = PipelineConfig(config_path)
        self.state = PipelineState()

        # Initialize components
        self.downloader = DatasetDownloader(self.config.config_path)
        self.validator = DataValidator(self.config.config)
        self.feature_engineer = FeatureEngineer(self.config.config)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup pipeline logging."""
        log_config = self.config.config.get("logging", {})

        # Create log directory
        log_file = Path(log_config.get("file", "data/logs/pipeline.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def run_full_pipeline(
        self,
        force_refresh: bool = False,
        skip_download: bool = False,
        export_formats: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete data pipeline.

        Args:
            force_refresh: Force re-download of data
            skip_download: Skip download step (use existing data)
            export_formats: List of formats to export ('csv', 'json', 'parquet', 'numpy')

        Returns:
            Dictionary with pipeline results and metadata
        """
        logger.info("Starting full data pipeline")
        self.state.start_time = datetime.now()

        try:
            # Step 1: Data Acquisition
            if not skip_download:
                movies_data = self._run_acquisition_step(force_refresh)
            else:
                movies_data = self._load_existing_data()

            # Step 2: Data Validation
            validation_result, valid_movies = self._run_validation_step(movies_data)

            # Step 3: Bias Detection
            bias_metrics = self._run_bias_detection_step(valid_movies)

            # Step 4: Feature Engineering
            split_data = self._run_feature_engineering_step(valid_movies)

            # Step 5: Export Results
            exported_files = self._run_export_step(
                split_data,
                valid_movies,
                validation_result,
                bias_metrics,
                export_formats,
            )

            # Generate final report
            self._generate_pipeline_report(validation_result, bias_metrics, split_data)

            self.state.end_time = datetime.now()
            self.state.current_step = "completed"

            results = {
                "status": "success",
                "summary": self.state.get_summary(),
                "validation_result": validation_result,
                "bias_metrics": bias_metrics,
                "exported_files": exported_files,
                "feature_info": {
                    "train_features": split_data["train"]["features"].shape,
                    "test_features": split_data["test"]["features"].shape,
                    "feature_count": len(split_data["feature_names"]),
                },
            }

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            self.state.add_error(e, self.state.current_step)
            self.state.end_time = datetime.now()
            self.state.current_step = "failed"

            logger.error(f"Pipeline failed: {e}")

            return {
                "status": "failed",
                "error": str(e),
                "summary": self.state.get_summary(),
            }

    def _run_acquisition_step(self, force_refresh: bool) -> List[Dict[str, Any]]:
        """Run data acquisition step."""
        self.state.start_step("data_acquisition")

        try:
            dataset_path = self.downloader.download_dataset(force_refresh=force_refresh)

            # Extract if needed
            extracted_path = self.downloader.extract_dataset(dataset_path)

            # Load movie data
            movies_data = self._load_movie_data(extracted_path)

            # Apply limits
            max_movies = self.config.config["processing"]["max_movies"]
            if len(movies_data) > max_movies:
                logger.info(f"Limiting dataset to {max_movies} movies")
                movies_data = movies_data[:max_movies]

            self.state.total_movies = len(movies_data)
            self.state.complete_step("data_acquisition", movies_data)

            return movies_data

        except Exception as e:
            logger.error(f"Data acquisition failed: {e}")
            raise

    def _load_existing_data(self) -> List[Dict[str, Any]]:
        """Load existing processed data."""
        # Try to find existing processed data
        data_files = ["data/processed/movies.json", "data/raw/fallback_movies.json"]

        for file_path in data_files:
            if Path(file_path).exists():
                logger.info(f"Loading existing data from: {file_path}")
                with open(file_path, "r") as f:
                    return json.load(f)

        raise FileNotFoundError(
            "No existing data found. Run pipeline with download enabled."
        )

    def _load_movie_data(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load movie data from various formats."""
        if data_path.is_file() and data_path.suffix.lower() == ".json":
            with open(data_path, "r") as f:
                return json.load(f)
        elif data_path.is_dir():
            # Look for common movie data files in directory
            for filename in ["movies.json", "movies.dat", "movies.csv"]:
                file_path = data_path / filename
                if file_path.exists():
                    if filename.endswith(".json"):
                        with open(file_path, "r") as f:
                            return json.load(f)
                    # Could add CSV/DAT parsers here

        # If no specific format found, create minimal dataset
        logger.warning("No standard movie data found, creating minimal dataset")
        return self._create_minimal_dataset()

    def _create_minimal_dataset(self) -> List[Dict[str, Any]]:
        """Create minimal dataset for testing."""
        return [
            {
                "movie_id": "1",
                "title": "The Shawshank Redemption",
                "synopsis": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
                "release_year": 1994,
                "runtime_mins": 142,
                "genres": ["Drama"],
                "ratings": {"average": 9.3, "count": 2500000},
            },
            {
                "movie_id": "2",
                "title": "The Godfather",
                "synopsis": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
                "release_year": 1972,
                "runtime_mins": 175,
                "genres": ["Crime", "Drama"],
                "ratings": {"average": 9.2, "count": 1800000},
            },
        ]

    def _run_validation_step(
        self, movies_data: List[Dict[str, Any]]
    ) -> Tuple[ValidationResult, List[Movie]]:
        """Run data validation step."""
        self.state.start_step("data_validation")
        self.state.update_memory_usage()

        try:
            validation_result, valid_movies = self.validator.validate_dataset(
                movies_data
            )

            self.state.valid_movies = len(valid_movies)
            self.state.complete_step("data_validation", validation_result)

            # Check if we meet minimum requirements
            min_movies = self.config.config["processing"]["min_movies"]
            if len(valid_movies) < min_movies:
                raise ValueError(
                    f"Too few valid movies: {len(valid_movies)} < {min_movies}"
                )

            return validation_result, valid_movies

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise

    def _run_bias_detection_step(self, valid_movies: List[Movie]) -> BiasMetrics:
        """Run bias detection step."""
        self.state.start_step("bias_detection")

        try:
            bias_metrics = self.validator.detect_bias(valid_movies)
            self.state.complete_step("bias_detection", bias_metrics)

            return bias_metrics

        except Exception as e:
            logger.error(f"Bias detection failed: {e}")
            raise

    def _run_feature_engineering_step(
        self, valid_movies: List[Movie]
    ) -> Dict[str, Any]:
        """Run feature engineering step."""
        self.state.start_step("feature_engineering")
        self.state.update_memory_usage()

        try:
            split_data = self.feature_engineer.create_train_test_split(
                valid_movies,
                test_size=0.2,
                stratify_by_genre=len(valid_movies) > 20,
                random_state=42,
            )

            self.state.complete_step("feature_engineering", split_data)

            return split_data

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

    def _run_export_step(
        self,
        split_data: Dict[str, Any],
        valid_movies: List[Movie],
        validation_result: ValidationResult,
        bias_metrics: BiasMetrics,
        export_formats: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Run data export step."""
        self.state.start_step("data_export")

        try:
            if export_formats is None:
                export_formats = ["json", "csv", "parquet", "numpy"]

            exported_files = {}

            # Export features
            train_files = self.feature_engineer.export_features(
                split_data["train"]["features"],
                split_data["train"]["movies"],
                split_data["feature_names"],
                "data/processed/train",
                export_formats,
            )

            test_files = self.feature_engineer.export_features(
                split_data["test"]["features"],
                split_data["test"]["movies"],
                split_data["feature_names"],
                "data/processed/test",
                export_formats,
            )

            exported_files.update({f"train_{k}": v for k, v in train_files.items()})
            exported_files.update({f"test_{k}": v for k, v in test_files.items()})

            # Export validation report
            report_path = "data/reports/validation_report.html"
            self.validator.generate_html_report(
                validation_result, bias_metrics, valid_movies, report_path
            )
            exported_files["validation_report"] = report_path

            self.state.exported_formats = export_formats
            self.state.complete_step("data_export", exported_files)

            return exported_files

        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise

    def _generate_pipeline_report(
        self,
        validation_result: ValidationResult,
        bias_metrics: BiasMetrics,
        split_data: Dict[str, Any],
    ) -> None:
        """Generate final pipeline report."""
        report_path = Path("data/reports/pipeline_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "pipeline_execution": self.state.get_summary(),
            "configuration": self.config.config,
            "validation_summary": {
                "is_valid": validation_result.is_valid,
                "error_count": validation_result.error_count,
                "warning_count": validation_result.warning_count,
                "completeness_rate": validation_result.summary.get(
                    "completeness_rate", 0
                ),
            },
            "bias_analysis": {
                "overall_bias_score": bias_metrics.overall_bias_score,
                "recommendations_count": len(bias_metrics.recommendations),
                "genre_diversity": bias_metrics.genre_diversity.get("unique_genres", 0),
                "geographic_diversity": bias_metrics.geographic_distribution.get(
                    "unique_countries", 0
                ),
            },
            "feature_engineering": {
                "total_features": len(split_data["feature_names"]),
                "train_samples": split_data["train"]["features"].shape[0],
                "test_samples": split_data["test"]["features"].shape[0],
                "feature_categories": self._count_feature_categories(
                    split_data["feature_names"]
                ),
            },
            "generated_at": datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Pipeline report generated: {report_path}")

    def _count_feature_categories(self, feature_names: List[str]) -> Dict[str, int]:
        """Count features by category."""
        categories = {
            "text": len([n for n in feature_names if n.startswith("text_")]),
            "genre": len([n for n in feature_names if n.startswith("genre_")]),
            "temporal": len([n for n in feature_names if n.startswith("decade_")]),
            "runtime": len([n for n in feature_names if n.startswith("runtime_")]),
            "rating": len([n for n in feature_names if n.startswith("rating_")]),
            "cast_crew": len(
                [n for n in feature_names if n.startswith(("cast_", "crew_"))]
            ),
        }

        return categories


def create_cli() -> argparse.ArgumentParser:
    """Create command-line interface."""
    parser = argparse.ArgumentParser(
        description="MovieRecs Data Pipeline - Complete data processing pipeline"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/data_config.yaml",
        help="Configuration file path",
    )

    parser.add_argument(
        "--step",
        choices=["all", "download", "validate", "features", "export"],
        default="all",
        help="Pipeline step to run",
    )

    parser.add_argument(
        "--force-refresh", action="store_true", help="Force re-download of data"
    )

    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download step"
    )

    parser.add_argument(
        "--export-formats",
        nargs="+",
        choices=["csv", "json", "parquet", "numpy"],
        default=["json", "csv"],
        help="Export formats",
    )

    parser.add_argument("--output-dir", type=str, help="Override output directory")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser


def main():
    """Main CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()

    # Set environment overrides
    if args.output_dir:
        os.environ["MOVIERECS_OUTPUT_DIR"] = args.output_dir

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize pipeline
        pipeline = DataPipeline(args.config)

        if args.dry_run:
            print("Dry run - would execute the following steps:")
            print(
                f"1. Data acquisition (force_refresh={args.force_refresh}, skip={args.skip_download})"
            )
            print(f"2. Data validation")
            print(f"3. Bias detection")
            print(f"4. Feature engineering")
            print(f"5. Data export (formats={args.export_formats})")
            return

        # Run pipeline
        if args.step == "all":
            results = pipeline.run_full_pipeline(
                force_refresh=args.force_refresh,
                skip_download=args.skip_download,
                export_formats=args.export_formats,
            )
        else:
            # Individual step execution would be implemented here
            raise NotImplementedError(
                f"Individual step execution not yet implemented: {args.step}"
            )

        # Print results
        if results["status"] == "success":
            print("\n🎉 Pipeline completed successfully!")
            summary = results["summary"]
            print(
                f"📊 Processed {summary['total_movies']} movies ({summary['valid_movies']} valid)"
            )
            print(f"⏱️  Execution time: {summary['execution_time_seconds']:.1f} seconds")
            print(f"💾 Memory peak: {summary['memory_peak_mb']:.1f} MB")
            print(f"🎯 Features generated: {results['feature_info']['feature_count']}")
            print(f"📁 Exported files: {len(results['exported_files'])}")

            if args.verbose:
                print("\n📋 Detailed Results:")
                print(json.dumps(results["summary"], indent=2))
        else:
            print(f"\n❌ Pipeline failed: {results['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
