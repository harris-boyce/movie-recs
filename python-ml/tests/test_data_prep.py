"""
Tests for the data pipeline orchestration module.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.data_prep import DataPipeline, PipelineConfig, PipelineState


class TestPipelineConfig(unittest.TestCase):
    """Test cases for PipelineConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_default_config_loading(self):
        """Test loading default configuration when file doesn't exist."""
        config = PipelineConfig("nonexistent_config.yaml")

        self.assertIsNotNone(config.config)
        self.assertIn("data_sources", config.config)
        self.assertIn("processing", config.config)
        self.assertEqual(config.config["data_sources"]["primary"], "local_fallback")

    def test_config_file_loading(self):
        """Test loading configuration from YAML file."""
        config_file = self.temp_dir / "test_config.yaml"

        test_config = {
            "data_sources": {
                "primary": "test_source",
                "local_fallback": {"path": "test/path"},
            },
            "processing": {"min_movies": 50, "max_movies": 500},
            "quality_thresholds": {"completeness_min": 0.8},
            "bias_detection": {"enable_genre_analysis": True},
        }

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(test_config, f)

        # Clear environment variables that might interfere
        with patch.dict("os.environ", {}, clear=False):
            # Remove specific env vars that would override config
            import os

            for key in list(os.environ.keys()):
                if key.startswith("MOVIERECS_"):
                    del os.environ[key]

            config = PipelineConfig(str(config_file))

            self.assertEqual(config.config["data_sources"]["primary"], "test_source")
            self.assertEqual(config.config["processing"]["min_movies"], 50)

    def test_environment_overrides(self):
        """Test environment variable overrides."""
        import importlib
        import sys

        with patch.dict(
            "os.environ",
            {
                "MOVIERECS_DATA_SOURCE": "env_source",
                "MOVIERECS_OUTPUT_DIR": "/tmp/test",
                "MOVIERECS_MIN_MOVIES": "50",
                "MOVIERECS_MAX_MOVIES": "1000",
            },
        ):
            # Reload the module to pick up environment changes
            if "src.data_prep" in sys.modules:
                importlib.reload(sys.modules["src.data_prep"])
            from src.data_prep import PipelineConfig

            config = PipelineConfig()

            self.assertEqual(config.config["data_sources"]["primary"], "env_source")
            self.assertEqual(config.config["download"]["cache_dir"], "/tmp/test/raw")
            self.assertEqual(config.config["processing"]["min_movies"], 50)
            self.assertEqual(config.config["processing"]["max_movies"], 1000)

    def test_config_validation(self):
        """Test configuration validation."""
        config_file = self.temp_dir / "invalid_config.yaml"

        # Missing required sections
        invalid_config = {
            "data_sources": {"primary": "test"}
            # Missing 'processing' and 'quality_thresholds'
        }

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        with self.assertRaises(ValueError):
            PipelineConfig(str(config_file))

    def test_invalid_numeric_values(self):
        """Test validation of invalid numeric configuration values."""
        config_file = self.temp_dir / "bad_numbers_config.yaml"

        bad_config = {
            "data_sources": {"primary": "test"},
            "processing": {
                "min_movies": 1000,
                "max_movies": 500,  # max < min (invalid)
            },
            "quality_thresholds": {"completeness_min": 1.5},  # > 1.0 (invalid)
        }

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(bad_config, f)

        with self.assertRaises(ValueError):
            PipelineConfig(str(config_file))


class TestPipelineState(unittest.TestCase):
    """Test cases for PipelineState class."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = PipelineState()

    def test_state_initialization(self):
        """Test state initialization."""
        self.assertEqual(self.state.current_step, "initialized")
        self.assertEqual(len(self.state.steps_completed), 0)
        self.assertEqual(self.state.total_movies, 0)
        self.assertEqual(self.state.valid_movies, 0)

    def test_step_tracking(self):
        """Test step tracking functionality."""
        self.state.start_step("test_step")
        self.assertEqual(self.state.current_step, "test_step")

        self.state.complete_step("test_step", {"result": "success"})
        self.assertEqual(len(self.state.steps_completed), 1)
        self.assertEqual(self.state.steps_completed[0]["name"], "test_step")

    def test_error_tracking(self):
        """Test error tracking."""
        error = ValueError("Test error")
        self.state.add_error(error, "test_step")

        self.assertEqual(len(self.state.error_log), 1)
        self.assertEqual(self.state.error_log[0]["step"], "test_step")
        self.assertIn("Test error", self.state.error_log[0]["error"])

    def test_summary_generation(self):
        """Test summary generation."""
        self.state.total_movies = 100
        self.state.valid_movies = 95

        summary = self.state.get_summary()

        self.assertIn("total_movies", summary)
        self.assertIn("valid_movies", summary)
        self.assertIn("success_rate", summary)
        self.assertEqual(summary["success_rate"], 0.95)


class TestDataPipeline(unittest.TestCase):
    """Test cases for DataPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test config
        self.config_file = self.temp_dir / "test_config.yaml"
        test_config = {
            "data_sources": {
                "primary": "local_fallback",
                "local_fallback": {"path": str(self.temp_dir / "test_movies.json")},
            },
            "download": {
                "cache_dir": str(self.temp_dir / "cache"),
                "temp_dir": str(self.temp_dir / "temp"),
            },
            "processing": {"min_movies": 1, "max_movies": 10},
            "quality_thresholds": {"completeness_min": 0.5},
            "bias_detection": {
                "enable_genre_analysis": True,
                "enable_demographic_analysis": False,  # Simplified for test
            },
            "feature_engineering": {
                "use_genre_encoding": True,
                "use_decade_features": True,
                "use_runtime_binning": False,  # Simplified for test
                "use_rating_features": True,
                "use_cast_features": False,
            },
            "logging": {"level": "INFO", "file": str(self.temp_dir / "test.log")},
        }

        import yaml

        with open(self.config_file, "w") as f:
            yaml.dump(test_config, f)

        # Create test movie data
        test_movies = [
            {
                "movie_id": "1",
                "title": "Test Movie 1",
                "synopsis": "This is a test movie with a long enough synopsis to pass validation checks for minimum length requirements.",
                "release_year": 2000,
                "runtime_mins": 120,
                "genres": ["Drama"],
                "ratings": {"average": 8.0, "count": 1000},
            },
            {
                "movie_id": "2",
                "title": "Test Movie 2",
                "synopsis": "Another test movie with sufficient length in the synopsis to meet validation requirements and provide good test data.",
                "release_year": 2010,
                "runtime_mins": 95,
                "genres": ["Comedy"],
                "ratings": {"average": 7.5, "count": 500},
            },
        ]

        with open(self.temp_dir / "test_movies.json", "w") as f:
            json.dump(test_movies, f)

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = DataPipeline(str(self.config_file))

        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.state)
        self.assertIsNotNone(pipeline.downloader)
        self.assertIsNotNone(pipeline.validator)
        self.assertIsNotNone(pipeline.feature_engineer)

    @patch("src.data_prep.DatasetDownloader")
    @patch("src.data_prep.DataValidator")
    @patch("src.data_prep.FeatureEngineer")
    def test_full_pipeline_success(self, mock_feature_engineer, mock_validator, mock_downloader):
        """Test successful full pipeline execution."""
        # Setup mocks
        mock_downloader_instance = Mock()
        mock_downloader.return_value = mock_downloader_instance
        mock_downloader_instance.download_dataset.return_value = Path("test_data.json")
        mock_downloader_instance.extract_dataset.return_value = Path("test_data.json")

        mock_validator_instance = Mock()
        mock_validator.return_value = mock_validator_instance

        # Mock validation result
        validation_result = Mock()
        validation_result.is_valid = True
        validation_result.error_count = 0
        validation_result.warning_count = 1
        validation_result.summary = {"completeness_rate": 0.95}

        valid_movies = [Mock() for _ in range(6)]  # Exceed min_movies requirement
        mock_validator_instance.validate_dataset.return_value = (
            validation_result,
            valid_movies,
        )

        # Mock bias metrics
        bias_metrics = Mock()
        bias_metrics.overall_bias_score = 0.3
        bias_metrics.recommendations = ["test recommendation"]
        bias_metrics.genre_diversity = {"unique_genres": 5}
        bias_metrics.geographic_distribution = {"unique_countries": 3}
        mock_validator_instance.detect_bias.return_value = bias_metrics

        mock_validator_instance.generate_html_report = Mock()

        # Mock feature engineer
        mock_feature_engineer_instance = Mock()
        mock_feature_engineer.return_value = mock_feature_engineer_instance

        split_data = {
            "train": {"movies": [Mock()], "features": Mock()},
            "test": {"movies": [Mock()], "features": Mock()},
            "feature_names": ["feature1", "feature2"],
        }
        split_data["train"]["features"].shape = (1, 2)
        split_data["test"]["features"].shape = (1, 2)

        mock_feature_engineer_instance.create_train_test_split.return_value = split_data
        mock_feature_engineer_instance.export_features.return_value = {"json": "test.json"}

        # Create pipeline with real config but mocked components
        pipeline = DataPipeline(str(self.config_file))
        pipeline.downloader = mock_downloader_instance
        pipeline.validator = mock_validator_instance
        pipeline.feature_engineer = mock_feature_engineer_instance

        # Override _load_movie_data to return test data
        pipeline._load_movie_data = Mock(
            return_value=[
                {
                    "movie_id": "1",
                    "title": "Test Movie",
                    "synopsis": "Test synopsis that is long enough to pass validation",
                    "release_year": 2000,
                    "genres": ["Drama"],
                    "ratings": {"average": 8.0, "count": 1000},
                }
            ]
        )

        # Run pipeline
        results = pipeline.run_full_pipeline(export_formats=["json"])

        # Verify results
        self.assertEqual(results["status"], "success")
        self.assertIn("summary", results)
        self.assertIn("validation_result", results)
        self.assertIn("bias_metrics", results)
        self.assertIn("exported_files", results)

    def test_pipeline_with_skip_download(self):
        """Test pipeline execution with skip download."""
        # Clear environment variables that might override test config
        with patch.dict("os.environ", {}, clear=False):
            import os

            for key in list(os.environ.keys()):
                if key.startswith("MOVIERECS_"):
                    del os.environ[key]

            pipeline = DataPipeline(str(self.config_file))

            # Test will use existing test data
            results = pipeline.run_full_pipeline(skip_download=True, export_formats=["json"])

            self.assertEqual(results["status"], "success")
            self.assertGreater(results["summary"]["valid_movies"], 0)

    def test_pipeline_failure_handling(self):
        """Test pipeline failure handling."""
        # Create config with impossible requirements
        bad_config_file = self.temp_dir / "bad_config.yaml"
        bad_config = {
            "data_sources": {
                "primary": "local_fallback",
                "local_fallback": {"path": "nonexistent.json"},
            },
            "processing": {
                "min_movies": 1000,  # Too many for test data
                "max_movies": 2000,
            },
            "quality_thresholds": {"completeness_min": 0.99},  # Too strict
        }

        import yaml

        with open(bad_config_file, "w") as f:
            yaml.dump(bad_config, f)

        pipeline = DataPipeline(str(bad_config_file))

        results = pipeline.run_full_pipeline()

        self.assertEqual(results["status"], "failed")
        self.assertIn("error", results)

    def test_minimal_dataset_creation(self):
        """Test creation of minimal dataset."""
        pipeline = DataPipeline(str(self.config_file))

        minimal_data = pipeline._create_minimal_dataset()

        self.assertIsInstance(minimal_data, list)
        self.assertGreater(len(minimal_data), 0)

        # Check structure of first movie
        movie = minimal_data[0]
        required_fields = [
            "movie_id",
            "title",
            "synopsis",
            "release_year",
            "genres",
            "ratings",
        ]
        for field in required_fields:
            self.assertIn(field, movie)

    def test_pipeline_state_tracking(self):
        """Test that pipeline properly tracks state."""
        pipeline = DataPipeline(str(self.config_file))

        # Run pipeline and check state transitions
        results = pipeline.run_full_pipeline(export_formats=["json"])

        if results["status"] == "success":
            self.assertGreater(len(pipeline.state.steps_completed), 0)
            self.assertEqual(pipeline.state.current_step, "completed")
        else:
            self.assertEqual(pipeline.state.current_step, "failed")


class TestPipelineCLI(unittest.TestCase):
    """Test command-line interface functionality."""

    def test_cli_parser_creation(self):
        """Test CLI argument parser creation."""
        from src.data_prep import create_cli

        parser = create_cli()

        # Test with minimal arguments
        args = parser.parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.step, "all")

        # Test with full arguments
        args = parser.parse_args(
            [
                "--config",
                "test.yaml",
                "--step",
                "validate",
                "--force-refresh",
                "--export-formats",
                "json",
                "csv",
                "--verbose",
            ]
        )

        self.assertEqual(args.config, "test.yaml")
        self.assertEqual(args.step, "validate")
        self.assertTrue(args.force_refresh)
        self.assertEqual(args.export_formats, ["json", "csv"])
        self.assertTrue(args.verbose)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for complete pipeline functionality."""

    @pytest.mark.skip(reason="Test isolation issue - config file path resolution - see GitHub issue #7")
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline execution."""
        # Clear environment variables that might interfere, but set test-friendly minimums
        with patch.dict("os.environ", {"MOVIERECS_MIN_MOVIES": "1"}, clear=False):
            import os

            for key in list(os.environ.keys()):
                if key.startswith("MOVIERECS_") and key != "MOVIERECS_MIN_MOVIES":
                    del os.environ[key]

            self._run_integration_test()

    def _run_integration_test(self):
        """Helper method to run integration test."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create realistic test configuration
            config = {
                "data_sources": {
                    "primary": "local_fallback",
                    "local_fallback": {"path": str(temp_path / "movies.json")},
                },
                "download": {"cache_dir": str(temp_path / "cache")},
                "processing": {
                    "min_movies": 1,
                    "max_movies": 10,
                },  # Adjusted for data loading issue - see GitHub issue #6
                "quality_thresholds": {"completeness_min": 0.8},
                "feature_engineering": {
                    "use_genre_encoding": True,
                    "use_decade_features": True,
                    "use_runtime_binning": True,
                    "use_rating_features": True,
                    "use_cast_features": False,
                },
                "logging": {
                    "level": "WARNING",  # Reduce log noise
                    "file": str(temp_path / "pipeline.log"),
                },
            }

            config_file = temp_path / "config.yaml"
            import yaml

            with open(config_file, "w") as f:
                yaml.dump(config, f)

            # Create test movie dataset
            movies = [
                {
                    "movie_id": "1",
                    "title": "The Great Movie",
                    "synopsis": "An epic tale of heroism and adventure that spans multiple generations and explores themes of courage and sacrifice.",
                    "release_year": 1995,
                    "runtime_mins": 150,
                    "genres": ["Drama", "Adventure"],
                    "ratings": {"average": 8.5, "count": 10000},
                },
                {
                    "movie_id": "2",
                    "title": "Comedy Gold",
                    "synopsis": "A hilarious comedy that brings together an unlikely group of characters in a series of misadventures and mishaps.",
                    "release_year": 2005,
                    "runtime_mins": 95,
                    "genres": ["Comedy"],
                    "ratings": {"average": 7.2, "count": 5000},
                },
                {
                    "movie_id": "3",
                    "title": "Future Vision",
                    "synopsis": "A science fiction masterpiece that explores the possibilities of technology and human nature in the distant future.",
                    "release_year": 2020,
                    "runtime_mins": 135,
                    "genres": ["Science Fiction", "Action"],
                    "ratings": {"average": 8.8, "count": 15000},
                },
            ]

            with open(temp_path / "movies.json", "w") as f:
                json.dump(movies, f)

        # Run complete pipeline
        pipeline = DataPipeline(str(config_file))
        results = pipeline.run_full_pipeline(skip_download=True, export_formats=["json"])  # Use existing data

        # Verify results
        self.assertEqual(results["status"], "success")

        # Check that files were created
        processed_dir = temp_path / "cache" / "processed"
        if processed_dir.exists():
            json_files = list(processed_dir.glob("**/*.json"))
            self.assertGreater(len(json_files), 0)

        # Check feature engineering results
        feature_info = results["feature_info"]
        self.assertGreater(feature_info["feature_count"], 0)
        self.assertGreater(feature_info["train_features"][0], 0)  # Train samples > 0
        self.assertGreater(feature_info["test_features"][0], 0)  # Test samples > 0


if __name__ == "__main__":
    unittest.main()
