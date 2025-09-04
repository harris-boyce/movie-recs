"""
Integration tests for the complete MovieRecs data pipeline.

Tests the entire pipeline from data acquisition through feature export,
including performance benchmarks and data quality validation.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

import psutil

from src.data_prep import DataPipeline
from src.preprocessing import FeatureEngineer
from src.validation import DataValidator


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for complete pipeline functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for all integration tests."""
        cls.start_time = time.time()
        cls.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

    @classmethod
    def tearDownClass(cls):
        """Clean up and report performance metrics."""
        total_time = time.time() - cls.start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - cls.initial_memory

        print(f"\n📊 Integration Test Performance:")
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"💾 Memory usage increase: {memory_increase:.1f} MB")
        print(f"🏁 Final memory usage: {final_memory:.1f} MB")

    def setUp(self):
        """Set up individual test case."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_start = time.time()

    def tearDown(self):
        """Clean up individual test case."""
        test_duration = time.time() - self.test_start
        print(f"⏱️  Test '{self._testMethodName}' took {test_duration:.2f}s")

    def test_full_pipeline_with_sample_data(self):
        """Test complete pipeline with realistic sample data."""
        print("\n🧪 Testing full pipeline with sample data...")

        # Load sample data
        sample_data_path = Path(__file__).parent / "fixtures" / "sample_movies.json"
        with open(sample_data_path, "r") as f:
            sample_data = json.load(f)

        # Create test configuration
        config = {
            "data_sources": {
                "primary": "local_fallback",
                "local_fallback": {"path": str(sample_data_path)},
            },
            "download": {"cache_dir": str(self.temp_dir / "cache")},
            "processing": {"min_movies": 3, "max_movies": 20},
            "quality_thresholds": {"completeness_min": 0.8},
            "feature_engineering": {
                "use_genre_encoding": True,
                "use_decade_features": True,
                "use_runtime_binning": True,
                "use_rating_features": True,
                "use_cast_features": True,
            },
            "logging": {
                "level": "WARNING",  # Reduce noise
                "file": str(self.temp_dir / "pipeline.log"),
            },
        }

        config_file = self.temp_dir / "config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Run pipeline
        pipeline = DataPipeline(str(config_file))
        results = pipeline.run_full_pipeline(skip_download=True, export_formats=["json", "csv"])

        # Validate results
        self.assertEqual(results["status"], "success")
        self.assertGreater(results["summary"]["valid_movies"], 0)
        self.assertGreater(results["feature_info"]["feature_count"], 10)
        self.assertIn("train_json", results["exported_files"])

        # Check that files were actually created
        train_json = results["exported_files"]["train_json"]
        self.assertTrue(Path(train_json).exists())

        print(f"✅ Pipeline processed {results['summary']['valid_movies']} movies")
        print(f"🎯 Generated {results['feature_info']['feature_count']} features")

    def test_pipeline_with_malformed_data(self):
        """Test pipeline behavior with malformed data."""
        print("\n🧪 Testing pipeline with malformed data...")

        # Load malformed data
        malformed_data_path = Path(__file__).parent / "fixtures" / "malformed_data.json"

        config = {
            "data_sources": {
                "primary": "local_fallback",
                "local_fallback": {"path": str(malformed_data_path)},
            },
            "processing": {
                "min_movies": 1,  # Allow pipeline to run even with few valid movies
                "max_movies": 10,
            },
            "quality_thresholds": {"completeness_min": 0.5},  # Lower threshold to allow some processing
            "logging": {"level": "ERROR"},  # Only show errors
        }

        config_file = self.temp_dir / "malformed_config.yaml"
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Run pipeline - should handle errors gracefully
        pipeline = DataPipeline(str(config_file))
        results = pipeline.run_full_pipeline(skip_download=True, export_formats=["json"])

        # Pipeline might fail or succeed with warnings
        self.assertIn(results["status"], ["success", "failed"])

        if results["status"] == "failed":
            self.assertIn("error", results)
            print(f"⚠️  Pipeline appropriately failed: {results['error'][:100]}...")
        else:
            print(f"✅ Pipeline handled malformed data: {results['summary']['valid_movies']} valid movies")

    def test_bias_detection_with_biased_dataset(self):
        """Test bias detection with intentionally biased data."""
        print("\n🧪 Testing bias detection with biased dataset...")

        # Load biased test data
        bias_data_path = Path(__file__).parent / "fixtures" / "bias_test_data.json"
        with open(bias_data_path, "r") as f:
            bias_data = json.load(f)

        # Run validation and bias detection
        validator = DataValidator()
        validation_result, valid_movies = validator.validate_dataset(bias_data)

        self.assertGreater(len(valid_movies), 0, "Should have some valid movies")

        # Detect bias
        bias_metrics = validator.detect_bias(valid_movies)

        # This dataset is intentionally biased, so bias score should be high
        self.assertGreater(
            bias_metrics.overall_bias_score,
            0.3,
            "Should detect high bias in test dataset",
        )

        # Should have recommendations
        self.assertGreater(
            len(bias_metrics.recommendations),
            0,
            "Should provide bias reduction recommendations",
        )

        print(f"⚖️  Detected bias score: {bias_metrics.overall_bias_score:.2f}")
        print(f"📋 Generated {len(bias_metrics.recommendations)} recommendations")

        # Check specific bias types
        self.assertIn("genre_diversity", bias_metrics.__dict__)
        self.assertIn("geographic_distribution", bias_metrics.__dict__)

    def test_feature_engineering_scalability(self):
        """Test feature engineering with larger dataset."""
        print("\n🧪 Testing feature engineering scalability...")

        # Generate larger synthetic dataset
        large_dataset = []
        genres = ["Drama", "Comedy", "Action", "Science Fiction", "Horror", "Romance"]

        for i in range(50):
            movie = {
                "movie_id": f"scale_{i}",
                "title": f"Scalability Test Movie {i}",
                "synopsis": f"A comprehensive test movie number {i} with sufficient length to pass validation requirements and provide realistic text processing challenges for the feature engineering pipeline.",
                "release_year": 1990 + (i % 35),
                "runtime_mins": 90 + (i % 120),
                "genres": [genres[i % len(genres)]],
                "ratings": {"average": 5.0 + (i % 50) / 10, "count": 1000 + i * 50},
            }
            large_dataset.append(movie)

        # Time feature engineering
        start_time = time.time()

        validator = DataValidator()
        validation_result, valid_movies = validator.validate_dataset(large_dataset)

        engineer = FeatureEngineer()
        features, feature_names = engineer.process_movies(valid_movies, include_text=True)

        processing_time = time.time() - start_time

        # Performance assertions
        self.assertLess(
            processing_time,
            30.0,
            f"Feature engineering too slow: {processing_time:.2f}s",
        )

        movies_per_sec = len(valid_movies) / processing_time
        self.assertGreater(
            movies_per_sec,
            1.0,
            f"Processing rate too slow: {movies_per_sec:.2f} movies/sec",
        )

        # Quality assertions
        self.assertEqual(len(valid_movies), 50, "All synthetic movies should be valid")
        self.assertGreater(features.shape[1], 50, "Should generate substantial features")

        print(f"⏱️  Processed {len(valid_movies)} movies in {processing_time:.2f}s")
        print(f"🚀 Rate: {movies_per_sec:.1f} movies/sec")
        print(f"🎯 Generated {features.shape[1]} features")

    def test_memory_usage_under_load(self):
        """Test memory usage during pipeline execution."""
        print("\n🧪 Testing memory usage under load...")

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = initial_memory

        # Create moderately large dataset
        test_data = []
        for i in range(100):
            test_data.append(
                {
                    "movie_id": f"mem_{i}",
                    "title": f"Memory Test Movie {i}",
                    "synopsis": f"Memory usage test movie {i} with extended synopsis content to simulate realistic text processing memory requirements during feature engineering and validation phases of the data pipeline execution.",
                    "release_year": 2000 + (i % 25),
                    "runtime_mins": 90 + (i % 90),
                    "genres": ["Drama", "Action", "Comedy"][i % 3 : i % 3 + 1],
                    "ratings": {"average": 6.0 + (i % 40) / 10, "count": 500 + i * 20},
                }
            )

        # Monitor memory during processing
        validator = DataValidator()
        validation_result, valid_movies = validator.validate_dataset(test_data)

        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current_memory)

        if valid_movies:
            engineer = FeatureEngineer()
            features, feature_names = engineer.process_movies(valid_movies, include_text=True)

            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)

        memory_increase = peak_memory - initial_memory

        # Memory assertions (should not exceed 200MB increase)
        self.assertLess(
            memory_increase,
            200.0,
            f"Memory usage too high: {memory_increase:.1f} MB increase",
        )

        print(f"💾 Memory increase: {memory_increase:.1f} MB")
        print(f"📊 Peak memory: {peak_memory:.1f} MB")

    def test_concurrent_pipeline_execution(self):
        """Test pipeline behavior under concurrent execution."""
        print("\n🧪 Testing concurrent pipeline components...")

        # This test simulates concurrent-like behavior by rapidly executing
        # multiple pipeline components in sequence

        sample_data_path = Path(__file__).parent / "fixtures" / "sample_movies.json"
        with open(sample_data_path, "r") as f:
            sample_data = json.load(f)

        # Run multiple validation cycles rapidly
        validator = DataValidator()

        for i in range(5):
            start_time = time.time()
            validation_result, valid_movies = validator.validate_dataset(sample_data)
            cycle_time = time.time() - start_time

            self.assertGreater(len(valid_movies), 0)
            self.assertLess(cycle_time, 2.0, f"Validation cycle {i} too slow: {cycle_time:.2f}s")

        # Run multiple feature engineering cycles
        if len(sample_data) > 0:
            validator = DataValidator()
            validation_result, valid_movies = validator.validate_dataset(sample_data[:3])  # Smaller subset

            engineer = FeatureEngineer()

            for i in range(3):
                start_time = time.time()
                features, feature_names = engineer.process_movies(
                    valid_movies, include_text=False
                )  # Faster without text
                cycle_time = time.time() - start_time

                self.assertGreater(features.shape[1], 0)
                self.assertLess(cycle_time, 3.0, f"Feature cycle {i} too slow: {cycle_time:.2f}s")

        print("✅ Concurrent execution simulation completed")

    def test_export_format_consistency(self):
        """Test consistency across different export formats."""
        print("\n🧪 Testing export format consistency...")

        sample_data_path = Path(__file__).parent / "fixtures" / "sample_movies.json"
        with open(sample_data_path, "r") as f:
            sample_data = json.load(f)

        # Process data
        validator = DataValidator()
        validation_result, valid_movies = validator.validate_dataset(sample_data)

        engineer = FeatureEngineer()
        features, feature_names = engineer.process_movies(valid_movies[:3], include_text=False)

        # Export in multiple formats
        export_dir = self.temp_dir / "exports"
        exported_files = engineer.export_features(
            features,
            valid_movies[:3],
            feature_names,
            str(export_dir),
            format_types=["json", "numpy"],  # Test available formats
        )

        # Verify exports
        self.assertIn("json", exported_files)
        self.assertTrue(Path(exported_files["json"]).exists())

        if "numpy" in exported_files:
            self.assertTrue(Path(exported_files["numpy"]).exists())

            # Load and verify numpy export
            import numpy as np

            npz_data = np.load(exported_files["numpy"])

            self.assertEqual(npz_data["features"].shape, features.shape)
            self.assertEqual(len(npz_data["feature_names"]), len(feature_names))

        # Load and verify JSON export
        with open(exported_files["json"], "r") as f:
            json_data = json.load(f)

        self.assertIn("features", json_data)
        self.assertIn("feature_names", json_data)
        self.assertEqual(len(json_data["features"]), features.shape[0])
        self.assertEqual(len(json_data["feature_names"]), len(feature_names))

        print(f"📁 Exported {len(exported_files)} format(s)")
        print("✅ Export format consistency verified")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""

    def setUp(self):
        """Set up performance test environment."""
        self.benchmark_results = {}

    def test_validation_performance_benchmark(self):
        """Benchmark validation performance."""
        print("\n🏃 Running validation performance benchmark...")

        # Create benchmark dataset
        sizes = [10, 50, 100]

        for size in sizes:
            test_data = []
            for i in range(size):
                test_data.append(
                    {
                        "movie_id": f"bench_{i}",
                        "title": f"Benchmark Movie {i}",
                        "synopsis": f"Benchmark test movie {i} with adequate synopsis length for validation testing purposes.",
                        "release_year": 2000 + (i % 25),
                        "genres": ["Drama"],
                        "ratings": {"average": 7.0, "count": 1000},
                    }
                )

            # Benchmark validation
            validator = DataValidator()

            start_time = time.time()
            validation_result, valid_movies = validator.validate_dataset(test_data)
            validation_time = time.time() - start_time

            movies_per_sec = size / validation_time

            self.benchmark_results[f"validation_{size}"] = {
                "time": validation_time,
                "rate": movies_per_sec,
                "valid_count": len(valid_movies),
            }

            print(f"📊 {size} movies: {validation_time:.2f}s ({movies_per_sec:.1f} movies/sec)")

            # Performance assertions
            self.assertLess(validation_time, size * 0.1, f"Validation too slow for {size} movies")

    def test_feature_engineering_benchmark(self):
        """Benchmark feature engineering performance."""
        print("\n🏃 Running feature engineering benchmark...")

        # Create test dataset
        test_data = []
        for i in range(20):
            test_data.append(
                {
                    "movie_id": f"feat_{i}",
                    "title": f"Feature Test Movie {i}",
                    "synopsis": f"Feature engineering benchmark movie {i} with comprehensive synopsis content for text processing evaluation.",
                    "release_year": 1990 + i,
                    "runtime_mins": 90 + (i * 2),
                    "genres": ["Drama", "Comedy", "Action"][i % 3 : i % 3 + 1],
                    "ratings": {"average": 6.0 + (i % 4), "count": 1000 + i * 100},
                }
            )

        # Process for feature engineering
        validator = DataValidator()
        validation_result, valid_movies = validator.validate_dataset(test_data)

        engineer = FeatureEngineer()

        # Benchmark with text features
        start_time = time.time()
        features_with_text, names_with_text = engineer.process_movies(valid_movies, include_text=True)
        text_time = time.time() - start_time

        # Benchmark without text features
        start_time = time.time()
        features_no_text, names_no_text = engineer.process_movies(valid_movies, include_text=False)
        no_text_time = time.time() - start_time

        print(f"🎯 With text: {text_time:.2f}s, {features_with_text.shape[1]} features")
        print(f"🎯 Without text: {no_text_time:.2f}s, {features_no_text.shape[1]} features")

        # Performance assertions
        self.assertLess(text_time, 10.0, "Text feature engineering too slow")
        self.assertLess(no_text_time, 2.0, "Non-text feature engineering too slow")
        self.assertGreater(features_with_text.shape[1], features_no_text.shape[1])


if __name__ == "__main__":
    # Run with verbose output for CI
    unittest.main(verbosity=2)
