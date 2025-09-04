"""
Tests for the data acquisition module.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
from src.acquisition import (
    DatasetDownloader,
    DataAcquisitionError,
    validate_license_compliance,
)


class TestDatasetDownloader(unittest.TestCase):
    """Test cases for DatasetDownloader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test configuration
        self.test_config = {
            "data_sources": {
                "primary": "local_fallback",
                "backup": "local_fallback",
                "local_fallback": {
                    "path": "data/raw/fallback_movies.json",
                    "description": "Test fallback dataset",
                },
            },
            "download": {
                "cache_dir": str(self.temp_path / "cache"),
                "temp_dir": str(self.temp_path / "temp"),
                "timeout_seconds": 30,
                "max_retries": 2,
                "retry_delay_seconds": 1,
                "chunk_size": 1024,
                "verify_checksums": True,
            },
            "versioning": {
                "track_dataset_versions": True,
                "metadata_file": str(self.temp_path / "metadata.json"),
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": str(self.temp_path / "pipeline.log"),
            },
        }

    def test_config_loading(self):
        """Test configuration loading."""
        # Create test config file
        config_file = self.temp_path / "test_config.yaml"

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(self.test_config, f)

        downloader = DatasetDownloader(str(config_file))

        self.assertEqual(downloader.config["data_sources"]["primary"], "local_fallback")

    def test_local_fallback_creation(self):
        """Test creation of local fallback dataset."""
        config_file = self.temp_path / "test_config.yaml"

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(self.test_config, f)

        downloader = DatasetDownloader(str(config_file))

        # Download should create fallback dataset
        result_path = downloader.download_dataset("local_fallback")

        self.assertTrue(result_path.exists())

        # Check content
        with open(result_path, "r") as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("title", data[0])
        self.assertIn("synopsis", data[0])

    @patch("src.acquisition.requests.get")
    def test_download_with_progress(self, mock_get):
        """Test downloading with progress tracking."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"test_data"] * 10
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        config_file = self.temp_path / "test_config.yaml"

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(self.test_config, f)

        downloader = DatasetDownloader(str(config_file))

        target_path = self.temp_path / "test_download.txt"
        result = downloader._download_with_progress(
            "http://example.com/file.txt", target_path
        )

        self.assertEqual(result, target_path)
        self.assertTrue(target_path.exists())

    def test_checksum_verification(self):
        """Test file integrity verification."""
        config_file = self.temp_path / "test_config.yaml"

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(self.test_config, f)

        downloader = DatasetDownloader(str(config_file))

        # Create test file
        test_file = self.temp_path / "test.txt"
        test_content = b"test content"
        with open(test_file, "wb") as f:
            f.write(test_content)

        # Calculate expected checksum
        import hashlib

        expected_checksum = hashlib.md5(test_content).hexdigest()

        # Test valid checksum
        self.assertTrue(downloader._verify_file_integrity(test_file, expected_checksum))

        # Test invalid checksum
        self.assertFalse(
            downloader._verify_file_integrity(test_file, "invalid_checksum")
        )

    def test_metadata_update(self):
        """Test metadata tracking."""
        config_file = self.temp_path / "test_config.yaml"

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(self.test_config, f)

        downloader = DatasetDownloader(str(config_file))

        # Create test file
        test_file = self.temp_path / "test.txt"
        with open(test_file, "w") as f:
            f.write("test content")

        source_config = {
            "url": "http://example.com/test.txt",
            "attribution": "Test attribution",
            "license": "Test license",
        }

        downloader._update_metadata(source_config, test_file, "test")

        # Check metadata was saved
        metadata = downloader._load_metadata()
        self.assertGreater(len(metadata), 0)

        # Check metadata content
        latest_entry = list(metadata.values())[-1]
        self.assertEqual(latest_entry["action"], "test")
        self.assertEqual(latest_entry["attribution"], "Test attribution")

    def test_dataset_info(self):
        """Test dataset information retrieval."""
        config_file = self.temp_path / "test_config.yaml"

        import yaml

        with open(config_file, "w") as f:
            yaml.dump(self.test_config, f)

        downloader = DatasetDownloader(str(config_file))

        info = downloader.get_dataset_info()

        self.assertIn("available_sources", info)
        self.assertIn("primary_source", info)
        self.assertIn("cached_datasets", info)
        self.assertEqual(info["primary_source"], "local_fallback")


class TestLicenseCompliance(unittest.TestCase):
    """Test license compliance functions."""

    def test_validate_license_compliance(self):
        """Test license validation."""
        source_config = {
            "attribution": "Test attribution required",
            "license": "https://example.com/license",
        }

        requirements = validate_license_compliance(source_config)

        self.assertEqual(len(requirements), 2)
        self.assertIn("attribution", requirements[0].lower())
        self.assertIn("license", requirements[1].lower())

    def test_empty_license_config(self):
        """Test empty license configuration."""
        source_config = {}

        requirements = validate_license_compliance(source_config)

        self.assertEqual(len(requirements), 0)


if __name__ == "__main__":
    unittest.main()
