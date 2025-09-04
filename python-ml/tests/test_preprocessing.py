"""
Tests for the feature engineering and preprocessing module.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.feature_utils import (
    CategoryEncoder,
    FeatureNameMapper,
    FeatureSelector,
    ScalingUtilities,
)
from src.preprocessing import FeatureEngineer
from src.schema import Genre, Language, Metadata, Movie, PersonInfo, Ratings


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "text_processing": {
                "max_features": 100,
                "min_df": 1,
                "max_df": 0.95,
                "ngram_range": (1, 2),
                "remove_stopwords": True,
            },
            "feature_engineering": {
                "use_genre_encoding": True,
                "use_decade_features": True,
                "use_runtime_binning": True,
                "use_rating_features": True,
                "use_cast_features": True,
            },
        }

        self.engineer = FeatureEngineer(self.config)

        # Sample movies with diverse data
        self.sample_movies = [
            Movie(
                movie_id="1",
                title="The Shawshank Redemption",
                synopsis="Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
                release_year=1994,
                runtime_mins=142,
                genres=[Genre.DRAMA],
                ratings=Ratings(average=9.3, count=2500000),
                cast=[
                    PersonInfo(name="Tim Robbins", role="actor", birth_year=1958, gender="male"),
                    PersonInfo(
                        name="Morgan Freeman",
                        role="actor",
                        birth_year=1937,
                        gender="male",
                    ),
                ],
                crew=[
                    PersonInfo(
                        name="Frank Darabont",
                        role="director",
                        birth_year=1959,
                        gender="male",
                    )
                ],
                metadata=Metadata(language=Language.EN, country="US"),
                data_source="test",
            ),
            Movie(
                movie_id="2",
                title="Pulp Fiction",
                synopsis="The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
                release_year=1994,
                runtime_mins=154,
                genres=[Genre.CRIME, Genre.DRAMA],
                ratings=Ratings(average=8.9, count=2000000),
                cast=[
                    PersonInfo(
                        name="John Travolta",
                        role="actor",
                        birth_year=1954,
                        gender="male",
                    ),
                    PersonInfo(
                        name="Uma Thurman",
                        role="actor",
                        birth_year=1970,
                        gender="female",
                    ),
                ],
                crew=[
                    PersonInfo(
                        name="Quentin Tarantino",
                        role="director",
                        birth_year=1963,
                        gender="male",
                    )
                ],
                metadata=Metadata(language=Language.EN, country="US"),
                data_source="test",
            ),
            Movie(
                movie_id="3",
                title="Spirited Away",
                synopsis="During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits.",
                release_year=2001,
                runtime_mins=125,
                genres=[Genre.ANIMATION, Genre.FAMILY, Genre.FANTASY],
                ratings=Ratings(average=9.3, count=750000),
                # No cast/crew data for this one
                metadata=Metadata(language=Language.JA, country="JP"),
                data_source="test",
            ),
        ]

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initialization."""
        self.assertIsNotNone(self.engineer.config)
        self.assertIsNotNone(self.engineer.text_config)
        self.assertIsNotNone(self.engineer.feature_config)
        self.assertEqual(len(self.engineer.feature_names), 0)

    def test_text_preprocessing(self):
        """Test text preprocessing and vectorization."""
        texts = [movie.synopsis for movie in self.sample_movies]

        # Test fitting
        text_features = self.engineer.preprocess_text(texts, fit=True)

        self.assertEqual(text_features.shape[0], len(texts))
        self.assertGreater(text_features.shape[1], 0)
        self.assertIsNotNone(self.engineer.text_vectorizer)

        # Test transform (without fitting)
        text_features_2 = self.engineer.preprocess_text(texts, fit=False)
        self.assertEqual(text_features.shape, text_features_2.shape)

    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        dirty_text = "<p>This is a TEST with <b>HTML</b> tags and http://example.com URLs & special chars!!!</p>"
        cleaned = self.engineer._clean_text(dirty_text)

        self.assertNotIn("<", cleaned)
        self.assertNotIn("http:", cleaned)
        self.assertIn("test", cleaned)  # Should be lowercase

    def test_genre_encoding(self):
        """Test genre one-hot encoding."""
        genre_features = self.engineer.encode_genres(self.sample_movies, fit=True)

        self.assertEqual(genre_features.shape[0], len(self.sample_movies))
        self.assertEqual(genre_features.shape[1], len(Genre))

        # Check that Drama is encoded for movies 1 and 2
        drama_idx = list(Genre).index(Genre.DRAMA)
        self.assertEqual(genre_features[0, drama_idx], 1)  # Movie 1 has Drama
        self.assertEqual(genre_features[1, drama_idx], 1)  # Movie 2 has Drama
        self.assertEqual(genre_features[2, drama_idx], 0)  # Movie 3 doesn't have Drama

    def test_decade_features(self):
        """Test decade feature creation."""
        decade_features = self.engineer.create_decade_features(self.sample_movies)

        self.assertEqual(decade_features.shape[0], len(self.sample_movies))
        self.assertGreater(decade_features.shape[1], 4)  # At least 4 categories + normalized

        # Movies 1 and 2 are from 1994, Movie 3 is from 2001
        # They should have different decade category encodings
        self.assertFalse(np.array_equal(decade_features[0], decade_features[2]))

    def test_runtime_features(self):
        """Test runtime feature engineering."""
        runtime_features = self.engineer.create_runtime_features(self.sample_movies)

        self.assertEqual(runtime_features.shape[0], len(self.sample_movies))
        self.assertEqual(runtime_features.shape[1], 6)  # 4 bins + normalized + has_runtime

        # All test movies have runtime, so has_runtime should be 1
        self.assertTrue(np.all(runtime_features[:, -1] == 1))

    def test_runtime_features_missing_values(self):
        """Test runtime features with missing values."""
        # Create movie without runtime
        movie_no_runtime = Movie(
            movie_id="test",
            title="Test Movie",
            synopsis="Test synopsis for a movie without runtime information available.",
            release_year=2000,
            genres=[Genre.DRAMA],
            ratings=Ratings(average=7.0, count=1000),
            runtime_mins=None,  # Missing runtime
            data_source="test",
        )

        movies_with_missing = self.sample_movies + [movie_no_runtime]
        runtime_features = self.engineer.create_runtime_features(movies_with_missing)

        # Last movie should have has_runtime = 0
        self.assertEqual(runtime_features[-1, -1], 0)

    def test_rating_features(self):
        """Test rating feature creation."""
        rating_features = self.engineer.create_rating_features(self.sample_movies)

        self.assertEqual(rating_features.shape[0], len(self.sample_movies))
        self.assertEqual(rating_features.shape[1], 6)  # average, count_log, interaction + 3 categories

        # Check that high-rated movies (>8.0) have rating_high = 1
        for i, movie in enumerate(self.sample_movies):
            if movie.ratings.average >= 8.0:
                self.assertEqual(rating_features[i, 3], 1)  # rating_high

    def test_cast_crew_features(self):
        """Test cast and crew feature engineering."""
        cast_features = self.engineer.create_cast_crew_features(self.sample_movies)

        self.assertEqual(cast_features.shape[0], len(self.sample_movies))
        self.assertEqual(cast_features.shape[1], 8)  # cast_size, crew_size + 6 aggregated features

        # Movies 1 and 2 have cast/crew, Movie 3 doesn't
        # Cast sizes should be different
        self.assertNotEqual(cast_features[0, 0], cast_features[2, 0])

    def test_process_movies_full_pipeline(self):
        """Test complete movie processing pipeline."""
        features, feature_names = self.engineer.process_movies(self.sample_movies, fit=True, include_text=True)

        self.assertEqual(features.shape[0], len(self.sample_movies))
        self.assertGreater(features.shape[1], 0)
        self.assertGreater(len(feature_names), 0)
        self.assertEqual(features.shape[1], len(feature_names))

        # Test without text features
        features_no_text, names_no_text = self.engineer.process_movies(
            self.sample_movies,
            fit=False,  # Use already fitted encoders
            include_text=False,
        )

        self.assertEqual(features_no_text.shape[0], len(self.sample_movies))
        self.assertLess(features_no_text.shape[1], features.shape[1])  # Should be smaller

    def test_train_test_split(self):
        """Test train/test split creation."""
        split_data = self.engineer.create_train_test_split(
            self.sample_movies,
            test_size=0.33,
            stratify_by_genre=False,  # Too few samples for stratification
            random_state=42,
        )

        self.assertIn("train", split_data)
        self.assertIn("test", split_data)
        self.assertIn("feature_names", split_data)
        self.assertIn("split_info", split_data)

        train_size = len(split_data["train"]["movies"])
        test_size = len(split_data["test"]["movies"])

        self.assertEqual(train_size + test_size, len(self.sample_movies))
        self.assertGreater(train_size, 0)
        self.assertGreater(test_size, 0)

        # Check feature matrix shapes
        self.assertEqual(split_data["train"]["features"].shape[0], train_size)
        self.assertEqual(split_data["test"]["features"].shape[0], test_size)

    def test_export_features(self):
        """Test feature export in multiple formats."""
        features, feature_names = self.engineer.process_movies(self.sample_movies)

        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = self.engineer.export_features(
                features,
                self.sample_movies,
                feature_names,
                temp_dir,
                format_types=["json", "numpy"],  # Skip CSV/Parquet for test
            )

            self.assertIn("json", exported_files)
            self.assertIn("numpy", exported_files)

            # Check files exist
            for file_path in exported_files.values():
                self.assertTrue(Path(file_path).exists())

            # Check JSON content
            import json

            with open(exported_files["json"], "r") as f:
                json_data = json.load(f)

            self.assertIn("features", json_data)
            self.assertIn("feature_names", json_data)
            self.assertIn("movie_metadata", json_data)
            self.assertEqual(len(json_data["movie_metadata"]), len(self.sample_movies))

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        features, feature_names = self.engineer.process_movies([])

        self.assertEqual(features.shape[0], 0)
        self.assertEqual(len(feature_names), 0)

    def test_configuration_options(self):
        """Test different configuration options."""
        # Test with minimal features
        minimal_config = {
            "feature_engineering": {
                "use_genre_encoding": True,
                "use_decade_features": False,
                "use_runtime_binning": False,
                "use_rating_features": False,
                "use_cast_features": False,
            }
        }

        minimal_engineer = FeatureEngineer(minimal_config)
        features, feature_names = minimal_engineer.process_movies(self.sample_movies, include_text=False)

        # Should only have genre features
        self.assertGreater(features.shape[1], 0)
        self.assertLess(features.shape[1], 50)  # Much smaller than full feature set


class TestFeatureUtils(unittest.TestCase):
    """Test feature utility functions."""

    def test_feature_name_mapper(self):
        """Test FeatureNameMapper functionality."""
        mapper = FeatureNameMapper()

        # Test description lookup
        description = mapper.get_description("genre_action")
        self.assertIn("Action", description)

        # Test category grouping
        feature_names = ["genre_action", "text_movie", "runtime_long", "rating_high"]
        categories = mapper.get_feature_categories(feature_names)

        self.assertIn("genre", categories)
        self.assertIn("text", categories)
        self.assertIn("runtime", categories)
        self.assertIn("rating", categories)

    def test_scaling_utilities(self):
        """Test scaling utilities."""
        # Create sample feature matrix
        features = np.random.randn(100, 10)

        # Test standard scaling
        scaled_features, scaler = ScalingUtilities.scale_features(features, "standard")

        self.assertEqual(scaled_features.shape, features.shape)
        # Check that features are approximately standardized
        self.assertAlmostEqual(np.mean(scaled_features), 0, places=1)
        self.assertAlmostEqual(np.std(scaled_features), 1, places=1)

    def test_category_encoder(self):
        """Test categorical encoding utilities."""
        categories = ["action", "drama", "comedy", "action", "drama"]

        encoded, category_map = CategoryEncoder.encode_categorical(categories)

        self.assertEqual(encoded.shape[0], len(categories))
        self.assertEqual(encoded.shape[1], len(set(categories)))

        # Check one-hot property
        for row in encoded:
            self.assertEqual(np.sum(row), 1)  # Exactly one category per row

    def test_multi_label_encoding(self):
        """Test multi-label categorical encoding."""
        multi_categories = [
            ["action", "adventure"],
            ["drama"],
            ["comedy", "romance", "drama"],
        ]

        encoded, category_map = CategoryEncoder.encode_multi_label(multi_categories)

        self.assertEqual(encoded.shape[0], len(multi_categories))
        self.assertGreater(encoded.shape[1], 0)

        # Check multi-hot property
        self.assertGreater(np.sum(encoded[0]), 1)  # First row should have multiple 1s
        self.assertEqual(np.sum(encoded[1]), 1)  # Second row should have one 1
        self.assertGreater(np.sum(encoded[2]), 1)  # Third row should have multiple 1s

    @patch("src.feature_utils.PANDAS_AVAILABLE", True)
    def test_feature_selector(self):
        """Test feature selection utilities."""
        # Create sample data with some features correlated to target
        n_samples, n_features = 100, 20
        features = np.random.randn(n_samples, n_features)

        # Make first few features correlated with target
        targets = features[:, 0] + features[:, 1] + np.random.randn(n_samples) * 0.1
        targets = (targets > np.median(targets)).astype(int)  # Binary classification

        # Test k-best selection
        selected_features, selected_indices, selector = FeatureSelector.select_k_best(features, targets, k=5)

        self.assertEqual(selected_features.shape[1], 5)
        self.assertEqual(len(selected_indices), 5)

        # Test variance-based selection
        var_features, var_indices = FeatureSelector.select_by_variance(features, threshold=0.5)

        self.assertLessEqual(var_features.shape[1], features.shape[1])
        self.assertEqual(len(var_indices), var_features.shape[1])


class TestFeatureEngineerIntegration(unittest.TestCase):
    """Integration tests for complete feature engineering pipeline."""

    def test_full_pipeline_with_diverse_movies(self):
        """Test complete pipeline with diverse movie dataset."""
        # Create diverse movie dataset
        movies = [
            # Classic drama
            Movie(
                movie_id="classic_1",
                title="Casablanca",
                synopsis="A cynical expatriate American cafe owner struggles to decide whether or not to help his former lover and her fugitive husband escape the Nazis in French Morocco.",
                release_year=1942,
                runtime_mins=102,
                genres=[Genre.DRAMA, Genre.ROMANCE],
                ratings=Ratings(average=8.5, count=558000),
                data_source="test",
            ),
            # Modern action
            Movie(
                movie_id="modern_1",
                title="Mad Max: Fury Road",
                synopsis="In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners.",
                release_year=2015,
                runtime_mins=120,
                genres=[Genre.ACTION, Genre.ADVENTURE],
                ratings=Ratings(average=8.1, count=920000),
                data_source="test",
            ),
            # Comedy
            Movie(
                movie_id="comedy_1",
                title="The Grand Budapest Hotel",
                synopsis="A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy in the hotel's glorious years under an exceptional concierge.",
                release_year=2014,
                runtime_mins=99,
                genres=[Genre.COMEDY, Genre.ADVENTURE],
                ratings=Ratings(average=8.1, count=800000),
                data_source="test",
            ),
        ]

        config = {
            "feature_engineering": {
                "use_genre_encoding": True,
                "use_decade_features": True,
                "use_runtime_binning": True,
                "use_rating_features": True,
                "use_cast_features": False,  # No cast data
            },
            "text_processing": {
                "max_features": 50,
                "min_df": 1,
                "max_df": 0.95,
                "ngram_range": (1, 2),
                "remove_stopwords": True,
            },  # Smaller for test
        }

        engineer = FeatureEngineer(config)

        # Process features
        features, feature_names = engineer.process_movies(movies, include_text=True)

        self.assertEqual(features.shape[0], len(movies))
        self.assertGreater(features.shape[1], 20)  # Should have substantial features
        self.assertEqual(len(feature_names), features.shape[1])

        # Create train/test split
        split_data = engineer.create_train_test_split(movies, test_size=0.33, stratify_by_genre=False)

        self.assertGreater(len(split_data["train"]["movies"]), 0)
        self.assertGreater(len(split_data["test"]["movies"]), 0)

        # Export features
        with tempfile.TemporaryDirectory() as temp_dir:
            exported = engineer.export_features(features, movies, feature_names, temp_dir, ["json"])

            self.assertIn("json", exported)
            self.assertTrue(Path(exported["json"]).exists())


if __name__ == "__main__":
    unittest.main()
