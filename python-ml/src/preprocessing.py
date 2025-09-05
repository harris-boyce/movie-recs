"""
Feature engineering pipeline for movie datasets.

This module provides scalable preprocessing that handles missing values,
feature engineering, and multiple export formats for ML training.
"""

import json
import logging
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available, some features will be limited")

from .schema import Genre, Movie

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main feature engineering pipeline for movie datasets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer with configuration."""
        self.config = config or {}
        self.processing_config = self.config.get("processing", {})

        # Feature engineering settings
        self.text_config = self.config.get(
            "text_processing",
            {
                "max_features": 10000,
                "min_df": 2,
                "max_df": 0.95,
                "ngram_range": (1, 2),
                "remove_stopwords": True,
            },
        )

        self.feature_config = self.config.get(
            "feature_engineering",
            {
                "use_genre_encoding": True,
                "use_decade_features": True,
                "use_runtime_binning": True,
                "use_rating_features": True,
                "use_cast_features": True,
                "normalize_numerical": True,
            },
        )

        # Initialize processors
        self.text_vectorizer = None
        self.genre_encoder = None
        self.scalers = {}

        # Feature name tracking
        self.feature_names = []
        self.feature_metadata = {}

    def preprocess_text(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Preprocess and vectorize text fields (synopsis).

        Args:
            texts: List of text strings to process
            fit: Whether to fit the vectorizer (True for training data)

        Returns:
            Text feature matrix
        """
        logger.info(f"Processing {len(texts)} text documents")

        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]

        # Initialize or use existing vectorizer
        if fit or self.text_vectorizer is None:
            self.text_vectorizer = TfidfVectorizer(
                max_features=self.text_config["max_features"],
                min_df=self.text_config["min_df"],
                max_df=self.text_config["max_df"],
                ngram_range=self.text_config["ngram_range"],
                stop_words="english" if self.text_config["remove_stopwords"] else None,
                lowercase=True,
                strip_accents="unicode",
            )

            text_features = self.text_vectorizer.fit_transform(cleaned_texts)

            # Store feature names
            self.feature_names.extend([f"text_{name}" for name in self.text_vectorizer.get_feature_names_out()])

        else:
            text_features = self.text_vectorizer.transform(cleaned_texts)

        logger.info(f"Generated {text_features.shape[1]} text features")
        return text_features.toarray()

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " ",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove special characters but keep spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove very short words (< 2 characters)
        text = " ".join([word for word in text.split() if len(word) >= 2])

        return text

    def encode_genres(self, movies: List[Movie], fit: bool = True) -> np.ndarray:
        """
        Create genre encoding features.

        Args:
            movies: List of Movie objects
            fit: Whether to fit the encoder

        Returns:
            Genre feature matrix (one-hot encoded)
        """
        logger.info(f"Encoding genres for {len(movies)} movies")

        # Get all possible genres
        all_genres = [genre.value for genre in Genre]

        if fit:
            self.genre_encoder = all_genres

        # Create one-hot encoding
        genre_matrix = []

        for movie in movies:
            movie_genres = [genre.value for genre in movie.genres]
            genre_vector = [1 if genre in movie_genres else 0 for genre in self.genre_encoder]
            genre_matrix.append(genre_vector)

        # Store feature names
        if fit:
            self.feature_names.extend([f"genre_{genre.lower().replace(' ', '_')}" for genre in self.genre_encoder])

        return np.array(genre_matrix, dtype=np.float32)

    def create_decade_features(self, movies: List[Movie]) -> np.ndarray:
        """Create decade-based categorical features."""
        logger.info(f"Creating decade features for {len(movies)} movies")

        # Extract decades
        decades = []
        decade_categories = []

        for movie in movies:
            decade = (movie.release_year // 10) * 10
            decades.append(decade)

            # Categorize decades
            if decade < 1950:
                decade_categories.append("pre_1950")
            elif decade < 1980:
                decade_categories.append("mid_century")
            elif decade < 2000:
                decade_categories.append("modern")
            else:
                decade_categories.append("contemporary")

        # One-hot encode decade categories
        unique_categories = ["pre_1950", "mid_century", "modern", "contemporary"]
        decade_matrix = []

        for category in decade_categories:
            category_vector = [1 if cat == category else 0 for cat in unique_categories]
            decade_matrix.append(category_vector)

        # Store feature names
        self.feature_names.extend([f"decade_{cat}" for cat in unique_categories])

        # Add raw decade as normalized feature
        decade_array = np.array(decades, dtype=np.float32).reshape(-1, 1)
        if "decade_scaler" not in self.scalers:
            self.scalers["decade_scaler"] = StandardScaler()
            normalized_decades = self.scalers["decade_scaler"].fit_transform(decade_array)
        else:
            normalized_decades = self.scalers["decade_scaler"].transform(decade_array)

        self.feature_names.append("decade_normalized")

        # Combine categorical and normalized features
        result = np.hstack([np.array(decade_matrix, dtype=np.float32), normalized_decades])

        logger.info(f"Generated {result.shape[1]} decade features")
        return result

    def create_runtime_features(self, movies: List[Movie]) -> np.ndarray:
        """Create runtime-based features with binning and normalization."""
        logger.info(f"Creating runtime features for {len(movies)} movies")

        # Extract runtimes, handle missing values
        runtimes = []
        has_runtime = []

        for movie in movies:
            if movie.runtime_mins:
                runtimes.append(movie.runtime_mins)
                has_runtime.append(1)
            else:
                # Use median runtime for missing values
                runtimes.append(0)  # Will be filled later
                has_runtime.append(0)

        runtimes = np.array(runtimes, dtype=np.float32)

        # Fill missing values with median
        valid_runtimes = runtimes[runtimes > 0]
        if len(valid_runtimes) > 0:
            median_runtime = np.median(valid_runtimes)
            runtimes[runtimes == 0] = median_runtime
        else:
            runtimes[runtimes == 0] = 120  # Default 2 hours

        # Create runtime bins
        runtime_bins = self._create_runtime_bins(runtimes)

        # Normalize runtime
        if "runtime_scaler" not in self.scalers:
            self.scalers["runtime_scaler"] = StandardScaler()
            normalized_runtime = self.scalers["runtime_scaler"].fit_transform(runtimes.reshape(-1, 1))
        else:
            normalized_runtime = self.scalers["runtime_scaler"].transform(runtimes.reshape(-1, 1))

        # Combine features
        features = np.hstack(
            [
                runtime_bins,
                normalized_runtime,
                np.array(has_runtime, dtype=np.float32).reshape(-1, 1),
            ]
        )

        # Store feature names
        bin_names = ["runtime_short", "runtime_medium", "runtime_long", "runtime_epic"]
        self.feature_names.extend(bin_names + ["runtime_normalized", "has_runtime"])

        logger.info(f"Generated {features.shape[1]} runtime features")
        return features

    def _create_runtime_bins(self, runtimes: np.ndarray) -> np.ndarray:
        """Create runtime bins: short, medium, long, epic."""
        bins = np.zeros((len(runtimes), 4), dtype=np.float32)

        for i, runtime in enumerate(runtimes):
            if runtime < 90:
                bins[i, 0] = 1  # short
            elif runtime < 120:
                bins[i, 1] = 1  # medium
            elif runtime < 180:
                bins[i, 2] = 1  # long
            else:
                bins[i, 3] = 1  # epic

        return bins

    def create_rating_features(self, movies: List[Movie]) -> np.ndarray:
        """Create rating-based features."""
        logger.info(f"Creating rating features for {len(movies)} movies")

        features = []

        for movie in movies:
            rating_features = [
                movie.ratings.average,
                np.log1p(movie.ratings.count),  # Log-scaled count
                movie.ratings.average * np.log1p(movie.ratings.count),  # Interaction
            ]

            # Rating category
            if movie.ratings.average >= 8.0:
                rating_category = [1, 0, 0]  # High
            elif movie.ratings.average >= 6.0:
                rating_category = [0, 1, 0]  # Medium
            else:
                rating_category = [0, 0, 1]  # Low

            rating_features.extend(rating_category)
            features.append(rating_features)

        features = np.array(features, dtype=np.float32)

        # Normalize continuous features
        if "rating_scaler" not in self.scalers:
            self.scalers["rating_scaler"] = StandardScaler()
            features[:, :3] = self.scalers["rating_scaler"].fit_transform(features[:, :3])
        else:
            features[:, :3] = self.scalers["rating_scaler"].transform(features[:, :3])

        # Store feature names
        self.feature_names.extend(
            [
                "rating_average",
                "rating_count_log",
                "rating_interaction",
                "rating_high",
                "rating_medium",
                "rating_low",
            ]
        )

        logger.info(f"Generated {features.shape[1]} rating features")
        return features

    def create_cast_crew_features(self, movies: List[Movie]) -> np.ndarray:
        """Create aggregated cast and crew features."""
        logger.info(f"Creating cast/crew features for {len(movies)} movies")

        features = []

        for movie in movies:
            cast_features = self._aggregate_person_features(movie.cast or [])
            crew_features = self._aggregate_person_features(movie.crew or [])

            # Combine features
            movie_features = [
                len(movie.cast) if movie.cast else 0,  # Cast size
                len(movie.crew) if movie.crew else 0,  # Crew size
            ]

            movie_features.extend(cast_features)
            movie_features.extend(crew_features)

            features.append(movie_features)

        features = np.array(features, dtype=np.float32)

        # Normalize
        if "cast_crew_scaler" not in self.scalers:
            self.scalers["cast_crew_scaler"] = StandardScaler()
            features = self.scalers["cast_crew_scaler"].fit_transform(features)
        else:
            features = self.scalers["cast_crew_scaler"].transform(features)

        # Store feature names
        self.feature_names.extend(
            [
                "cast_size",
                "crew_size",
                "cast_avg_age",
                "cast_gender_balance",
                "cast_known_ratio",
                "crew_avg_age",
                "crew_gender_balance",
                "crew_known_ratio",
            ]
        )

        logger.info(f"Generated {features.shape[1]} cast/crew features")
        return features

    def _aggregate_person_features(self, people: List) -> List[float]:
        """Aggregate features from cast or crew list."""
        if not people:
            return [0.0, 0.5, 0.0]  # Default values

        # Age features (approximate from birth year)
        ages = []
        genders = []
        known_count = 0

        for person in people:
            if hasattr(person, "birth_year") and person.birth_year:
                # Approximate current age
                ages.append(2024 - person.birth_year)

            if hasattr(person, "gender") and person.gender:
                genders.append(person.gender)

            # Check if person has additional info (considered "known")
            if (hasattr(person, "birth_year") and person.birth_year) or (hasattr(person, "gender") and person.gender):
                known_count += 1

        # Calculate aggregated features
        avg_age = np.mean(ages) if ages else 35.0  # Default age

        # Gender balance (0 = all male, 1 = all female, 0.5 = balanced)
        if genders:
            female_ratio = sum(1 for g in genders if g == "female") / len(genders)
        else:
            female_ratio = 0.5  # Unknown, assume balanced

        known_ratio = known_count / len(people)

        return [avg_age, female_ratio, known_ratio]

    def process_movies(
        self, movies: List[Movie], fit: bool = True, include_text: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Process complete movie dataset into feature matrix.

        Args:
            movies: List of Movie objects to process
            fit: Whether to fit encoders/scalers (True for training data)
            include_text: Whether to include text features

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        logger.info(f"Processing {len(movies)} movies into features (fit={fit})")

        if not movies:
            return np.array([]), []

        # Reset feature names if fitting
        if fit:
            self.feature_names = []

        feature_matrices = []

        # Text features
        if include_text and self.feature_config.get("use_text_features", True):
            texts = [movie.synopsis for movie in movies]
            text_features = self.preprocess_text(texts, fit=fit)
            feature_matrices.append(text_features)

        # Genre features
        if self.feature_config.get("use_genre_encoding", True):
            genre_features = self.encode_genres(movies, fit=fit)
            feature_matrices.append(genre_features)

        # Decade features
        if self.feature_config.get("use_decade_features", True):
            decade_features = self.create_decade_features(movies)
            feature_matrices.append(decade_features)

        # Runtime features
        if self.feature_config.get("use_runtime_binning", True):
            runtime_features = self.create_runtime_features(movies)
            feature_matrices.append(runtime_features)

        # Rating features
        if self.feature_config.get("use_rating_features", True):
            rating_features = self.create_rating_features(movies)
            feature_matrices.append(rating_features)

        # Cast/crew features
        if self.feature_config.get("use_cast_features", True):
            cast_features = self.create_cast_crew_features(movies)
            feature_matrices.append(cast_features)

        # Combine all features
        if feature_matrices:
            combined_features = np.hstack(feature_matrices)
        else:
            combined_features = np.empty((len(movies), 0))

        logger.info(f"Generated feature matrix: {combined_features.shape}")

        return combined_features, self.feature_names.copy()

    def create_train_test_split(
        self,
        movies: List[Movie],
        test_size: float = 0.2,
        stratify_by_genre: bool = True,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Create stratified train/test split.

        Args:
            movies: List of movies to split
            test_size: Proportion for test set
            stratify_by_genre: Whether to stratify by primary genre
            random_state: Random seed

        Returns:
            Dictionary with train/test splits and metadata
        """
        logger.info(f"Creating train/test split: test_size={test_size}")

        if stratify_by_genre and len(movies) > 10:
            # Use primary genre for stratification
            primary_genres = [movie.genres[0].value if movie.genres else "Unknown" for movie in movies]

            # Ensure we have enough samples per genre
            genre_counts = Counter(primary_genres)
            stratify_labels = []

            for genre in primary_genres:
                if genre_counts[genre] >= 2:  # Need at least 2 for split
                    stratify_labels.append(genre)
                else:
                    stratify_labels.append("Other")  # Group rare genres

            train_idx, test_idx = train_test_split(
                range(len(movies)),
                test_size=test_size,
                stratify=stratify_labels,
                random_state=random_state,
            )
        else:
            # Simple random split
            train_idx, test_idx = train_test_split(range(len(movies)), test_size=test_size, random_state=random_state)

        # Create splits
        train_movies = [movies[i] for i in train_idx]
        test_movies = [movies[i] for i in test_idx]

        # Process features
        train_features, feature_names = self.process_movies(train_movies, fit=True)
        test_features, _ = self.process_movies(test_movies, fit=False)

        split_data = {
            "train": {
                "movies": train_movies,
                "features": train_features,
                "indices": train_idx,
            },
            "test": {
                "movies": test_movies,
                "features": test_features,
                "indices": test_idx,
            },
            "feature_names": feature_names,
            "split_info": {
                "test_size": test_size,
                "train_size": len(train_movies),
                "test_size_actual": len(test_movies),
                "stratified": stratify_by_genre,
                "random_state": random_state,
            },
        }

        logger.info(f"Split complete: {len(train_movies)} train, {len(test_movies)} test")

        return split_data

    def export_features(
        self,
        features: np.ndarray,
        movies: List[Movie],
        feature_names: List[str],
        output_dir: str,
        format_types: List[str] = None,
    ) -> Dict[str, str]:
        """
        Export features in multiple formats.

        Args:
            features: Feature matrix
            movies: Original movie objects
            feature_names: List of feature names
            output_dir: Output directory path
            format_types: List of formats to export ('csv', 'json', 'parquet', 'numpy')

        Returns:
            Dictionary mapping format to output file path
        """
        if format_types is None:
            format_types = ["csv", "json", "parquet", "numpy"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        logger.info(f"Exporting features in formats: {format_types}")

        # CSV export
        if "csv" in format_types and PANDAS_AVAILABLE:
            csv_path = output_dir / "features.csv"

            # Create DataFrame
            df = pd.DataFrame(features, columns=feature_names)

            # Add movie metadata
            df.insert(0, "movie_id", [movie.movie_id for movie in movies])
            df.insert(1, "title", [movie.title for movie in movies])
            df.insert(2, "release_year", [movie.release_year for movie in movies])

            df.to_csv(csv_path, index=False)
            exported_files["csv"] = str(csv_path)
            logger.info(f"Exported CSV: {csv_path}")

        # JSON export
        if "json" in format_types:
            json_path = output_dir / "features.json"

            export_data = {
                "features": features.tolist(),
                "feature_names": feature_names,
                "movie_metadata": [
                    {
                        "movie_id": movie.movie_id,
                        "title": movie.title,
                        "release_year": movie.release_year,
                        "genres": [genre.value for genre in movie.genres],
                    }
                    for movie in movies
                ],
                "shape": features.shape,
                "export_timestamp": (pd.Timestamp.now().isoformat() if PANDAS_AVAILABLE else "unknown"),
            }

            with open(json_path, "w") as f:
                json.dump(export_data, f, indent=2)

            exported_files["json"] = str(json_path)
            logger.info(f"Exported JSON: {json_path}")

        # Parquet export
        if "parquet" in format_types and PANDAS_AVAILABLE:
            try:
                parquet_path = output_dir / "features.parquet"

                df = pd.DataFrame(features, columns=feature_names)
                df.insert(0, "movie_id", [movie.movie_id for movie in movies])
                df.insert(1, "title", [movie.title for movie in movies])
                df.insert(2, "release_year", [movie.release_year for movie in movies])

                df.to_parquet(parquet_path, index=False)
                exported_files["parquet"] = str(parquet_path)
                logger.info(f"Exported Parquet: {parquet_path}")

            except ImportError:
                logger.warning("Parquet export requires pyarrow, skipping")

        # NumPy export
        if "numpy" in format_types:
            numpy_path = output_dir / "features.npz"

            movie_ids = [movie.movie_id for movie in movies]

            np.savez_compressed(
                numpy_path,
                features=features,
                feature_names=np.array(feature_names),
                movie_ids=np.array(movie_ids),
                shape=features.shape,
            )

            exported_files["numpy"] = str(numpy_path)
            logger.info(f"Exported NumPy: {numpy_path}")

        return exported_files


if __name__ == "__main__":
    # Example usage
    from .schema import Ratings

    # Sample movies for testing
    sample_movies = [
        Movie(
            movie_id="1",
            title="The Shawshank Redemption",
            synopsis="Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
            release_year=1994,
            runtime_mins=142,
            genres=[Genre.DRAMA],
            ratings=Ratings(average=9.3, count=2500000),
            data_source="example",
        ),
        Movie(
            movie_id="2",
            title="The Godfather",
            synopsis="The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
            release_year=1972,
            runtime_mins=175,
            genres=[Genre.CRIME, Genre.DRAMA],
            ratings=Ratings(average=9.2, count=1800000),
            data_source="example",
        ),
    ]

    # Initialize feature engineer
    config = {
        "feature_engineering": {
            "use_genre_encoding": True,
            "use_decade_features": True,
            "use_runtime_binning": True,
            "use_rating_features": True,
            "use_cast_features": False,  # No cast data in sample
        }
    }

    engineer = FeatureEngineer(config)

    # Process movies
    features, feature_names = engineer.process_movies(sample_movies, include_text=True)

    print(f"Generated features: {features.shape}")
    print(f"Feature names: {len(feature_names)}")
    print(f"First few features: {feature_names[:10]}")

    # Create train/test split
    split_data = engineer.create_train_test_split(sample_movies, test_size=0.5)
    print(f"Train set: {split_data['train']['features'].shape}")
    print(f"Test set: {split_data['test']['features'].shape}")

    # Export features
    if features.size > 0:
        exported = engineer.export_features(features, sample_movies, feature_names, "data/processed", ["json", "numpy"])
        print(f"Exported files: {exported}")
