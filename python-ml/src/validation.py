"""
Data validation and bias detection system for movie datasets.

This module provides comprehensive data quality validation and bias detection
capabilities including:
- Schema validation with detailed error reporting
- Completeness and consistency checks
- Bias analysis across multiple dimensions (genre, demographic, geographic, temporal)
- HTML report generation with visualizations
- Configurable thresholds and metrics
"""

import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available, visualizations will be disabled")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available, some analysis features will be limited")

try:
    from jinja2 import Template

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("Jinja2 not available, HTML report generation will be disabled")

from pydantic import BaseModel, Field

from .schema import (
    Genre,
    Language,
    Movie,
    QualityThresholds,
    ValidationResult,
    create_movie_from_dict,
)


class BiasMetrics(BaseModel):
    """Container for bias detection metrics."""

    genre_diversity: Dict[str, float] = Field(default_factory=dict)
    demographic_representation: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    geographic_distribution: Dict[str, float] = Field(default_factory=dict)
    temporal_distribution: Dict[str, float] = Field(default_factory=dict)
    rating_bias_analysis: Dict[str, Any] = Field(default_factory=dict)
    overall_bias_score: float = 0.0
    recommendations: List[str] = Field(default_factory=list)


class DataValidator:
    """Main data validation and bias detection class."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validator with configuration."""
        self.config = config or {}
        self.thresholds = QualityThresholds(**self.config.get("quality_thresholds", {}))
        self.bias_config = self.config.get("bias_detection", {})

        # Setup matplotlib for headless operation
        plt.switch_backend("Agg")

    def validate_dataset(self, movies: List[Dict[str, Any]]) -> Tuple[ValidationResult, List[Movie]]:
        """
        Validate entire dataset and return results with valid movies.

        Args:
            movies: List of movie dictionaries to validate

        Returns:
            Tuple of (validation_result, list_of_valid_movies)
        """
        logger.info(f"Starting validation of {len(movies)} movies")

        result = ValidationResult(is_valid=True)
        valid_movies = []

        # Track validation statistics
        result.summary = {
            "total_movies": len(movies),
            "valid_movies": 0,
            "invalid_movies": 0,
            "field_completeness": {},
            "validation_start": datetime.now().isoformat(),
        }

        # Validate each movie
        for i, movie_data in enumerate(movies):
            movie_id = movie_data.get("movie_id", f"movie_{i}")

            try:
                # Validate against schema
                movie = create_movie_from_dict(movie_data)

                # Additional quality checks
                self._validate_movie_quality(movie, result)

                valid_movies.append(movie)
                result.summary["valid_movies"] += 1

            except ValueError as e:
                result.add_error(movie_id, "schema", str(e))
                result.summary["invalid_movies"] += 1
                logger.warning(f"Movie {movie_id} failed validation: {e}")

        # Calculate field completeness
        if valid_movies:
            result.summary["field_completeness"] = self._calculate_field_completeness(valid_movies)

        # Overall validation status
        completeness_rate = result.summary["valid_movies"] / result.summary["total_movies"] if result.summary["total_movies"] > 0 else 0
        if completeness_rate < self.thresholds.completeness_min:
            result.add_error(
                "dataset",
                "completeness",
                f"Dataset completeness {completeness_rate:.2%} below threshold {self.thresholds.completeness_min:.2%}",
            )

        result.summary["completeness_rate"] = completeness_rate
        result.summary["validation_end"] = datetime.now().isoformat()

        logger.info(
            f"Validation complete: {result.summary['valid_movies']}/{result.summary['total_movies']} movies valid"
        )

        return result, valid_movies

    def _validate_movie_quality(self, movie: Movie, result: ValidationResult) -> None:
        """Perform additional quality checks on individual movie."""

        # Check synopsis length
        if len(movie.synopsis) < self.thresholds.synopsis_min_length:
            result.add_warning(
                movie.movie_id,
                "synopsis",
                f"Synopsis too short: {len(movie.synopsis)} < {self.thresholds.synopsis_min_length}",
                len(movie.synopsis),
            )

        # Check title length
        if len(movie.title) < self.thresholds.title_min_length:
            result.add_error(
                movie.movie_id,
                "title",
                f"Title too short: {len(movie.title)} < {self.thresholds.title_min_length}",
                movie.title,
            )

        # Check year range
        if not (self.thresholds.valid_year_range[0] <= movie.release_year <= self.thresholds.valid_year_range[1]):
            result.add_warning(
                movie.movie_id,
                "release_year",
                f"Release year {movie.release_year} outside expected range {self.thresholds.valid_year_range}",
                movie.release_year,
            )

        # Check rating count
        if movie.ratings.count < self.thresholds.min_rating_count:
            result.add_warning(
                movie.movie_id,
                "ratings",
                f"Low rating count: {movie.ratings.count} < {self.thresholds.min_rating_count}",
                movie.ratings.count,
            )

        # Check cast/crew sizes
        if movie.cast and len(movie.cast) > self.thresholds.max_cast_size:
            result.add_warning(
                movie.movie_id,
                "cast",
                f"Large cast size: {len(movie.cast)} > {self.thresholds.max_cast_size}",
                len(movie.cast),
            )

        if movie.crew and len(movie.crew) > self.thresholds.max_crew_size:
            result.add_warning(
                movie.movie_id,
                "crew",
                f"Large crew size: {len(movie.crew)} > {self.thresholds.max_crew_size}",
                len(movie.crew),
            )

    def _calculate_field_completeness(self, movies: List[Movie]) -> Dict[str, float]:
        """Calculate completeness percentage for each field."""
        if not movies:
            return {}

        completeness = {}
        total_count = len(movies)

        # Required fields (always present due to schema validation)
        completeness["movie_id"] = 1.0
        completeness["title"] = 1.0
        completeness["synopsis"] = 1.0
        completeness["release_year"] = 1.0
        completeness["genres"] = 1.0
        completeness["ratings"] = 1.0

        # Optional fields
        optional_fields = [
            ("runtime_mins", lambda m: m.runtime_mins is not None),
            ("cast", lambda m: m.cast is not None and len(m.cast) > 0),
            ("crew", lambda m: m.crew is not None and len(m.crew) > 0),
            ("metadata", lambda m: m.metadata is not None),
            ("imdb_id", lambda m: m.metadata and m.metadata.imdb_id is not None),
            ("tmdb_id", lambda m: m.metadata and m.metadata.tmdb_id is not None),
            ("budget", lambda m: m.metadata and m.metadata.budget is not None),
            ("revenue", lambda m: m.metadata and m.metadata.revenue is not None),
        ]

        for field_name, checker in optional_fields:
            present_count = sum(1 for movie in movies if checker(movie))
            completeness[field_name] = present_count / total_count

        return completeness

    def detect_bias(self, movies: List[Movie]) -> BiasMetrics:
        """
        Perform comprehensive bias detection analysis.

        Args:
            movies: List of validated movies

        Returns:
            BiasMetrics with detailed bias analysis
        """
        logger.info(f"Starting bias detection analysis on {len(movies)} movies")

        metrics = BiasMetrics()

        if not movies:
            logger.warning("No movies provided for bias analysis")
            return metrics

        # Convert to DataFrame for easier analysis
        df = self._movies_to_dataframe(movies)

        # Genre diversity analysis
        if self.bias_config.get("enable_genre_analysis", True):
            metrics.genre_diversity = self._analyze_genre_diversity(df, movies)

        # Demographic representation analysis
        if self.bias_config.get("enable_demographic_analysis", True):
            metrics.demographic_representation = self._analyze_demographic_representation(movies)

        # Geographic distribution analysis
        if self.bias_config.get("enable_geographic_analysis", True):
            metrics.geographic_distribution = self._analyze_geographic_distribution(movies)

        # Temporal distribution analysis
        if self.bias_config.get("enable_temporal_analysis", True):
            metrics.temporal_distribution = self._analyze_temporal_distribution(df)

        # Rating bias analysis
        metrics.rating_bias_analysis = self._analyze_rating_bias(df, movies)

        # Calculate overall bias score
        metrics.overall_bias_score = self._calculate_overall_bias_score(metrics)

        # Generate recommendations
        metrics.recommendations = self._generate_bias_recommendations(metrics)

        logger.info(f"Bias analysis complete. Overall bias score: {metrics.overall_bias_score:.3f}")

        return metrics

    def _movies_to_dataframe(self, movies: List[Movie]) -> pd.DataFrame:
        """Convert movies to pandas DataFrame for analysis."""
        data = []

        for movie in movies:
            row = {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "release_year": movie.release_year,
                "runtime_mins": movie.runtime_mins,
                "rating_average": movie.ratings.average,
                "rating_count": movie.ratings.count,
                "genre_count": len(movie.genres),
                "primary_genre": movie.genres[0].value if movie.genres else None,
                "language": (movie.metadata.language.value if movie.metadata else Language.EN.value),
                "country": movie.metadata.country if movie.metadata else "US",
                "budget": movie.metadata.budget if movie.metadata else None,
                "revenue": movie.metadata.revenue if movie.metadata else None,
                "cast_size": len(movie.cast) if movie.cast else 0,
                "crew_size": len(movie.crew) if movie.crew else 0,
            }

            # Add genre flags
            all_genres = set(genre.value for genre in Genre)
            for genre in all_genres:
                row[f'genre_{genre.lower().replace(" ", "_")}'] = genre in [g.value for g in movie.genres]

            data.append(row)

        return pd.DataFrame(data)

    def _analyze_genre_diversity(self, df: pd.DataFrame, movies: List[Movie]) -> Dict[str, float]:
        """Analyze genre representation and diversity."""
        genre_analysis = {}

        # Count genre occurrences
        genre_counts = Counter()
        total_movies = len(movies)

        for movie in movies:
            for genre in movie.genres:
                genre_counts[genre.value] += 1

        # Calculate genre representation percentages
        genre_percentages = {genre: count / total_movies for genre, count in genre_counts.items()}

        # Shannon diversity index for genres
        if genre_counts:
            total_genre_instances = sum(genre_counts.values())
            shannon_diversity = -sum(
                (count / total_genre_instances) * np.log(count / total_genre_instances)
                for count in genre_counts.values()
                if count > 0
            )
        else:
            shannon_diversity = 0.0

        genre_analysis["shannon_diversity"] = shannon_diversity
        genre_analysis["unique_genres"] = len(genre_counts)
        genre_analysis["total_genre_instances"] = sum(genre_counts.values())
        genre_analysis["average_genres_per_movie"] = sum(genre_counts.values()) / total_movies
        genre_analysis["genre_distribution"] = genre_percentages

        # Identify under-represented genres
        min_representation = 0.01  # 1% threshold
        underrepresented = [genre for genre, pct in genre_percentages.items() if pct < min_representation]
        genre_analysis["underrepresented_genres"] = underrepresented

        # Genre concentration (Gini coefficient approximation)
        sorted_counts = sorted(genre_counts.values(), reverse=True)
        n = len(sorted_counts)
        if n > 1:
            cumsum = np.cumsum(sorted_counts)
            gini = (n + 1 - 2 * sum((n + 1 - i) * count for i, count in enumerate(sorted_counts, 1))) / (
                n * sum(sorted_counts)
            )
        else:
            gini = 0.0
        genre_analysis["concentration_gini"] = gini

        return genre_analysis

    def _analyze_demographic_representation(self, movies: List[Movie]) -> Dict[str, Dict[str, float]]:
        """Analyze demographic representation in cast and crew."""
        demo_analysis = {"cast": defaultdict(list), "crew": defaultdict(list)}

        total_cast_members = 0
        total_crew_members = 0

        for movie in movies:
            # Analyze cast demographics
            if movie.cast:
                for person in movie.cast:
                    total_cast_members += 1
                    demo_analysis["cast"]["gender"].append(person.gender or "unknown")
                    demo_analysis["cast"]["ethnicity"].append(person.ethnicity or "unknown")

                    # Age analysis (approximate from birth year)
                    if person.birth_year:
                        age_at_release = movie.release_year - person.birth_year
                        demo_analysis["cast"]["age_at_release"].append(age_at_release)

            # Analyze crew demographics
            if movie.crew:
                for person in movie.crew:
                    total_crew_members += 1
                    demo_analysis["crew"]["gender"].append(person.gender or "unknown")
                    demo_analysis["crew"]["ethnicity"].append(person.ethnicity or "unknown")
                    demo_analysis["crew"]["role"].append(person.role)

                    if person.birth_year:
                        age_at_release = movie.release_year - person.birth_year
                        demo_analysis["crew"]["age_at_release"].append(age_at_release)

        # Convert to percentage distributions
        result = {}

        for group in ["cast", "crew"]:
            result[group] = {}

            for demographic, values in demo_analysis[group].items():
                if demographic in ["age_at_release"]:
                    # Numerical analysis
                    if values:
                        result[group][demographic] = {
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "std": statistics.stdev(values) if len(values) > 1 else 0,
                            "min": min(values),
                            "max": max(values),
                        }
                else:
                    # Categorical analysis
                    if values:
                        counts = Counter(values)
                        total = len(values)
                        result[group][demographic] = {category: count / total for category, count in counts.items()}

        return result

    def _analyze_geographic_distribution(self, movies: List[Movie]) -> Dict[str, float]:
        """Analyze geographic representation."""
        geo_analysis = {}

        # Country distribution
        countries = []
        languages = []

        for movie in movies:
            if movie.metadata:
                countries.append(movie.metadata.country)
                languages.append(movie.metadata.language.value)
            else:
                countries.append("US")  # Default
                languages.append(Language.EN.value)  # Default

        # Calculate distributions
        total_movies = len(movies)

        country_counts = Counter(countries)
        geo_analysis["country_distribution"] = {
            country: count / total_movies for country, count in country_counts.items()
        }

        language_counts = Counter(languages)
        geo_analysis["language_distribution"] = {
            language: count / total_movies for language, count in language_counts.items()
        }

        # Diversity metrics
        geo_analysis["unique_countries"] = len(country_counts)
        geo_analysis["unique_languages"] = len(language_counts)

        # Concentration analysis
        dominant_country_pct = max(country_counts.values()) / total_movies
        dominant_language_pct = max(language_counts.values()) / total_movies

        geo_analysis["dominant_country_percentage"] = dominant_country_pct
        geo_analysis["dominant_language_percentage"] = dominant_language_pct

        return geo_analysis

    def _analyze_temporal_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze temporal patterns and bias."""
        temporal_analysis = {}

        if df.empty:
            return temporal_analysis

        years = df["release_year"].values

        # Basic temporal statistics
        temporal_analysis["year_range"] = {
            "min": int(years.min()),
            "max": int(years.max()),
            "span": int(years.max() - years.min()),
        }

        temporal_analysis["year_statistics"] = {
            "mean": float(years.mean()),
            "median": float(np.median(years)),
            "std": float(years.std()),
        }

        # Decade distribution
        decades = (years // 10) * 10
        decade_counts = Counter(decades)
        total_movies = len(df)

        temporal_analysis["decade_distribution"] = {
            f"{int(decade)}s": count / total_movies for decade, count in decade_counts.items()
        }

        # Identify temporal bias (over-representation of recent movies)
        recent_threshold = 2000
        recent_count = sum(1 for year in years if year >= recent_threshold)
        recent_percentage = recent_count / total_movies

        temporal_analysis["modern_bias"] = {
            "recent_movies_pct": recent_percentage,
            "threshold_year": recent_threshold,
            "is_biased_toward_recent": recent_percentage > 0.7,  # 70% threshold
        }

        return temporal_analysis

    def _analyze_rating_bias(self, df: pd.DataFrame, movies: List[Movie]) -> Dict[str, Any]:
        """Analyze potential bias in ratings."""
        rating_analysis = {}

        if df.empty:
            return rating_analysis

        # Basic rating statistics
        ratings = df["rating_average"].values
        rating_counts = df["rating_count"].values

        rating_analysis["rating_statistics"] = {
            "mean": float(ratings.mean()),
            "median": float(np.median(ratings)),
            "std": float(ratings.std()),
            "min": float(ratings.min()),
            "max": float(ratings.max()),
        }

        rating_analysis["rating_count_statistics"] = {
            "mean": float(rating_counts.mean()),
            "median": float(np.median(rating_counts)),
            "std": float(rating_counts.std()),
        }

        # Rating bias by genre (if sufficient data)
        genre_rating_analysis = {}
        for genre in Genre:
            genre_col = f'genre_{genre.value.lower().replace(" ", "_")}'
            if genre_col in df.columns:
                genre_movies = df[df[genre_col] == True]
                if len(genre_movies) >= 10:  # Minimum sample size
                    genre_rating_analysis[genre.value] = {
                        "mean_rating": float(genre_movies["rating_average"].mean()),
                        "movie_count": len(genre_movies),
                    }

        rating_analysis["rating_by_genre"] = genre_rating_analysis

        # Rating bias by time period
        time_rating_analysis = {}
        for decade in [1970, 1980, 1990, 2000, 2010, 2020]:
            decade_movies = df[(df["release_year"] >= decade) & (df["release_year"] < decade + 10)]
            if len(decade_movies) >= 5:
                time_rating_analysis[f"{decade}s"] = {
                    "mean_rating": float(decade_movies["rating_average"].mean()),
                    "movie_count": len(decade_movies),
                }

        rating_analysis["rating_by_decade"] = time_rating_analysis

        return rating_analysis

    def _calculate_overall_bias_score(self, metrics: BiasMetrics) -> float:
        """Calculate overall bias score (0 = no bias, 1 = high bias)."""
        bias_components = []

        # Genre concentration bias
        if metrics.genre_diversity.get("concentration_gini"):
            bias_components.append(metrics.genre_diversity["concentration_gini"])

        # Geographic concentration bias
        if metrics.geographic_distribution.get("dominant_country_percentage"):
            country_bias = min(metrics.geographic_distribution["dominant_country_percentage"], 1.0)
            bias_components.append(country_bias)

        # Temporal bias (modern bias)
        if metrics.temporal_distribution.get("modern_bias", {}).get("recent_movies_pct"):
            temporal_bias = abs(metrics.temporal_distribution["modern_bias"]["recent_movies_pct"] - 0.5) * 2
            bias_components.append(temporal_bias)

        # Demographic bias (simplified - check gender balance)
        if "cast" in metrics.demographic_representation:
            gender_dist = metrics.demographic_representation["cast"].get("gender", {})
            if "male" in gender_dist and "female" in gender_dist:
                gender_imbalance = abs(gender_dist["male"] - gender_dist["female"])
                bias_components.append(gender_imbalance)

        # Calculate weighted average
        if bias_components:
            overall_bias = sum(bias_components) / len(bias_components)
        else:
            overall_bias = 0.0

        return min(max(overall_bias, 0.0), 1.0)  # Ensure between 0.0 and 1.0

    def _generate_bias_recommendations(self, metrics: BiasMetrics) -> List[str]:
        """Generate recommendations to reduce bias."""
        recommendations = []

        # Genre diversity recommendations
        if metrics.genre_diversity.get("concentration_gini", 0) > 0.7:
            recommendations.append("Consider including more diverse genres to reduce concentration bias")

        underrepresented = metrics.genre_diversity.get("underrepresented_genres", [])
        if underrepresented:
            recommendations.append(
                f"Increase representation of underrepresented genres: {', '.join(underrepresented[:3])}"
            )

        # Geographic recommendations
        dominant_country_pct = metrics.geographic_distribution.get("dominant_country_percentage", 0)
        if dominant_country_pct > 0.8:
            recommendations.append("Consider including more international films to reduce geographic bias")

        # Temporal recommendations
        modern_bias = metrics.temporal_distribution.get("modern_bias", {})
        if modern_bias.get("is_biased_toward_recent", False):
            recommendations.append("Include more classic/historical films to balance temporal distribution")

        # Demographic recommendations
        cast_gender = metrics.demographic_representation.get("cast", {}).get("gender", {})
        if "male" in cast_gender and "female" in cast_gender:
            male_pct = cast_gender["male"]
            female_pct = cast_gender["female"]
            if abs(male_pct - female_pct) > 0.3:
                recommendations.append("Consider improving gender balance in cast representation")

        if not recommendations:
            recommendations.append("Dataset shows good diversity with minimal detected bias")

        return recommendations

    def generate_html_report(
        self,
        validation_result: ValidationResult,
        bias_metrics: BiasMetrics,
        movies: List[Movie],
        output_path: str = "data/reports/data_quality_report.html",
    ) -> None:
        """Generate comprehensive HTML report with visualizations."""

        logger.info(f"Generating HTML report: {output_path}")

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate visualizations
        viz_paths = self._create_visualizations(bias_metrics, movies, Path(output_path).parent)

        # HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>MovieRecs Data Quality & Bias Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; margin-top: 30px; }
        h3 { color: #555; }
        .metric { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2980b9; }
        .error { background: #ffebee; border-left: 4px solid #e74c3c; padding: 10px; margin: 5px 0; }
        .warning { background: #fff3e0; border-left: 4px solid #f39c12; padding: 10px; margin: 5px 0; }
        .success { background: #e8f5e8; border-left: 4px solid #27ae60; padding: 10px; margin: 5px 0; }
        .recommendation { background: #e3f2fd; border-left: 4px solid #2196f3; padding: 10px; margin: 10px 0; }
        .viz-container { text-align: center; margin: 20px 0; }
        .viz-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
        .completeness-bar { width: 100%; height: 20px; background: #ecf0f1; border-radius: 10px; overflow: hidden; }
        .completeness-fill { height: 100%; background: #27ae60; transition: width 0.3s; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .bias-score { font-size: 48px; font-weight: bold; text-align: center; margin: 20px 0; }
        .bias-low { color: #27ae60; }
        .bias-medium { color: #f39c12; }
        .bias-high { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 MovieRecs Data Quality & Bias Analysis Report</h1>
        
        <div class="metric">
            <strong>Report Generated:</strong> {{ report_date }}<br>
            <strong>Dataset Size:</strong> {{ total_movies }} movies<br>
            <strong>Valid Movies:</strong> {{ valid_movies }} ({{ validity_rate }}%)<br>
            <strong>Schema Version:</strong> 1.0
        </div>
        
        <h2>📊 Validation Summary</h2>
        {% if validation_result.is_valid %}
        <div class="success">✅ Dataset validation passed all quality checks</div>
        {% else %}
        <div class="error">❌ Dataset validation found {{ validation_result.error_count }} errors and {{ validation_result.warning_count }} warnings</div>
        {% endif %}
        
        <div class="stats-grid">
            <div class="metric">
                <div class="metric-value">{{ validation_result.error_count }}</div>
                <div>Validation Errors</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ validation_result.warning_count }}</div>
                <div>Validation Warnings</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(bias_metrics.overall_bias_score * 100) }}%</div>
                <div>Overall Bias Score</div>
            </div>
        </div>
        
        <h3>Field Completeness</h3>
        {% for field, completeness in field_completeness.items() %}
        <div style="margin: 10px 0;">
            <strong>{{ field }}:</strong> {{ "%.1f"|format(completeness * 100) }}%
            <div class="completeness-bar">
                <div class="completeness-fill" style="width: {{ completeness * 100 }}%;"></div>
            </div>
        </div>
        {% endfor %}
        
        {% if validation_result.errors %}
        <h3>🚨 Validation Errors</h3>
        {% for error in validation_result.errors[:10] %}
        <div class="error">
            <strong>{{ error.movie_id }}</strong> - {{ error.field }}: {{ error.message }}
        </div>
        {% endfor %}
        {% if validation_result.errors|length > 10 %}
        <p><em>... and {{ validation_result.errors|length - 10 }} more errors</em></p>
        {% endif %}
        {% endif %}
        
        <h2>🎭 Bias Analysis</h2>
        
        <div class="bias-score {% if bias_metrics.overall_bias_score < 0.3 %}bias-low{% elif bias_metrics.overall_bias_score < 0.7 %}bias-medium{% else %}bias-high{% endif %}">
            Bias Score: {{ "%.1f"|format(bias_metrics.overall_bias_score * 100) }}%
        </div>
        
        <h3>📈 Visualizations</h3>
        {% for viz_name, viz_path in visualizations.items() %}
        <div class="viz-container">
            <h4>{{ viz_name }}</h4>
            <img src="{{ viz_path }}" alt="{{ viz_name }}">
        </div>
        {% endfor %}
        
        <h3>🌍 Geographic Distribution</h3>
        <div class="stats-grid">
            <div class="metric">
                <div class="metric-value">{{ bias_metrics.geographic_distribution.unique_countries }}</div>
                <div>Unique Countries</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ bias_metrics.geographic_distribution.unique_languages }}</div>
                <div>Unique Languages</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(bias_metrics.geographic_distribution.dominant_country_percentage * 100) }}%</div>
                <div>Dominant Country</div>
            </div>
        </div>
        
        <h3>🎬 Genre Analysis</h3>
        <div class="stats-grid">
            <div class="metric">
                <div class="metric-value">{{ bias_metrics.genre_diversity.unique_genres }}</div>
                <div>Unique Genres</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.2f"|format(bias_metrics.genre_diversity.shannon_diversity) }}</div>
                <div>Shannon Diversity</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(bias_metrics.genre_diversity.average_genres_per_movie) }}</div>
                <div>Avg Genres per Movie</div>
            </div>
        </div>
        
        <h3>📅 Temporal Distribution</h3>
        <div class="stats-grid">
            <div class="metric">
                <div class="metric-value">{{ bias_metrics.temporal_distribution.year_range.span }}</div>
                <div>Year Range ({{ bias_metrics.temporal_distribution.year_range.min }} - {{ bias_metrics.temporal_distribution.year_range.max }})</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(bias_metrics.temporal_distribution.modern_bias.recent_movies_pct * 100) }}%</div>
                <div>Recent Movies (2000+)</div>
            </div>
        </div>
        
        <h2>💡 Recommendations</h2>
        {% for recommendation in bias_metrics.recommendations %}
        <div class="recommendation">
            {{ loop.index }}. {{ recommendation }}
        </div>
        {% endfor %}
        
        <h2>📋 Technical Details</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Validation Runtime</td><td>{{ validation_runtime }}</td></tr>
            <tr><td>Memory Usage</td><td>{{ memory_usage }}</td></tr>
            <tr><td>Configuration</td><td>{{ config_summary }}</td></tr>
        </table>
        
        <hr style="margin: 40px 0;">
        <p style="text-align: center; color: #7f8c8d;">
            <em>Generated by MovieRecs Data Pipeline v1.0</em><br>
            <small>Report ID: {{ report_id }}</small>
        </p>
    </div>
</body>
</html>
        """

        # Prepare template data
        template_data = {
            "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_movies": validation_result.summary["total_movies"],
            "valid_movies": validation_result.summary["valid_movies"],
            "validity_rate": round(validation_result.summary["completeness_rate"] * 100, 1),
            "validation_result": validation_result,
            "bias_metrics": bias_metrics,
            "field_completeness": validation_result.summary.get("field_completeness", {}),
            "visualizations": viz_paths,
            "validation_runtime": "N/A",  # Would track actual runtime
            "memory_usage": "N/A",  # Would track actual memory usage
            "config_summary": "Default configuration",
            "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }

        # Render template
        template = Template(html_template)
        html_content = template.render(**template_data)

        # Write report
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {output_path}")

    def _create_visualizations(
        self, bias_metrics: BiasMetrics, movies: List[Movie], output_dir: Path
    ) -> Dict[str, str]:
        """Create visualization plots for the report."""

        viz_paths = {}

        try:
            # Set up matplotlib
            plt.style.use("default")
            sns.set_palette("husl")

            # 1. Genre distribution pie chart
            if bias_metrics.genre_diversity.get("genre_distribution"):
                plt.figure(figsize=(10, 8))
                genre_data = bias_metrics.genre_diversity["genre_distribution"]

                # Take top 10 genres for readability
                sorted_genres = sorted(genre_data.items(), key=lambda x: x[1], reverse=True)[:10]
                labels, sizes = zip(*sorted_genres)

                plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
                plt.title("Genre Distribution", fontsize=16, fontweight="bold")
                plt.axis("equal")

                viz_path = output_dir / "genre_distribution.png"
                plt.savefig(viz_path, dpi=300, bbox_inches="tight")
                plt.close()
                viz_paths["Genre Distribution"] = viz_path.name

            # 2. Release year histogram
            if movies:
                plt.figure(figsize=(12, 6))
                years = [movie.release_year for movie in movies]

                plt.hist(years, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
                plt.title("Movies by Release Year", fontsize=16, fontweight="bold")
                plt.xlabel("Release Year")
                plt.ylabel("Number of Movies")
                plt.grid(True, alpha=0.3)

                viz_path = output_dir / "year_distribution.png"
                plt.savefig(viz_path, dpi=300, bbox_inches="tight")
                plt.close()
                viz_paths["Release Year Distribution"] = viz_path.name

            # 3. Rating distribution
            if movies:
                plt.figure(figsize=(10, 6))
                ratings = [movie.ratings.average for movie in movies]

                plt.hist(ratings, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
                plt.title("Rating Distribution", fontsize=16, fontweight="bold")
                plt.xlabel("Average Rating")
                plt.ylabel("Number of Movies")
                plt.grid(True, alpha=0.3)

                viz_path = output_dir / "rating_distribution.png"
                plt.savefig(viz_path, dpi=300, bbox_inches="tight")
                plt.close()
                viz_paths["Rating Distribution"] = viz_path.name

        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")

        return viz_paths


# Import numpy for bias calculations
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available, some bias calculations may be limited")
    np = None


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load sample configuration
    config = {
        "quality_thresholds": {"completeness_min": 0.9, "synopsis_min_length": 50},
        "bias_detection": {
            "enable_genre_analysis": True,
            "enable_demographic_analysis": True,
            "enable_geographic_analysis": True,
            "enable_temporal_analysis": True,
        },
    }

    validator = DataValidator(config)

    # Example with sample data
    sample_movies = [
        {
            "movie_id": "1",
            "title": "The Shawshank Redemption",
            "synopsis": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
            "release_year": 1994,
            "runtime_mins": 142,
            "genres": ["Drama"],
            "ratings": {"average": 9.3, "count": 2500000},
        }
    ]

    # Validate dataset
    validation_result, valid_movies = validator.validate_dataset(sample_movies)
    print(f"Validation: {validation_result.is_valid}, Valid movies: {len(valid_movies)}")

    # Detect bias
    if valid_movies:
        bias_metrics = validator.detect_bias(valid_movies)
        print(f"Bias score: {bias_metrics.overall_bias_score:.3f}")

        # Generate report
        validator.generate_html_report(validation_result, bias_metrics, valid_movies)
        print("HTML report generated!")
