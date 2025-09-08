"""
Comprehensive test suite for Constitutional AI validation and bias detection.

Tests all bias detection algorithms, diversity calculations, Constitutional AI
compliance validation, and performance with large datasets.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import pytest

from src.constitutional_ai_validator import (
    BiasDetector,
    BiasSeverity,
    BiasType,
    ConstitutionalAIValidator,
    DemographicAnalyzer,
    DiversityCalculator,
)
from src.movie_schema import Genre, Language, MovieMetadata, MovieSchema, RatingInfo


class TestDiversityCalculator:
    """Test suite for diversity calculation algorithms."""

    def test_shannon_diversity_index(self):
        """Test Shannon diversity index calculation."""
        # Perfect diversity (all equal)
        equal_counts = {"A": 10, "B": 10, "C": 10, "D": 10}
        shannon = DiversityCalculator.shannon_diversity_index(equal_counts)
        expected = 1.386  # log(4) ≈ 1.386
        assert abs(shannon - expected) < 0.01

        # No diversity (single category)
        single_counts = {"A": 40}
        shannon = DiversityCalculator.shannon_diversity_index(single_counts)
        assert shannon == 0.0

        # Partial diversity
        mixed_counts = {"A": 20, "B": 15, "C": 10, "D": 5}
        shannon = DiversityCalculator.shannon_diversity_index(mixed_counts)
        assert 0 < shannon < 1.386

        # Empty input
        shannon = DiversityCalculator.shannon_diversity_index({})
        assert shannon == 0.0

    def test_simpson_diversity_index(self):
        """Test Simpson's diversity index calculation."""
        # High diversity
        diverse_counts = {"A": 25, "B": 25, "C": 25, "D": 25}
        simpson = DiversityCalculator.simpson_diversity_index(diverse_counts)
        assert simpson > 0.7  # High diversity

        # Low diversity
        skewed_counts = {"A": 90, "B": 5, "C": 3, "D": 2}
        simpson = DiversityCalculator.simpson_diversity_index(skewed_counts)
        assert simpson < 0.3  # Low diversity

        # Single category
        single_counts = {"A": 100}
        simpson = DiversityCalculator.simpson_diversity_index(single_counts)
        assert simpson == 0.0

    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        # Perfect equality
        equal_values = [10, 10, 10, 10]
        gini = DiversityCalculator.gini_coefficient(equal_values)
        assert abs(gini) < 0.01  # Should be close to 0

        # Perfect inequality
        unequal_values = [100, 0, 0, 0]
        gini = DiversityCalculator.gini_coefficient(unequal_values)
        assert gini > 0.7  # High inequality

        # Moderate inequality
        moderate_values = [40, 30, 20, 10]
        gini = DiversityCalculator.gini_coefficient(moderate_values)
        assert 0.1 < gini < 0.5

    def test_normalized_diversity_score(self):
        """Test normalized diversity score calculation."""
        # High diversity scenario
        diverse_counts = {"A": 20, "B": 18, "C": 16, "D": 14, "E": 12}
        score = DiversityCalculator.normalized_diversity_score(diverse_counts, 10)
        assert score > 0.6

        # Low diversity scenario
        skewed_counts = {"A": 80, "B": 10, "C": 5, "D": 5}
        score = DiversityCalculator.normalized_diversity_score(skewed_counts, 10)
        assert score < 0.5

        # Empty input
        score = DiversityCalculator.normalized_diversity_score({})
        assert score == 0.0


class TestDemographicAnalyzer:
    """Test suite for demographic analysis."""

    def test_analyze_cast_demographics(self):
        """Test cast demographic analysis."""
        from src.movie_schema import CastMember

        cast = [
            CastMember(name="John Smith", order=0, gender="male", ethnicity="Caucasian", birth_year=1980),
            CastMember(name="Jane Doe", order=1, gender="female", ethnicity="African American", birth_year=1985),
            CastMember(name="Carlos Rodriguez", order=2, gender="male", ethnicity="Hispanic", birth_year=1990),
            CastMember(name="Liu Wei", order=3, gender="female", ethnicity="Asian", birth_year=1988),
        ]

        analysis = DemographicAnalyzer.analyze_cast_demographics(cast)

        assert analysis["total_cast"] == 4
        assert "gender_distribution" in analysis
        assert analysis["gender_distribution"]["male"] == 0.5
        assert analysis["gender_distribution"]["female"] == 0.5

        assert "ethnicity_distribution" in analysis
        assert len(analysis["ethnicity_distribution"]) == 4

        assert "age_distribution" in analysis
        assert "mean_age" in analysis["age_distribution"]

    def test_analyze_cast_bias_detection(self):
        """Test cast bias flag detection."""
        from src.movie_schema import CastMember

        # Gender-imbalanced cast
        biased_cast = [CastMember(name=f"Actor {i}", order=i, gender="male") for i in range(8)] + [
            CastMember(name=f"Actress {i}", order=i + 8, gender="female") for i in range(2)
        ]

        analysis = DemographicAnalyzer.analyze_cast_demographics(biased_cast)

        bias_flags = analysis["demographic_flags"]
        assert any("gender imbalance" in flag.lower() for flag in bias_flags)

    def test_analyze_crew_demographics(self):
        """Test crew demographic analysis."""
        from src.movie_schema import CrewMember

        crew = [
            CrewMember(name="John Director", job="Director", department="Directing", gender="male"),
            CrewMember(name="Jane Writer", job="Writer", department="Writing", gender="female"),
            CrewMember(name="Bob Producer", job="Producer", department="Production", gender="male"),
            CrewMember(name="Alice Editor", job="Editor", department="Editing", gender="female"),
        ]

        analysis = DemographicAnalyzer.analyze_crew_demographics(crew)

        assert analysis["total_crew"] == 4
        assert "role_distribution" in analysis
        assert "gender_by_role" in analysis
        assert analysis["gender_by_role"]["Director"]["male"] == 1.0

    def test_crew_key_role_bias_detection(self):
        """Test detection of bias in key crew roles."""
        from src.movie_schema import CrewMember

        # Male-dominated key roles
        biased_crew = [
            CrewMember(name="Male Director 1", job="Director", department="Directing", gender="male"),
            CrewMember(name="Male Director 2", job="Director", department="Directing", gender="male"),
            CrewMember(name="Male Writer 1", job="Writer", department="Writing", gender="male"),
            CrewMember(name="Male Writer 2", job="Writer", department="Writing", gender="male"),
            CrewMember(name="Male Producer", job="Producer", department="Production", gender="male"),
        ]

        analysis = DemographicAnalyzer.analyze_crew_demographics(biased_crew)

        bias_flags = analysis["demographic_flags"]
        assert any("Director" in flag and "male-dominated" in flag for flag in bias_flags)


class TestBiasDetector:
    """Test suite for bias detection algorithms."""

    def setUp(self):
        """Set up test data."""
        self.bias_detector = BiasDetector()

    def create_test_movies(self, scenarios: List[Dict[str, Any]]) -> List[MovieSchema]:
        """Create test movies from scenarios."""
        movies = []
        for i, scenario in enumerate(scenarios):
            movie_data = {
                "movie_id": f"test_{i}",
                "title": f"Test Movie {i}",
                "synopsis": "A comprehensive test movie synopsis that meets all minimum requirements for validation and testing purposes.",
                "release_year": scenario.get("release_year", 2020),
                "genres": scenario.get("genres", [Genre.DRAMA]),
                "ratings": RatingInfo(average=7.0, count=1000),
                "metadata": MovieMetadata(
                    original_language=scenario.get("language", Language.EN),
                    origin_country=scenario.get("country", "US"),
                ),
            }
            movies.append(MovieSchema(**movie_data))
        return movies

    def test_detect_geographic_bias(self):
        """Test geographic bias detection."""
        # Create dataset with US bias
        scenarios = [{"country": "US"} for _ in range(8)] + [{"country": "CA"} for _ in range(2)]

        movies = self.create_test_movies(scenarios)
        bias_detector = BiasDetector()

        analysis = bias_detector.detect_geographic_bias(movies)

        assert analysis["bias_type"] == BiasType.GEOGRAPHIC
        assert analysis["dominant_percentage"] == 0.8  # 80% US
        assert analysis["severity"] in [BiasSeverity.MEDIUM, BiasSeverity.HIGH]
        assert len(analysis["bias_flags"]) > 0
        assert any("US over-representation" in flag for flag in analysis["bias_flags"])

    def test_detect_language_bias(self):
        """Test language bias detection."""
        # Create dataset with English bias
        scenarios = [{"language": Language.EN} for _ in range(9)] + [{"language": Language.ES}]

        movies = self.create_test_movies(scenarios)
        bias_detector = BiasDetector()

        analysis = bias_detector.detect_language_bias(movies)

        assert analysis["bias_type"] == BiasType.LANGUAGE
        assert analysis["english_percentage"] == 0.9  # 90% English
        assert analysis["severity"] == BiasSeverity.HIGH
        assert any("English over-representation" in flag for flag in analysis["bias_flags"])

    def test_detect_temporal_bias(self):
        """Test temporal bias detection."""
        # Create dataset with modern bias
        scenarios = [{"release_year": year} for year in ([2020, 2021, 2022, 2023] * 2 + [1990, 1995])]  # 80% modern

        movies = self.create_test_movies(scenarios)
        bias_detector = BiasDetector()

        analysis = bias_detector.detect_temporal_bias(movies)

        assert analysis["bias_type"] == BiasType.TEMPORAL
        assert analysis["modern_percentage"] == 0.8  # 80% from 2000+
        assert analysis["severity"] == BiasSeverity.MEDIUM
        assert any("Modern bias" in flag for flag in analysis["bias_flags"])

    def test_detect_genre_bias(self):
        """Test genre bias detection."""
        # Create dataset with genre concentration
        scenarios = (
            [{"genres": [Genre.ACTION]} for _ in range(6)]
            + [{"genres": [Genre.DRAMA]} for _ in range(2)]
            + [{"genres": [Genre.COMEDY]} for _ in range(2)]
        )

        movies = self.create_test_movies(scenarios)
        bias_detector = BiasDetector()

        analysis = bias_detector.detect_genre_bias(movies)

        assert analysis["bias_type"] == BiasType.GENRE
        assert analysis["dominant_genre_percentage"] == 0.6  # 60% Action
        assert any("over-concentration" in flag for flag in analysis["bias_flags"])

    def test_detect_demographic_bias(self):
        """Test demographic bias detection."""
        from src.movie_schema import CastMember, CrewMember

        # Create movies with demographic bias
        movies = []
        for i in range(5):
            cast = [CastMember(name=f"Male Actor {j}", order=j, gender="male") for j in range(4)] + [
                CastMember(name=f"Female Actor {j}", order=j + 4, gender="female") for j in range(1)
            ]

            crew = [CrewMember(name=f"Male Director {i}", job="Director", department="Directing", gender="male")]

            movie_data = {
                "movie_id": f"demo_test_{i}",
                "title": f"Demo Test Movie {i}",
                "synopsis": "A test movie for demographic bias detection with gender imbalanced cast and crew.",
                "release_year": 2020,
                "genres": [Genre.DRAMA],
                "ratings": RatingInfo(average=7.0, count=1000),
                "cast": cast,
                "crew": crew,
            }
            movies.append(MovieSchema(**movie_data))

        bias_detector = BiasDetector()
        analysis = bias_detector.detect_demographic_bias(movies)

        assert analysis["bias_type"] == BiasType.DEMOGRAPHIC
        assert analysis["severity"] in [BiasSeverity.MEDIUM, BiasSeverity.HIGH]
        bias_flags = analysis["bias_flags"]
        assert any("gender imbalance" in flag.lower() for flag in bias_flags)
        assert any("male-dominated" in flag.lower() for flag in bias_flags)


class TestConstitutionalAIValidator:
    """Test suite for main Constitutional AI validator."""

    def test_validate_constitutional_compliance(self):
        """Test comprehensive Constitutional AI compliance validation."""
        # Load test data
        test_data_path = os.path.join(os.path.dirname(__file__), "fixtures", "movie_test_data.json")
        with open(test_data_path, "r") as f:
            test_data = json.load(f)

        # Convert test data to MovieSchema objects
        movies = []
        for movie_data in test_data["diverse_dataset"]:
            # Add required fields for validation
            movie_data["ratings"] = movie_data.get("ratings", {"average": 7.0, "count": 1000})
            movie_data["synopsis"] = movie_data.get(
                "synopsis", "A test synopsis for validation purposes that meets minimum length requirements."
            )

            movie = MovieSchema(**movie_data)
            movies.append(movie)

        # Initialize validator
        validator = ConstitutionalAIValidator()

        # Perform validation
        compliance_report = validator.validate_constitutional_compliance(movies)

        # Verify report structure
        assert "validation_timestamp" in compliance_report
        assert "total_movies" in compliance_report
        assert "overall_compliance" in compliance_report
        assert "bias_analyses" in compliance_report
        assert "overall_bias_score" in compliance_report
        assert "constitutional_ai_metrics" in compliance_report
        assert "recommendations" in compliance_report

        # Verify bias analyses
        bias_analyses = compliance_report["bias_analyses"]
        expected_bias_types = ["geographic", "language", "temporal", "genre", "demographic"]

        for bias_type in expected_bias_types:
            assert bias_type in bias_analyses
            analysis = bias_analyses[bias_type]
            assert "bias_type" in analysis
            assert "severity" in analysis
            assert "bias_flags" in analysis
            assert "recommendations" in analysis

    def test_compliance_scoring(self):
        """Test compliance scoring and status determination."""
        validator = ConstitutionalAIValidator()

        # Mock bias analyses with different severities
        mock_analyses = {
            "geographic": {"severity": BiasSeverity.LOW},
            "language": {"severity": BiasSeverity.MEDIUM},
            "temporal": {"severity": BiasSeverity.HIGH},
            "genre": {"severity": BiasSeverity.LOW},
            "demographic": {"severity": BiasSeverity.MEDIUM},
        }

        score, status = validator._calculate_overall_compliance(mock_analyses)

        assert 0.0 <= score <= 1.0
        assert status in ["PASS", "WARNING", "FAIL", "CRITICAL"]

        # High severity should result in FAIL or CRITICAL
        assert status in ["FAIL", "CRITICAL"]

    def test_update_movie_constitutional_metrics(self):
        """Test updating individual movie Constitutional AI metrics."""
        validator = ConstitutionalAIValidator()

        # Create test movie
        movie = MovieSchema(
            movie_id="metric_test",
            title="Metrics Test Movie",
            synopsis="A test movie for Constitutional AI metrics updating and validation testing purposes.",
            release_year=2023,
            genres=[Genre.DRAMA],
            ratings=RatingInfo(average=7.5, count=2000),
            metadata=MovieMetadata(original_language=Language.EN, origin_country="US"),
        )

        # Mock dataset analysis
        mock_analysis = {
            "overall_bias_score": 0.4,
            "bias_analyses": {
                "geographic": {"country_distribution": {"US": 0.8, "CA": 0.2}},
                "language": {"language_distribution": {"en": 0.9, "fr": 0.1}},
            },
            "constitutional_ai_metrics": {
                "bias_flag_summary": ["US over-representation", "English over-representation"]
            },
        }

        # Update movie metrics
        updated_movie = validator.update_movie_constitutional_metrics(movie, mock_analysis)

        # Verify metrics were updated
        assert updated_movie.constitutional_ai.overall_bias_score == 0.4
        assert updated_movie.constitutional_ai.geographic_diversity_score == 0.2  # 1.0 - 0.8
        assert updated_movie.constitutional_ai.language_diversity_score == 0.1  # 1.0 - 0.9
        assert "contributes_to_geographic_bias" in updated_movie.constitutional_ai.bias_flags
        assert "contributes_to_language_bias" in updated_movie.constitutional_ai.bias_flags

    def test_diverse_dataset_validation(self):
        """Test validation with a properly diverse dataset."""
        # Create balanced dataset
        diverse_scenarios = [
            {"country": "US", "language": Language.EN, "release_year": 2020, "genres": [Genre.DRAMA]},
            {"country": "FR", "language": Language.FR, "release_year": 2019, "genres": [Genre.COMEDY]},
            {"country": "JP", "language": Language.JA, "release_year": 2018, "genres": [Genre.ANIMATION]},
            {"country": "IN", "language": Language.HI, "release_year": 2017, "genres": [Genre.ROMANCE]},
            {"country": "KR", "language": Language.KO, "release_year": 2021, "genres": [Genre.THRILLER]},
            {"country": "BR", "language": Language.PT, "release_year": 1995, "genres": [Genre.CRIME]},
            {"country": "DE", "language": Language.DE, "release_year": 1985, "genres": [Genre.HISTORY]},
            {"country": "ES", "language": Language.ES, "release_year": 2005, "genres": [Genre.FANTASY]},
        ]

        movies = []
        for i, scenario in enumerate(diverse_scenarios):
            movie_data = {
                "movie_id": f"diverse_{i}",
                "title": f"Diverse Movie {i}",
                "synopsis": f"A culturally diverse test movie from {scenario['country']} that represents global cinema and international storytelling traditions.",
                "release_year": scenario["release_year"],
                "genres": scenario["genres"],
                "ratings": RatingInfo(average=7.5, count=5000),
                "metadata": MovieMetadata(original_language=scenario["language"], origin_country=scenario["country"]),
            }
            movies.append(MovieSchema(**movie_data))

        validator = ConstitutionalAIValidator()
        compliance_report = validator.validate_constitutional_compliance(movies)

        # Diverse dataset should have lower bias scores
        assert compliance_report["overall_bias_score"] < 0.5
        assert compliance_report["overall_compliance"] in ["PASS", "WARNING"]

        # Check diversity metrics
        metrics = compliance_report["constitutional_ai_metrics"]
        assert metrics["compliance_indicators"]["geographic_diversity"] >= 6  # At least 6 countries
        assert metrics["compliance_indicators"]["language_diversity"] >= 6  # At least 6 languages

    def test_performance_with_large_dataset(self):
        """Test performance with large number of movies."""
        import time

        # Create large dataset
        large_dataset = []
        for i in range(100):  # 100 movies for performance test
            movie_data = {
                "movie_id": f"perf_test_{i}",
                "title": f"Performance Test Movie {i}",
                "synopsis": f"Performance test movie number {i} created to validate system scalability and processing efficiency.",
                "release_year": 2000 + (i % 24),  # Spread across years 2000-2023
                "genres": [list(Genre)[i % len(Genre)]],  # Rotate through genres
                "ratings": RatingInfo(average=5.0 + (i % 5), count=1000 + i * 10),
                "metadata": MovieMetadata(
                    original_language=list(Language)[i % len(Language)],
                    origin_country=["US", "CA", "GB", "FR", "DE", "JP", "KR", "IN"][i % 8],
                ),
            }
            large_dataset.append(MovieSchema(**movie_data))

        validator = ConstitutionalAIValidator()

        start_time = time.time()
        compliance_report = validator.validate_constitutional_compliance(large_dataset)
        end_time = time.time()

        processing_time = end_time - start_time

        # Verify report was generated
        assert compliance_report["total_movies"] == 100
        assert "bias_analyses" in compliance_report

        # Performance should be reasonable (less than 10 seconds for 100 movies)
        assert processing_time < 10.0, f"Processing took too long: {processing_time:.2f} seconds"

        print(f"Processed {len(large_dataset)} movies in {processing_time:.2f} seconds")


class TestIntegrationWithValidation:
    """Test integration between Constitutional AI and existing validation."""

    def test_validation_pipeline_integration(self):
        """Test integration with existing validation pipeline."""
        # This test would integrate with the existing validation.py
        # to ensure Constitutional AI metrics are properly incorporated

        # Create test movie with constitutional AI considerations
        movie_data = {
            "movie_id": "integration_test",
            "title": "Integration Test Movie",
            "synopsis": "A comprehensive test movie for validating the integration between Constitutional AI metrics and existing validation systems.",
            "release_year": 2023,
            "genres": [Genre.DRAMA],
            "ratings": RatingInfo(average=8.0, count=10000),
            "metadata": MovieMetadata(tmdb_id=12345, original_language=Language.EN, origin_country="US"),
        }

        movie = MovieSchema(**movie_data)

        # Verify Constitutional AI metrics are initialized
        assert movie.constitutional_ai is not None
        assert hasattr(movie.constitutional_ai, "genre_diversity_score")
        assert hasattr(movie.constitutional_ai, "bias_flags")
        assert hasattr(movie.constitutional_ai, "overall_bias_score")

    def test_schema_evolution_compatibility(self):
        """Test compatibility with schema evolution."""
        # Test that Constitutional AI features work with different schema versions

        movie_data = {
            "movie_id": "evolution_test",
            "title": "Schema Evolution Test",
            "synopsis": "Testing Constitutional AI compatibility across different schema versions and evolution paths.",
            "release_year": 2023,
            "genres": [Genre.SCIENCE_FICTION],
            "ratings": RatingInfo(average=7.5, count=3000),
        }

        movie = MovieSchema(**movie_data)

        # Verify schema version is tracked
        assert movie.schema_version.value == "1.1"

        # Verify Constitutional AI metrics are present regardless of schema version
        assert movie.constitutional_ai is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
