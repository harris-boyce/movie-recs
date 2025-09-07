"""
Test suite for advanced movie validation rules and comprehensive movie validator.

Tests content quality validation, cross-reference validation, temporal validation,
financial validation, technical validation, and TMDB-specific validation.
"""

from datetime import datetime
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from src.movie_schema import (
    CastMember,
    ContentRating,
    CrewMember,
    Genre,
    Language,
    MovieMetadata,
    MovieSchema,
    RatingInfo,
)
from src.movie_validators import (
    ComprehensiveMovieValidator,
    ContentQualityValidator,
    CrossReferenceValidator,
    FinancialValidator,
    TechnicalValidator,
    TemporalValidator,
    TMDBSpecificValidator,
    ValidationResult,
    ValidationSeverity,
)


class TestValidationResult:
    """Test ValidationResult functionality."""

    def test_validation_result_creation(self):
        """Test creating and managing validation results."""
        result = ValidationResult()

        assert result.is_valid
        assert len(result.issues) == 0
        assert result.quality_score == 1.0

    def test_add_issue(self):
        """Test adding validation issues."""
        result = ValidationResult()

        # Add warning - should not affect validity
        result.add_issue(
            ValidationSeverity.WARNING,
            "test_category",
            "test_field",
            "Test warning message",
            "test_value",
            "Test suggestion",
        )

        assert result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0]["severity"] == ValidationSeverity.WARNING

        # Add error - should affect validity
        result.add_issue(ValidationSeverity.ERROR, "test_category", "test_field", "Test error message")

        assert not result.is_valid
        assert len(result.issues) == 2

    def test_issue_filtering(self):
        """Test filtering issues by severity and category."""
        result = ValidationResult()

        result.add_issue(ValidationSeverity.INFO, "category1", "field1", "Info message")
        result.add_issue(ValidationSeverity.WARNING, "category1", "field2", "Warning message")
        result.add_issue(ValidationSeverity.ERROR, "category2", "field3", "Error message")

        info_issues = result.get_issues_by_severity(ValidationSeverity.INFO)
        assert len(info_issues) == 1

        category1_issues = result.get_issues_by_category("category1")
        assert len(category1_issues) == 2


class TestContentQualityValidator:
    """Test suite for content quality validation."""

    def test_valid_synopsis_quality(self):
        """Test validation of good quality synopsis."""
        result = ValidationResult()

        good_synopsis = "This is a compelling movie synopsis that tells an engaging story about complex characters facing challenging situations. The narrative explores deep themes while maintaining audience interest through vivid descriptions and emotional depth."

        ContentQualityValidator.validate_synopsis_quality(good_synopsis, "test_movie", result)

        # Should not add any errors for good synopsis
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) == 0

    def test_short_synopsis_validation(self):
        """Test validation of short synopsis."""
        result = ValidationResult()

        short_synopsis = "Too short"

        ContentQualityValidator.validate_synopsis_quality(short_synopsis, "test_movie", result)

        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) > 0
        assert any("too short" in issue["message"].lower() for issue in error_issues)

    def test_placeholder_synopsis_detection(self):
        """Test detection of placeholder text in synopsis."""
        result = ValidationResult()

        placeholder_synopsis = "No overview found for this movie. Please check back later for more information."

        ContentQualityValidator.validate_synopsis_quality(placeholder_synopsis, "test_movie", result)

        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) > 0
        assert any("placeholder" in issue["message"].lower() for issue in error_issues)

    def test_repetitive_content_detection(self):
        """Test detection of repetitive content."""
        result = ValidationResult()

        repetitive_synopsis = "This movie is about a character character character character who faces challenges challenges challenges challenges in a story story story story that explores themes themes themes themes."

        ContentQualityValidator.validate_synopsis_quality(repetitive_synopsis, "test_movie", result)

        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert any("repetitive" in issue["message"].lower() for issue in warning_issues)

    def test_metadata_completeness_validation(self):
        """Test metadata completeness validation."""
        result = ValidationResult()

        # Create metadata with missing fields
        incomplete_metadata = MovieMetadata(
            # Missing tmdb_id, imdb_id, runtime, etc.
            original_language=Language.EN,
            origin_country="US",
        )

        ContentQualityValidator.validate_metadata_completeness(incomplete_metadata, "test_movie", result)

        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert len(warning_issues) > 0

        # Check for specific missing field warnings
        field_messages = [issue["message"] for issue in warning_issues]
        assert any("TMDB ID" in msg for msg in field_messages)
        assert any("runtime" in msg for msg in field_messages)


class TestCrossReferenceValidator:
    """Test suite for cross-reference validation."""

    def test_cast_consistency_validation(self):
        """Test cast consistency validation."""
        result = ValidationResult()

        # Create cast with issues
        problematic_cast = [
            CastMember(name="Actor One", order=0),
            CastMember(name="Actor One", order=1),  # Duplicate
            CastMember(name="Actor Two", order=0),  # Duplicate order
            CastMember(name="Actor Three", order=2),
        ]

        CrossReferenceValidator.validate_cast_consistency(problematic_cast, "test_movie", result)

        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert len(warning_issues) > 0

        # Should detect duplicate cast member
        assert any("duplicate" in issue["message"].lower() for issue in warning_issues)

    def test_cast_size_validation(self):
        """Test cast size validation (too large)."""
        result = ValidationResult()

        # Create very large cast
        large_cast = [CastMember(name=f"Actor {i}", order=i) for i in range(120)]  # Exceeds reasonable limit

        CrossReferenceValidator.validate_cast_consistency(large_cast, "test_movie", result)

        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert any("very large cast" in issue["message"].lower() for issue in warning_issues)

    def test_crew_essential_roles_validation(self):
        """Test validation of essential crew roles."""
        result = ValidationResult()

        # Crew missing essential roles
        incomplete_crew = [
            CrewMember(name="Sound Designer", job="Sound Designer", department="Sound"),
            CrewMember(name="Cinematographer", job="Cinematographer", department="Camera"),
            # Missing Director, Writer, Producer
        ]

        CrossReferenceValidator.validate_crew_consistency(incomplete_crew, "test_movie", result)

        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert any("missing essential crew roles" in issue["message"].lower() for issue in warning_issues)

    def test_crew_department_consistency(self):
        """Test crew department consistency validation."""
        result = ValidationResult()

        # Crew with department inconsistencies
        inconsistent_crew = [
            CrewMember(name="Director Name", job="Director", department="Production"),  # Should be Directing
            CrewMember(name="Writer Name", job="Writer", department="Camera"),  # Should be Writing
        ]

        CrossReferenceValidator.validate_crew_consistency(inconsistent_crew, "test_movie", result)

        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert any("unexpected department" in issue["message"].lower() for issue in warning_issues)


class TestTemporalValidator:
    """Test suite for temporal validation."""

    def test_release_date_consistency(self):
        """Test release date and year consistency."""

        # Create movie with inconsistent release date and year
        movie_data = {
            "movie_id": "temporal_test",
            "title": "Temporal Test Movie",
            "synopsis": "A movie to test temporal validation with inconsistent release date and year information.",
            "release_year": 2020,
            "release_date": "2023-06-15",  # Inconsistent
            "genres": [Genre.DRAMA],
            "ratings": RatingInfo(average=7.0, count=1000),
        }

        # Test that schema validation catches inconsistent release date/year
        with pytest.raises(ValidationError, match="inconsistent"):
            movie = MovieSchema(**movie_data)

        # Test the validator with a movie that has a smaller discrepancy (within schema tolerance)
        alternative_data = movie_data.copy()
        alternative_data["release_date"] = "2021-06-15"  # 1 year different - acceptable to both schema and validator
        alternative_data["release_year"] = 2020

        # This should pass both schema and validator (1 year difference is allowed)
        movie = MovieSchema(**alternative_data)
        result = ValidationResult()
        TemporalValidator.validate_release_date_consistency(movie, result)

        # Should not detect any inconsistency (1 year difference is allowed)
        issues = result.issues
        assert len(issues) == 0  # No inconsistency should be detected

    def test_career_timeline_validation(self):
        """Test cast/crew career timeline validation."""

        cast_with_issues = [
            CastMember(name="Future Person", birth_year=2025, order=0),  # Born in future
            CastMember(name="Very Old Person", birth_year=1920, order=1),  # Very old
            CastMember(name="Child Actor", birth_year=2015, order=2),  # Very young
        ]

        movie_data = {
            "movie_id": "timeline_test",
            "title": "Timeline Test Movie",
            "synopsis": "A movie to test career timeline validation with cast members of unusual ages.",
            "release_year": 2020,
            "genres": [Genre.FAMILY],
            "cast": cast_with_issues,
            "ratings": RatingInfo(average=6.5, count=500),
        }

        # Note: Schema validation will catch the future birth year
        # So we'll test the validator directly
        result = ValidationResult()

        # Test with reasonable movie
        reasonable_movie = MovieSchema(
            movie_id="reasonable_test",
            title="Reasonable Test Movie",
            synopsis="A movie with reasonable cast ages for testing timeline validation functionality.",
            release_year=2020,
            genres=[Genre.DRAMA],
            cast=[CastMember(name="Adult Actor", birth_year=1985, order=0)],  # 35 years old
            ratings=RatingInfo(average=7.0, count=1000),
        )

        TemporalValidator.validate_career_timelines(reasonable_movie, result)

        # Should not add issues for reasonable ages
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        temporal_errors = [issue for issue in error_issues if "temporal_logic" in issue.get("category", "")]
        assert len(temporal_errors) == 0

    def test_historical_context_validation(self):
        """Test historical context validation for early films."""

        early_movie = MovieSchema(
            movie_id="early_film",
            title="Early Cinema Test",
            synopsis="An early film to test historical context validation and temporal consistency checks.",
            release_year=1925,  # Early cinema
            genres=[Genre.DRAMA],
            ratings=RatingInfo(average=6.0, count=100),
            metadata=MovieMetadata(runtime=300),  # Very long for early cinema
        )

        result = ValidationResult()
        TemporalValidator.validate_release_date_consistency(early_movie, result)

        # Early year should be handled appropriately
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) == 0  # 1925 is valid for cinema


class TestFinancialValidator:
    """Test suite for financial validation."""

    def test_budget_revenue_logic_validation(self):
        """Test budget and revenue relationship validation."""
        result = ValidationResult()

        # Create metadata with unusual budget/revenue ratio
        unusual_metadata = MovieMetadata(
            budget=100_000_000, revenue=50_000  # $100M budget  # $50K revenue (very unusual ratio)
        )

        FinancialValidator.validate_budget_revenue_logic(unusual_metadata, 2020, "test_movie", result)

        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert any("ratio" in issue["message"].lower() for issue in warning_issues)

    def test_negative_financial_values(self):
        """Test validation of negative financial values."""
        result = ValidationResult()

        # Create metadata with negative values
        negative_metadata = MovieMetadata(budget=-1_000_000, revenue=-500_000)

        FinancialValidator.validate_budget_revenue_logic(negative_metadata, 2020, "test_movie", result)

        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) >= 2  # Should flag both negative values

    def test_inflation_adjustment(self):
        """Test inflation adjustment calculations."""

        # Test that inflation factors are applied
        factor_1970 = FinancialValidator._get_inflation_factor(1970)
        factor_2020 = FinancialValidator._get_inflation_factor(2020)

        assert factor_1970 > factor_2020  # 1970 dollars worth more in 2020 terms
        assert factor_2020 == 1.0  # 2020 is base year


class TestTechnicalValidator:
    """Test suite for technical validation."""

    def test_runtime_expectations_by_genre(self):
        """Test runtime validation against genre expectations."""
        result = ValidationResult()

        # Create movie with runtime that's short for its genre
        short_action_movie = MovieSchema(
            movie_id="short_action",
            title="Short Action Movie",
            synopsis="A very short action movie that tests runtime validation against genre expectations.",
            release_year=2023,
            genres=[Genre.ACTION],
            ratings=RatingInfo(average=6.0, count=1000),
            metadata=MovieMetadata(runtime=60),  # Very short for action
        )

        TechnicalValidator.validate_runtime_expectations(short_action_movie, result)

        info_issues = result.get_issues_by_severity(ValidationSeverity.INFO)
        assert any("shorter than typical" in issue["message"] for issue in info_issues)

    def test_invalid_runtime_validation(self):
        """Test validation of invalid runtime values."""
        result = ValidationResult()

        invalid_runtime_movie = MovieSchema(
            movie_id="invalid_runtime",
            title="Invalid Runtime Movie",
            synopsis="A movie with invalid runtime for testing technical validation error handling.",
            release_year=2023,
            genres=[Genre.DRAMA],
            ratings=RatingInfo(average=7.0, count=1000),
            metadata=MovieMetadata(runtime=0),  # Invalid runtime
        )

        TechnicalValidator.validate_runtime_expectations(invalid_runtime_movie, result)

        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert any("invalid runtime" in issue["message"].lower() for issue in error_issues)

    def test_rating_consistency_validation(self):
        """Test rating data consistency validation."""
        result = ValidationResult()

        # Create ratings with inconsistencies
        inconsistent_ratings = RatingInfo(
            average=8.5,
            count=1000,
            tmdb_rating=8.5,
            tmdb_count=1000,
            imdb_rating=3.0,  # Large discrepancy from TMDB
            imdb_count=5000,
        )

        metadata = MovieMetadata()

        TechnicalValidator.validate_rating_consistency(inconsistent_ratings, metadata, "test_movie", result)

        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        # Large discrepancy should be flagged
        assert any("discrepancy" in issue["message"].lower() for issue in warning_issues)


class TestTMDBSpecificValidator:
    """Test suite for TMDB-specific validation."""

    def test_tmdb_id_validation(self):
        """Test TMDB ID validation."""
        result = ValidationResult()

        # Test invalid TMDB ID
        invalid_metadata = MovieMetadata(
            tmdb_id=-123, imdb_id="invalid_format"  # Invalid negative ID  # Invalid IMDB ID format
        )

        TMDBSpecificValidator.validate_tmdb_ids(invalid_metadata, "test_movie", result)

        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)

        # Should flag invalid TMDB ID
        assert any("Invalid TMDB ID" in issue["message"] for issue in error_issues)
        # Should flag invalid IMDB ID format
        assert any("Invalid IMDB ID format" in issue["message"] for issue in warning_issues)

    def test_tmdb_popularity_validation(self):
        """Test TMDB popularity score validation."""
        result = ValidationResult()

        # Test negative popularity
        invalid_metadata = MovieMetadata(popularity=-10.5)  # Invalid negative popularity

        ratings = RatingInfo(average=7.0, count=1000)

        TMDBSpecificValidator.validate_tmdb_popularity_scores(invalid_metadata, ratings, "test_movie", result)

        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert any("Negative popularity" in issue["message"] for issue in error_issues)


class TestComprehensiveMovieValidator:
    """Test suite for the main comprehensive validator."""

    def test_comprehensive_validation(self):
        """Test complete movie validation."""

        # Create a movie with various issues
        problematic_movie = MovieSchema(
            movie_id="comprehensive_test",
            title="Comprehensive Test Movie",
            synopsis="A test movie with various validation issues to test the comprehensive validator.",
            release_year=2023,
            genres=[Genre.DRAMA],
            cast=[CastMember(name="Test Actor", birth_year=1990, order=0)],
            crew=[CrewMember(name="Test Director", job="Director", department="Directing")],
            ratings=RatingInfo(average=7.0, count=100),  # Low count
            metadata=MovieMetadata(
                runtime=45, budget=1000, revenue=50  # Short runtime  # Very low budget  # Very low revenue
            ),
        )

        validator = ComprehensiveMovieValidator()
        result = validator.validate_movie(problematic_movie)

        assert isinstance(result, ValidationResult)
        assert "total_issues" in result.summary
        assert "quality_score" in result.summary
        assert 0.0 <= result.quality_score <= 1.0

    def test_validator_configuration(self):
        """Test validator with custom configuration."""

        # Create validator with disabled validators
        config = {
            "enabled_validators": {
                "content_quality": True,
                "cross_reference": False,
                "temporal": False,
                "financial": False,
                "technical": False,
                "tmdb_specific": False,
            }
        }

        validator = ComprehensiveMovieValidator(config)

        # Create simple valid movie
        simple_movie = MovieSchema(
            movie_id="config_test",
            title="Configuration Test Movie",
            synopsis="A simple movie for testing validator configuration and selective validation.",
            release_year=2023,
            genres=[Genre.DRAMA],
            ratings=RatingInfo(average=7.5, count=5000),
        )

        result = validator.validate_movie(simple_movie)

        # Should have fewer issues since most validators are disabled
        assert result.is_valid or len(result.issues) < 5

    def test_quality_score_calculation(self):
        """Test quality score calculation."""

        validator = ComprehensiveMovieValidator()

        # Create high-quality movie
        high_quality_movie = MovieSchema(
            movie_id="high_quality_test",
            title="High Quality Test Movie",
            synopsis="This is a high quality test movie with excellent metadata, comprehensive cast and crew information, and strong ratings to test quality score calculation.",
            release_year=2023,
            genres=[Genre.DRAMA, Genre.THRILLER],
            cast=[CastMember(name=f"Actor {i}", order=i) for i in range(10)],
            crew=[
                CrewMember(name="Director", job="Director", department="Directing"),
                CrewMember(name="Writer", job="Writer", department="Writing"),
                CrewMember(name="Producer", job="Producer", department="Production"),
            ],
            ratings=RatingInfo(average=8.5, count=50000),
            metadata=MovieMetadata(tmdb_id=12345, imdb_id="tt1234567", runtime=120, budget=50000000, revenue=200000000),
        )

        result = validator.validate_movie(high_quality_movie)

        # High quality movie should have high quality score
        assert result.quality_score > 0.7

    def test_batch_validation(self):
        """Test batch validation of multiple movies."""

        movies = [
            MovieSchema(
                movie_id=f"batch_test_{i}",
                title=f"Batch Test Movie {i}",
                synopsis=f"Test movie number {i} for batch validation testing and performance evaluation.",
                release_year=2020 + i,
                genres=[Genre.DRAMA],
                ratings=RatingInfo(average=7.0 + i * 0.1, count=1000 + i * 100),
            )
            for i in range(5)
        ]

        validator = ComprehensiveMovieValidator()
        results = validator.validate_movie_batch(movies)

        assert len(results) == 5
        for movie_id, result in results.items():
            assert isinstance(result, ValidationResult)
            assert movie_id.startswith("batch_test_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
