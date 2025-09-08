"""
Comprehensive test suite for movie schema validation.

Tests all Pydantic models, custom validators, cross-field validation,
Constitutional AI metrics validation, performance with large datasets,
and TMDB API response mapping functionality.
"""

from datetime import datetime
from typing import Any, Dict, List

import pytest

from src.movie_schema import (
    CastMember,
    ConstitutionalAIMetrics,
    ContentRating,
    CrewMember,
    Genre,
    Language,
    MovieMetadata,
    MovieSchema,
    ProductionStatus,
    RatingInfo,
    create_movie_from_tmdb,
    get_enhanced_schema_info,
    map_tmdb_credits_response,
    map_tmdb_movie_response,
)


class TestMovieSchema:
    """Test suite for main MovieSchema model."""

    def test_valid_movie_creation(self):
        """Test creating a valid movie with all required fields."""
        movie_data = {
            "movie_id": "test_001",
            "title": "Test Movie",
            "synopsis": "This is a comprehensive test synopsis that meets the minimum length requirement of fifty characters and provides adequate detail.",
            "release_year": 2023,
            "genres": [Genre.DRAMA, Genre.THRILLER],
            "ratings": {"average": 7.5, "count": 10000},
        }

        movie = MovieSchema(**movie_data)

        assert movie.movie_id == "test_001"
        assert movie.title == "Test Movie"
        assert movie.release_year == 2023
        assert len(movie.genres) == 2
        assert movie.ratings.average == 7.5
        assert movie.quality_score > 0.0
        assert movie.schema_version.value == "1.1"

    def test_title_validation(self):
        """Test title field validation."""
        base_data = self._get_base_movie_data()

        # Valid titles
        valid_titles = ["The Godfather", "Pulp Fiction", "2001: A Space Odyssey"]
        for title in valid_titles:
            base_data["title"] = title
            movie = MovieSchema(**base_data)
            assert movie.title == title

        # Invalid titles
        with pytest.raises(ValueError, match="Title cannot be empty"):
            base_data["title"] = ""
            MovieSchema(**base_data)

        with pytest.raises(ValueError, match="Invalid placeholder title"):
            base_data["title"] = "Untitled"
            MovieSchema(**base_data)

    def test_synopsis_validation(self):
        """Test synopsis content validation."""
        base_data = self._get_base_movie_data()

        # Valid synopsis
        valid_synopsis = "A compelling story about human nature, exploring deep themes of redemption and hope through the journey of two unlikely friends."
        base_data["synopsis"] = valid_synopsis
        movie = MovieSchema(**base_data)
        assert len(movie.synopsis) >= 50

        # Too short
        with pytest.raises(ValueError, match="Synopsis must be at least 50 characters"):
            base_data["synopsis"] = "Too short"
            MovieSchema(**base_data)

        # Placeholder text
        with pytest.raises(ValueError, match="contains placeholder text"):
            base_data["synopsis"] = "No overview found for this movie"
            MovieSchema(**base_data)

    def test_genre_validation(self):
        """Test genre list validation."""
        base_data = self._get_base_movie_data()

        # Valid genres
        base_data["genres"] = [Genre.ACTION, Genre.COMEDY, Genre.DRAMA]
        movie = MovieSchema(**base_data)
        assert len(movie.genres) == 3

        # Duplicate removal
        base_data["genres"] = [Genre.ACTION, Genre.ACTION, Genre.COMEDY]
        movie = MovieSchema(**base_data)
        assert len(movie.genres) == 2
        assert movie.genres == [Genre.ACTION, Genre.COMEDY]

        # No genres
        with pytest.raises(ValueError, match="At least one genre is required"):
            base_data["genres"] = []
            MovieSchema(**base_data)

    def test_release_date_validation(self):
        """Test release date format and consistency validation."""
        base_data = self._get_base_movie_data()

        # Valid date
        base_data["release_date"] = "2023-06-15"
        base_data["release_year"] = 2023
        movie = MovieSchema(**base_data)
        assert movie.release_date == "2023-06-15"

        # Invalid format
        with pytest.raises(ValueError, match="Invalid release date format"):
            base_data["release_date"] = "2023/06/15"
            MovieSchema(**base_data)

        # Inconsistent with release year
        with pytest.raises(ValueError, match="inconsistent with release year"):
            base_data["release_date"] = "2025-06-15"
            base_data["release_year"] = 2023
            MovieSchema(**base_data)

    def test_quality_score_calculation(self):
        """Test automatic quality score calculation."""
        # High quality movie
        high_quality_data = {
            "movie_id": "hq_001",
            "title": "High Quality Film",
            "synopsis": "This is a very detailed synopsis that provides extensive information about the plot, characters, themes, and artistic vision of this exceptional cinematic work.",
            "release_year": 2023,
            "genres": [Genre.DRAMA],
            "cast": [
                {"name": "Actor One", "character": "Main Character", "order": 0},
                {"name": "Actor Two", "character": "Supporting Role", "order": 1},
            ],
            "ratings": {"average": 8.5, "count": 50000},
            "metadata": {
                "tmdb_id": 12345,
                "imdb_id": "tt1234567",
                "budget": 10000000,
                "revenue": 50000000,
                "runtime": 120,
            },
        }

        movie = MovieSchema(**high_quality_data)
        assert movie.quality_score > 0.8

        # Low quality movie
        low_quality_data = self._get_base_movie_data()
        low_quality_movie = MovieSchema(**low_quality_data)
        assert low_quality_movie.quality_score < movie.quality_score

    def test_cross_field_validation(self):
        """Test validation across related fields."""
        base_data = self._get_base_movie_data()

        # Add cast member with birth year validation
        base_data["cast"] = [{"name": "John Actor", "character": "Hero", "birth_year": 1980, "order": 0}]
        base_data["release_year"] = 2000

        movie = MovieSchema(**base_data)
        assert len(movie.cast) == 1

        # Invalid birth year (after movie release)
        with pytest.raises(ValueError, match="birth year.*is after movie release"):
            base_data["cast"][0]["birth_year"] = 2010
            MovieSchema(**base_data)

    def _get_base_movie_data(self) -> Dict[str, Any]:
        """Get base movie data for testing."""
        return {
            "movie_id": "test_base",
            "title": "Test Movie",
            "synopsis": "This is a test synopsis that meets the minimum character requirement for validation purposes.",
            "release_year": 2023,
            "genres": [Genre.DRAMA],
            "ratings": {"average": 7.0, "count": 1000},
        }


class TestCastMember:
    """Test suite for CastMember model."""

    def test_valid_cast_member(self):
        """Test creating valid cast member."""
        cast_data = {
            "name": "Jane Actor",
            "character": "Protagonist",
            "order": 0,
            "birth_year": 1985,
            "gender": "female",
            "tmdb_person_id": 12345,
        }

        cast_member = CastMember(**cast_data)
        assert cast_member.name == "Jane Actor"
        assert cast_member.character == "Protagonist"
        assert cast_member.birth_year == 1985

    def test_name_validation(self):
        """Test name field validation."""
        base_data = {"name": "Valid Name", "order": 0}

        # Valid names
        valid_names = ["John Smith", "María García", "李小明", "Jean-Luc Picard"]
        for name in valid_names:
            base_data["name"] = name
            member = CastMember(**base_data)
            assert member.name == name

        # Invalid names
        with pytest.raises(ValueError, match="Name must be at least 2 characters"):
            CastMember(name="A", order=0)

        with pytest.raises(ValueError, match="Name contains invalid characters"):
            CastMember(name="Test123", order=0)

    def test_birth_year_validation(self):
        """Test birth year validation."""
        current_year = datetime.now().year

        # Valid birth year
        member = CastMember(name="Actor Name", birth_year=1990, order=0)
        assert member.birth_year == 1990

        # Future birth year
        with pytest.raises(ValueError, match="Birth year cannot be in the future"):
            CastMember(name="Actor Name", birth_year=current_year + 1, order=0)


class TestCrewMember:
    """Test suite for CrewMember model."""

    def test_valid_crew_member(self):
        """Test creating valid crew member."""
        crew_data = {"name": "John Director", "job": "Director", "department": "Directing", "birth_year": 1975}

        crew_member = CrewMember(**crew_data)
        assert crew_member.name == "John Director"
        assert crew_member.job == "Director"
        assert crew_member.department == "Directing"

    def test_job_standardization(self):
        """Test job title standardization."""
        # Test job mapping
        job_mappings = {
            "director": "Director",
            "screenplay": "Writer",
            "executive producer": "Producer",
            "director of photography": "Cinematographer",
            "film editor": "Editor",
        }

        for input_job, expected_job in job_mappings.items():
            crew = CrewMember(name="Test Person", job=input_job, department="Test")
            assert crew.job == expected_job

    def test_department_validation(self):
        """Test department validation and mapping."""
        crew = CrewMember(name="Test Person", job="Director", department="Director")
        assert crew.department == "Directing"  # Should be mapped

        crew = CrewMember(name="Test Person", job="Sound Designer", department="Audio")
        assert crew.department == "Crew"  # Invalid department mapped to Crew


class TestRatingInfo:
    """Test suite for RatingInfo model."""

    def test_valid_rating_info(self):
        """Test creating valid rating information."""
        rating_data = {
            "average": 8.5,
            "count": 10000,
            "tmdb_rating": 8.4,
            "imdb_rating": 8.6,
            "distribution": {"8": 3000, "9": 4000, "7": 2000},
        }

        rating = RatingInfo(**rating_data)
        assert rating.average == 8.5
        assert rating.count == 10000
        assert rating.tmdb_rating == 8.4

    def test_distribution_validation(self):
        """Test rating distribution validation."""
        # Valid distribution
        rating = RatingInfo(average=8.0, count=1000, distribution={"8": 500, "9": 300, "7": 200})
        assert sum(rating.distribution.values()) == 1000

        # Invalid distribution (exceeds count)
        with pytest.raises(ValueError, match="Distribution total cannot exceed"):
            RatingInfo(average=8.0, count=500, distribution={"8": 300, "9": 400, "7": 200})  # Sums to 900

    def test_rating_range_validation(self):
        """Test rating value range validation."""
        # Valid ratings
        rating = RatingInfo(average=5.5, count=100)
        assert 0.0 <= rating.average <= 10.0

        # Invalid ratings
        with pytest.raises(ValueError):
            RatingInfo(average=-1.0, count=100)

        with pytest.raises(ValueError):
            RatingInfo(average=11.0, count=100)


class TestMovieMetadata:
    """Test suite for MovieMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid metadata."""
        metadata_data = {
            "tmdb_id": 12345,
            "imdb_id": "tt1234567",
            "original_language": Language.EN,
            "origin_country": "US",
            "budget": 50000000,
            "revenue": 150000000,
            "runtime": 120,
            "production_companies": ["Warner Bros", "Universal"],
        }

        metadata = MovieMetadata(**metadata_data)
        assert metadata.tmdb_id == 12345
        assert metadata.budget == 50000000
        assert len(metadata.production_companies) == 2

    def test_string_list_validation(self):
        """Test validation of string list fields."""
        # Test cleaning and deduplication
        metadata = MovieMetadata(production_companies=["  Disney  ", "Warner Bros", "", "Disney", "Universal"])

        # Should remove empty strings, strip whitespace, remove duplicates
        assert len(metadata.production_companies) == 3
        assert "Disney" in metadata.production_companies
        assert "Warner Bros" in metadata.production_companies
        assert "Universal" in metadata.production_companies
        assert "" not in metadata.production_companies

    def test_business_logic_validation(self):
        """Test business and financial data validation."""
        # Reasonable budget/revenue ratio
        metadata = MovieMetadata(budget=10000000, revenue=50000000)
        assert metadata.budget < metadata.revenue

        # The validator doesn't raise errors for unusual ratios, just logs warnings
        # Test that extreme ratios are handled gracefully
        extreme_metadata = MovieMetadata(budget=1000000, revenue=100000000)  # 100x ratio
        assert extreme_metadata.budget > 0
        assert extreme_metadata.revenue > 0


class TestConstitutionalAIMetrics:
    """Test suite for ConstitutionalAIMetrics model."""

    def test_valid_constitutional_metrics(self):
        """Test creating valid Constitutional AI metrics."""
        metrics_data = {
            "genre_diversity_score": 0.75,
            "geographic_diversity_score": 0.8,
            "language_diversity_score": 0.6,
            "temporal_diversity_score": 0.7,
            "bias_flags": ["geographic_concentration", "language_bias"],
            "overall_bias_score": 0.3,
        }

        metrics = ConstitutionalAIMetrics(**metrics_data)
        assert metrics.genre_diversity_score == 0.75
        assert len(metrics.bias_flags) == 2
        assert metrics.overall_bias_score == 0.3

    def test_diversity_score_validation(self):
        """Test diversity score range validation."""
        # Valid scores (0.0 to 1.0)
        metrics = ConstitutionalAIMetrics(
            genre_diversity_score=0.5, geographic_diversity_score=1.0, language_diversity_score=0.0
        )
        assert 0.0 <= metrics.genre_diversity_score <= 1.0

        # Invalid scores
        with pytest.raises(ValueError):
            ConstitutionalAIMetrics(genre_diversity_score=-0.1)

        with pytest.raises(ValueError):
            ConstitutionalAIMetrics(geographic_diversity_score=1.1)

    def test_diversity_consistency_validation(self):
        """Test internal consistency of diversity metrics."""
        # Test automatic bias score calculation
        metrics = ConstitutionalAIMetrics(
            genre_diversity_score=0.8,
            geographic_diversity_score=0.9,
            language_diversity_score=0.7,
            temporal_diversity_score=0.6,
        )

        # Should automatically calculate overall_bias_score
        expected_avg_diversity = (0.8 + 0.9 + 0.7 + 0.6) / 4  # 0.75
        expected_bias = 1.0 - expected_avg_diversity  # 0.25
        assert abs(metrics.overall_bias_score - expected_bias) < 0.01


class TestTMDBIntegration:
    """Test suite for TMDB API response mapping."""

    def test_map_tmdb_movie_response(self):
        """Test mapping TMDB movie API response."""
        tmdb_response = {
            "id": 550,
            "title": "Fight Club",
            "original_title": "Fight Club",
            "overview": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
            "release_date": "1999-10-15",
            "vote_average": 8.4,
            "vote_count": 26000,
            "genres": [{"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}],
            "original_language": "en",
            "origin_country": ["US"],
            "budget": 63000000,
            "revenue": 100853753,
            "runtime": 139,
            "production_companies": [{"name": "20th Century Fox"}, {"name": "Regency Enterprises"}],
        }

        mapped_data = map_tmdb_movie_response(tmdb_response)

        assert mapped_data["movie_id"] == "550"
        assert mapped_data["title"] == "Fight Club"
        assert mapped_data["release_year"] == 1999
        assert mapped_data["genres"] == [Genre.DRAMA, Genre.THRILLER]
        assert mapped_data["ratings"]["average"] == 8.4
        assert mapped_data["metadata"]["budget"] == 63000000

    def test_map_tmdb_credits_response(self):
        """Test mapping TMDB credits API response."""
        credits_response = {
            "cast": [
                {
                    "id": 819,
                    "name": "Edward Norton",
                    "character": "The Narrator",
                    "order": 0,
                    "gender": 2,
                    "known_for_department": "Acting",
                    "popularity": 15.5,
                }
            ],
            "crew": [
                {
                    "id": 7467,
                    "name": "David Fincher",
                    "job": "Director",
                    "department": "Directing",
                    "gender": 2,
                    "popularity": 10.2,
                }
            ],
        }

        mapped_credits = map_tmdb_credits_response(credits_response)

        assert len(mapped_credits["cast"]) == 1
        assert mapped_credits["cast"][0]["name"] == "Edward Norton"
        assert mapped_credits["cast"][0]["character"] == "The Narrator"
        assert mapped_credits["cast"][0]["gender"] == "male"

        assert len(mapped_credits["crew"]) == 1
        assert mapped_credits["crew"][0]["name"] == "David Fincher"
        assert mapped_credits["crew"][0]["job"] == "Director"

    def test_create_movie_from_tmdb(self):
        """Test creating MovieSchema from TMDB data."""
        tmdb_movie_data = {
            "id": 550,
            "title": "Fight Club",
            "overview": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
            "release_date": "1999-10-15",
            "vote_average": 8.4,
            "vote_count": 26000,
            "genres": [{"id": 18, "name": "Drama"}],
            "original_language": "en",
            "origin_country": ["US"],
            "budget": 63000000,
            "runtime": 139,
        }

        credits_data = {
            "cast": [{"id": 819, "name": "Edward Norton", "character": "The Narrator", "order": 0, "gender": 2}]
        }

        movie = create_movie_from_tmdb(tmdb_movie_data, credits_data)

        assert isinstance(movie, MovieSchema)
        assert movie.movie_id == "550"
        assert movie.title == "Fight Club"
        assert len(movie.cast) == 1
        assert movie.data_source == "tmdb_api"


class TestSchemaUtilities:
    """Test suite for schema utility functions."""

    def test_get_enhanced_schema_info(self):
        """Test schema information retrieval."""
        schema_info = get_enhanced_schema_info()

        assert "schema_version" in schema_info
        assert "supported_genres" in schema_info
        assert "supported_languages" in schema_info
        assert "constitutional_ai_fields" in schema_info

        assert len(schema_info["supported_genres"]) > 15
        assert len(schema_info["supported_languages"]) > 20
        assert "validation_features" in schema_info


class TestPerformanceAndEdgeCases:
    """Test suite for performance and edge cases."""

    def test_large_cast_crew(self):
        """Test handling of large cast and crew lists."""
        base_data = {
            "movie_id": "large_test",
            "title": "Epic Movie",
            "synopsis": "A movie with an enormous cast that tests the limits of our validation system.",
            "release_year": 2023,
            "genres": [Genre.DRAMA],
            "ratings": {"average": 7.0, "count": 1000},
        }

        # Create large cast (within limits)
        large_cast = [
            {"name": f"Actor {i}", "character": f"Character {i}", "order": i} for i in range(50)  # Max allowed
        ]

        base_data["cast"] = large_cast
        movie = MovieSchema(**base_data)
        assert len(movie.cast) == 50

        # Test exceeding limits
        with pytest.raises(ValueError, match="Cast size seems unreasonably large"):
            base_data["cast"] = [
                {"name": f"Actor {i}", "character": f"Character {i}", "order": i} for i in range(101)  # Exceeds limit
            ]
            MovieSchema(**base_data)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        movie_data = {
            "movie_id": "unicode_test",
            "title": "电影测试 & Películas Ñoñas: 2023!",
            "synopsis": "这是一个测试电影简介，包含中文字符和特殊符号，用于验证系统对国际化内容的支持能力。También incluye español con acentos.",
            "release_year": 2023,
            "genres": [Genre.DRAMA],
            "ratings": {"average": 8.0, "count": 5000},
            "cast": [
                {"name": "李小明", "character": "主角", "order": 0},
                {"name": "María José García-López", "character": "Supporting Role", "order": 1},
            ],
        }

        movie = MovieSchema(**movie_data)
        assert "电影测试" in movie.title
        assert "中文字符" in movie.synopsis
        assert movie.cast[0].name == "李小明"
        assert movie.cast[1].name == "María José García-López"

    def test_empty_and_none_values(self):
        """Test handling of empty and None values."""
        base_data = {
            "movie_id": "empty_test",
            "title": "Empty Values Test",
            "synopsis": "Testing how the schema handles empty and None values in optional fields.",
            "release_year": 2023,
            "genres": [Genre.DRAMA],
            "ratings": {"average": 7.0, "count": 1000},
        }

        # Test with None/empty optional fields
        movie = MovieSchema(**base_data)
        assert movie.cast == []  # Default empty list
        assert movie.crew == []  # Default empty list
        assert movie.metadata is not None  # Default metadata created
        assert movie.constitutional_ai is not None  # Default metrics created

    def test_boundary_values(self):
        """Test boundary values for numeric fields."""
        # Test minimum values
        movie_data = {
            "movie_id": "boundary_test",
            "title": "B",  # Minimum length
            "synopsis": "A" * 50,  # Minimum synopsis length
            "release_year": 1895,  # Minimum year
            "genres": [Genre.OTHER],  # Minimum genres
            "ratings": {"average": 0.0, "count": 0},  # Minimum ratings
        }

        movie = MovieSchema(**movie_data)
        assert movie.title == "B"
        assert movie.release_year == 1895
        assert movie.ratings.average == 0.0

        # Test maximum values
        movie_data.update(
            {
                "title": "A" * 500,  # Maximum title length
                "release_year": 2030,  # Maximum year
                "genres": [
                    Genre.ACTION,
                    Genre.ADVENTURE,
                    Genre.ANIMATION,
                    Genre.COMEDY,
                    Genre.CRIME,
                    Genre.DOCUMENTARY,
                    Genre.DRAMA,
                    Genre.FAMILY,
                    Genre.FANTASY,
                    Genre.HISTORY,
                ],  # Maximum genres
                "ratings": {"average": 10.0, "count": 999999999},  # Maximum ratings
            }
        )

        movie = MovieSchema(**movie_data)
        assert len(movie.title) == 500
        assert movie.release_year == 2030
        assert len(movie.genres) == 10
        assert movie.ratings.average == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
