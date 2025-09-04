"""
Data schema definitions and validation models for movie datasets.

This module defines the expected data structure for movie datasets using Pydantic models
with comprehensive validation rules, custom validators, and schema versioning support.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class SchemaVersion(str, Enum):
    """Supported schema versions."""

    V1_0 = "1.0"
    CURRENT = V1_0


class Language(str, Enum):
    """Common language codes."""

    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    JA = "ja"
    KO = "ko"
    ZH = "zh"
    OTHER = "other"


class Genre(str, Enum):
    """Standardized movie genres."""

    ACTION = "Action"
    ADVENTURE = "Adventure"
    ANIMATION = "Animation"
    COMEDY = "Comedy"
    CRIME = "Crime"
    DOCUMENTARY = "Documentary"
    DRAMA = "Drama"
    FAMILY = "Family"
    FANTASY = "Fantasy"
    HISTORY = "History"
    HORROR = "Horror"
    MUSIC = "Music"
    MYSTERY = "Mystery"
    ROMANCE = "Romance"
    SCIENCE_FICTION = "Science Fiction"
    THRILLER = "Thriller"
    WAR = "War"
    WESTERN = "Western"
    OTHER = "Other"


class PersonInfo(BaseModel):
    """Information about a person (actor, director, etc.)."""

    name: str = Field(..., min_length=1, max_length=200)
    character: Optional[str] = Field(None, max_length=200)
    role: str = Field(
        ...,
        pattern=r"^(actor|director|writer|producer|composer|cinematographer|editor)$",
    )
    birth_year: Optional[int] = Field(None, ge=1800, le=2025)
    gender: Optional[str] = Field(
        None, pattern=r"^(male|female|non-binary|other|unknown)$"
    )
    ethnicity: Optional[str] = Field(None, max_length=100)

    @field_validator("birth_year")
    def validate_birth_year(cls, v):
        if v and v > datetime.now().year:
            raise ValueError("Birth year cannot be in the future")
        return v


class Ratings(BaseModel):
    """Movie ratings information."""

    average: float = Field(
        ..., ge=0.0, le=10.0, description="Average rating (0-10 scale)"
    )
    count: int = Field(..., ge=0, description="Number of ratings")
    distribution: Optional[Dict[str, int]] = Field(
        None, description="Rating distribution by score"
    )

    @model_validator(mode="after")
    def validate_distribution(self):
        if self.distribution:
            total_distributed = sum(self.distribution.values())
            if total_distributed > self.count:
                raise ValueError("Distribution total cannot exceed rating count")
        return self


class Metadata(BaseModel):
    """Additional movie metadata."""

    language: Language = Field(Language.EN, description="Primary language")
    country: str = Field("US", max_length=2, pattern=r"^[A-Z]{2}$")
    budget: Optional[int] = Field(None, ge=0, description="Production budget in USD")
    revenue: Optional[int] = Field(None, ge=0, description="Box office revenue in USD")
    production_companies: Optional[List[str]] = Field(None, max_length=20)
    keywords: Optional[List[str]] = Field(None, max_length=50)
    imdb_id: Optional[str] = Field(None, pattern=r"^tt\d{7,8}$")
    tmdb_id: Optional[int] = Field(None, ge=1)

    @field_validator("production_companies", "keywords")
    def validate_string_lists(cls, v):
        if v:
            # Remove empty strings and duplicates
            cleaned = list(dict.fromkeys([item.strip() for item in v if item.strip()]))
            return cleaned if cleaned else None
        return v


class Movie(BaseModel):
    """Complete movie data model."""

    movie_id: str = Field(
        ..., min_length=1, max_length=50, description="Unique movie identifier"
    )
    title: str = Field(..., min_length=1, max_length=500, description="Movie title")
    synopsis: str = Field(
        ..., min_length=50, max_length=5000, description="Movie synopsis/plot"
    )
    release_year: int = Field(..., ge=1895, le=2030, description="Release year")
    runtime_mins: Optional[int] = Field(
        None, ge=1, le=500, description="Runtime in minutes"
    )
    genres: List[Genre] = Field(
        ..., min_length=1, max_length=10, description="Movie genres"
    )
    cast: Optional[List[PersonInfo]] = Field(
        None, max_length=50, description="Cast information"
    )
    crew: Optional[List[PersonInfo]] = Field(
        None, max_length=30, description="Crew information"
    )
    ratings: Ratings = Field(..., description="Ratings information")
    metadata: Optional[Metadata] = Field(None, description="Additional metadata")

    # Data quality tracking
    data_source: str = Field("unknown", description="Original data source")
    last_updated: datetime = Field(default_factory=datetime.now)
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Data quality score"
    )

    @field_validator("synopsis")
    def validate_synopsis(cls, v):
        # Remove excessive whitespace
        cleaned = " ".join(v.split())
        if len(cleaned) < 50:
            raise ValueError("Synopsis must be at least 50 characters after cleaning")
        return cleaned

    @field_validator("genres")
    def validate_genres(cls, v):
        # Remove duplicates while preserving order
        seen = set()
        unique_genres = []
        for genre in v:
            if genre not in seen:
                seen.add(genre)
                unique_genres.append(genre)
        return unique_genres

    @model_validator(mode="after")
    def validate_cast_crew(self):
        """Validate cast and crew information consistency."""
        cast = self.cast or []
        crew = self.crew or []

        # Check for reasonable cast/crew sizes
        if len(cast) > 100:
            raise ValueError("Cast size seems unreasonably large (>100)")
        if len(crew) > 50:
            raise ValueError("Crew size seems unreasonably large (>50)")

        # Validate person birth years against movie release year
        if self.release_year:
            for person in cast + crew:
                if person.birth_year and person.birth_year > self.release_year - 5:
                    # Allow 5-year buffer for child actors
                    if self.release_year - person.birth_year < 5:
                        continue
                    raise ValueError(
                        f"Person birth year {person.birth_year} inconsistent with release year {self.release_year}"
                    )

        return self

    @model_validator(mode="after")
    def validate_metadata_consistency(self):
        """Validate metadata consistency."""
        if self.metadata:
            # Check budget vs revenue logic
            if self.metadata.budget and self.metadata.revenue:
                # Revenue should typically be >= budget (allowing for losses)
                if (
                    self.metadata.revenue > 0
                    and self.metadata.budget > self.metadata.revenue * 10
                ):
                    # Flag if budget is more than 10x revenue (likely data error)
                    raise ValueError(
                        "Budget significantly exceeds revenue, possible data error"
                    )

        return self


class MovieCollection(BaseModel):
    """Collection of movies with metadata."""

    movies: List[Movie] = Field(..., min_length=1)
    schema_version: SchemaVersion = Field(SchemaVersion.CURRENT)
    collection_id: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.now)
    total_count: int = Field(..., ge=1)
    data_sources: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_total_count(self):
        if len(self.movies) != self.total_count:
            raise ValueError("Total count must match actual number of movies")
        return self

    @model_validator(mode="after")
    def validate_collection(self):
        """Validate collection-level constraints."""
        if self.movies:
            # Check for duplicate movie IDs
            movie_ids = [movie.movie_id for movie in self.movies]
            if len(movie_ids) != len(set(movie_ids)):
                raise ValueError("Duplicate movie IDs found in collection")

            # Collect data sources
            sources = set()
            for movie in self.movies:
                sources.add(movie.data_source)
            self.data_sources = list(sources)

        return self


class ValidationResult(BaseModel):
    """Result of data validation process."""

    is_valid: bool
    error_count: int = 0
    warning_count: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    validated_at: datetime = Field(default_factory=datetime.now)

    def add_error(self, movie_id: str, field: str, message: str, value: Any = None):
        """Add validation error."""
        self.errors.append(
            {
                "movie_id": movie_id,
                "field": field,
                "message": message,
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.error_count = len(self.errors)
        self.is_valid = self.error_count == 0

    def add_warning(self, movie_id: str, field: str, message: str, value: Any = None):
        """Add validation warning."""
        self.warnings.append(
            {
                "movie_id": movie_id,
                "field": field,
                "message": message,
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.warning_count = len(self.warnings)


class QualityThresholds(BaseModel):
    """Data quality thresholds for validation."""

    completeness_min: float = Field(0.95, ge=0.0, le=1.0)
    synopsis_min_length: int = Field(50, ge=10)
    title_min_length: int = Field(1, ge=1)
    valid_year_range: List[int] = Field([1895, 2030])
    max_genres_per_movie: int = Field(10, ge=1)
    min_rating_count: int = Field(1, ge=0)
    max_cast_size: int = Field(50, ge=1)
    max_crew_size: int = Field(30, ge=1)

    @field_validator("valid_year_range")
    def validate_year_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError(
                "Year range must be [start_year, end_year] with start < end"
            )
        return v


# Schema validation utilities
def get_schema_info() -> Dict[str, Any]:
    """Get information about the current schema."""
    return {
        "version": SchemaVersion.CURRENT,
        "supported_genres": [genre.value for genre in Genre],
        "supported_languages": [lang.value for lang in Language],
        "person_roles": [
            "actor",
            "director",
            "writer",
            "producer",
            "composer",
            "cinematographer",
            "editor",
        ],
        "validation_rules": {
            "title_max_length": 500,
            "synopsis_min_length": 50,
            "synopsis_max_length": 5000,
            "release_year_min": 1895,
            "release_year_max": 2030,
            "runtime_max": 500,
            "rating_range": [0.0, 10.0],
        },
    }


def create_movie_from_dict(data: Dict[str, Any]) -> Movie:
    """Create Movie instance from dictionary with error handling."""
    try:
        return Movie(**data)
    except Exception as e:
        # Add movie_id to error for better debugging
        movie_id = data.get("movie_id", "unknown")
        raise ValueError(f"Error creating movie {movie_id}: {str(e)}") from e


def validate_movie_dict(data: Dict[str, Any]) -> ValidationResult:
    """Validate movie dictionary without creating Movie instance."""
    result = ValidationResult(is_valid=True)
    movie_id = data.get("movie_id", "unknown")

    try:
        Movie(**data)
    except Exception as e:
        result.add_error(movie_id, "validation", str(e), data)

    return result


if __name__ == "__main__":
    # Example usage and schema information
    schema_info = get_schema_info()
    print("MovieRecs Data Schema Information:")
    print(f"Version: {schema_info['version']}")
    print(f"Supported Genres: {len(schema_info['supported_genres'])}")
    print(f"Supported Languages: {len(schema_info['supported_languages'])}")

    # Example movie creation
    example_movie = {
        "movie_id": "1",
        "title": "The Shawshank Redemption",
        "synopsis": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency. This powerful drama explores themes of hope, friendship, and the human spirit's resilience in the face of adversity.",
        "release_year": 1994,
        "runtime_mins": 142,
        "genres": [Genre.DRAMA],
        "ratings": {"average": 9.3, "count": 2500000},
        "data_source": "example",
    }

    try:
        movie = create_movie_from_dict(example_movie)
        print(f"\nSuccessfully created movie: {movie.title}")
    except ValueError as e:
        print(f"\nValidation error: {e}")
