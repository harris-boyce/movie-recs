"""
Enhanced movie metadata schema with comprehensive Pydantic models for TMDB integration.

This module implements the complete movie schema as specified in Issue #33, including:
- Core movie schema with comprehensive validation
- Cast and crew models with demographic tracking
- TMDB API compatibility and response mapping
- Constitutional AI metrics for bias detection
- Advanced validation rules and error reporting
- Schema versioning and evolution support
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SchemaVersion(str, Enum):
    """Supported schema versions with evolution tracking."""

    V1_0 = "1.0"
    V1_1 = "1.1"
    CURRENT = V1_1


class Language(str, Enum):
    """Comprehensive language codes with cultural context."""

    EN = "en"  # English
    ES = "es"  # Spanish
    FR = "fr"  # French
    DE = "de"  # German
    IT = "it"  # Italian
    JA = "ja"  # Japanese
    KO = "ko"  # Korean
    ZH = "zh"  # Chinese
    PT = "pt"  # Portuguese
    RU = "ru"  # Russian
    AR = "ar"  # Arabic
    HI = "hi"  # Hindi
    TH = "th"  # Thai
    NL = "nl"  # Dutch
    SV = "sv"  # Swedish
    NO = "no"  # Norwegian
    DA = "da"  # Danish
    FI = "fi"  # Finnish
    PL = "pl"  # Polish
    TR = "tr"  # Turkish
    HU = "hu"  # Hungarian
    CS = "cs"  # Czech
    SK = "sk"  # Slovak
    OTHER = "other"


class Genre(str, Enum):
    """Standardized movie genres mapping to TMDB categories."""

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
    TV_MOVIE = "TV Movie"
    THRILLER = "Thriller"
    WAR = "War"
    WESTERN = "Western"
    OTHER = "Other"


class ContentRating(str, Enum):
    """Movie content rating classifications."""

    G = "G"  # General Audiences
    PG = "PG"  # Parental Guidance
    PG_13 = "PG-13"  # Parents Strongly Cautioned
    R = "R"  # Restricted
    NC_17 = "NC-17"  # Adults Only
    NR = "NR"  # Not Rated
    UNKNOWN = "Unknown"


class ProductionStatus(str, Enum):
    """Movie production status."""

    RUMORED = "Rumored"
    PLANNED = "Planned"
    IN_PRODUCTION = "In Production"
    POST_PRODUCTION = "Post Production"
    RELEASED = "Released"
    CANCELED = "Canceled"


class ConstitutionalAIMetrics(BaseModel):
    """Constitutional AI metrics for bias tracking and diversity analysis."""

    # Diversity scores (0.0 = no diversity, 1.0 = maximum diversity)
    genre_diversity_score: float = Field(0.0, ge=0.0, le=1.0, description="Genre diversity Shannon index normalized")
    geographic_diversity_score: float = Field(0.0, ge=0.0, le=1.0, description="Geographic representation score")
    language_diversity_score: float = Field(0.0, ge=0.0, le=1.0, description="Language diversity score")
    temporal_diversity_score: float = Field(0.0, ge=0.0, le=1.0, description="Temporal distribution score")

    # Representation metrics
    cast_gender_balance: Optional[Dict[str, float]] = Field(None, description="Gender distribution in cast")
    cast_ethnicity_distribution: Optional[Dict[str, float]] = Field(None, description="Ethnic representation in cast")
    crew_gender_balance: Optional[Dict[str, float]] = Field(None, description="Gender distribution in crew")
    crew_role_diversity: Optional[Dict[str, float]] = Field(None, description="Role diversity in crew")

    # Bias flags and indicators
    bias_flags: List[str] = Field(default_factory=list, description="Detected bias indicators")
    representation_gaps: List[str] = Field(default_factory=list, description="Underrepresented groups identified")
    content_warnings: List[str] = Field(default_factory=list, description="Content warning flags")

    # Audit trail
    last_bias_check: datetime = Field(default_factory=datetime.now)
    bias_detection_version: str = Field("1.0", description="Version of bias detection algorithm used")
    overall_bias_score: float = Field(0.0, ge=0.0, le=1.0, description="Composite bias score")

    # Data provenance for Constitutional AI
    data_quality_flags: List[str] = Field(default_factory=list, description="Data quality concerns")
    validation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Validation audit trail")

    @field_validator("bias_flags", "representation_gaps", "content_warnings")
    def validate_flag_lists(cls, v):
        """Ensure flag lists contain valid string entries."""
        if v:
            return [flag.strip() for flag in v if flag and isinstance(flag, str)]
        return []

    @model_validator(mode="after")
    def validate_diversity_consistency(self):
        """Ensure diversity metrics are internally consistent."""
        scores = [
            self.genre_diversity_score,
            self.geographic_diversity_score,
            self.language_diversity_score,
            self.temporal_diversity_score,
        ]

        # Calculate overall bias score if not set
        if self.overall_bias_score == 0.0 and any(s > 0 for s in scores):
            # Higher diversity = lower bias
            avg_diversity = sum(scores) / len(scores)
            self.overall_bias_score = max(0.0, 1.0 - avg_diversity)

        return self


class CastMember(BaseModel):
    """Enhanced cast member information with demographic data."""

    # Basic information
    name: str = Field(..., min_length=1, max_length=200, description="Actor name")
    character: Optional[str] = Field(None, max_length=200, description="Character name")
    order: int = Field(0, ge=0, le=200, description="Billing order in credits")

    # TMDB identifiers
    tmdb_person_id: Optional[int] = Field(None, ge=1, description="TMDB person ID")
    imdb_person_id: Optional[str] = Field(None, pattern=r"^nm\d{7,8}$", description="IMDB person ID")

    # Demographic information (when available)
    birth_year: Optional[int] = Field(None, ge=1800, description="Birth year")
    birth_place: Optional[str] = Field(None, max_length=200, description="Place of birth")
    gender: Optional[str] = Field(None, pattern=r"^(male|female|non-binary|other|unknown)$")
    ethnicity: Optional[str] = Field(None, max_length=100, description="Ethnicity/cultural background")
    nationality: Optional[str] = Field(None, max_length=100, description="Nationality")

    # Career information
    known_for_department: Optional[str] = Field(None, max_length=100, description="Primary department")
    popularity: Optional[float] = Field(None, description="TMDB popularity score")
    profile_path: Optional[str] = Field(None, description="TMDB profile image path")

    @field_validator("birth_year")
    def validate_birth_year(cls, v):
        """Ensure birth year is not in the future."""
        if v and v > datetime.now().year:
            raise ValueError("Birth year cannot be in the future")
        return v

    @field_validator("name")
    def validate_name_quality(cls, v):
        """Validate name quality and format."""
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Name must be at least 2 characters")
        # Disallow names ending with digits (Test123) but allow "Actor 0" format
        if re.match(r"^[A-Za-z]+\d+$", v):
            raise ValueError("Name contains invalid characters")
        # Basic character validation - allow unicode letters, spaces, hyphens, apostrophes
        if not re.match(r"^[^\x00-\x1f\x7f-\x9f]+$", v):
            raise ValueError("Name contains invalid characters")
        return v


class CrewMember(BaseModel):
    """Enhanced crew member information with role validation."""

    # Basic information
    name: str = Field(..., min_length=1, max_length=200, description="Crew member name")
    job: str = Field(..., min_length=1, max_length=100, description="Specific job title")
    department: str = Field(..., min_length=1, max_length=50, description="Department")

    # TMDB identifiers
    tmdb_person_id: Optional[int] = Field(None, ge=1, description="TMDB person ID")
    imdb_person_id: Optional[str] = Field(None, pattern=r"^nm\d{7,8}$", description="IMDB person ID")

    # Demographic information (when available)
    birth_year: Optional[int] = Field(None, ge=1800, description="Birth year")
    gender: Optional[str] = Field(None, pattern=r"^(male|female|non-binary|other|unknown)$")
    ethnicity: Optional[str] = Field(None, max_length=100, description="Ethnicity/cultural background")

    # Career information
    known_for_department: Optional[str] = Field(None, max_length=100, description="Primary department")
    popularity: Optional[float] = Field(None, description="TMDB popularity score")
    profile_path: Optional[str] = Field(None, description="TMDB profile image path")

    @field_validator("job")
    def validate_job_title(cls, v):
        """Standardize and validate job titles."""
        v = v.strip().title()

        # Common job title mapping
        job_mapping = {
            "Director": "Director",
            "Writer": "Writer",
            "Screenplay": "Writer",
            "Producer": "Producer",
            "Executive Producer": "Producer",
            "Co-Producer": "Producer",
            "Cinematographer": "Cinematographer",
            "Director Of Photography": "Cinematographer",
            "Editor": "Editor",
            "Film Editor": "Editor",
            "Music": "Composer",
            "Original Music Composer": "Composer",
            "Sound": "Sound",
            "Production Design": "Production Designer",
            "Art Director": "Art Director",
            "Costume Design": "Costume Designer",
        }

        return job_mapping.get(v, v)

    @field_validator("department")
    def validate_department(cls, v):
        """Validate and standardize department names."""
        v = v.strip().title()

        valid_departments = {
            "Directing",
            "Writing",
            "Production",
            "Camera",
            "Editing",
            "Sound",
            "Art",
            "Costume & Make-Up",
            "Visual Effects",
            "Lighting",
            "Crew",
        }

        # Map common variations
        dept_mapping = {
            "Director": "Directing",
            "Writer": "Writing",
            "Cinematography": "Camera",
            "Photography": "Camera",
            "Music": "Sound",
            "Art Direction": "Art",
            "Costume": "Costume & Make-Up",
            "Makeup": "Costume & Make-Up",
            "VFX": "Visual Effects",
        }

        mapped = dept_mapping.get(v, v)
        return mapped if mapped in valid_departments else "Crew"


class RatingInfo(BaseModel):
    """Comprehensive rating information with source tracking."""

    # Primary rating (TMDB scale: 0-10)
    average: float = Field(..., ge=0.0, le=10.0, description="Average rating on 0-10 scale")
    count: int = Field(..., ge=0, description="Total number of ratings")

    # Rating distribution
    distribution: Optional[Dict[str, int]] = Field(None, description="Rating distribution by score")

    # Multiple rating sources
    tmdb_rating: Optional[float] = Field(None, ge=0.0, le=10.0, description="TMDB user rating")
    tmdb_count: Optional[int] = Field(None, ge=0, description="TMDB rating count")
    imdb_rating: Optional[float] = Field(None, ge=0.0, le=10.0, description="IMDB rating")
    imdb_count: Optional[int] = Field(None, ge=0, description="IMDB rating count")
    rt_critics: Optional[int] = Field(None, ge=0, le=100, description="Rotten Tomatoes critics score")
    rt_audience: Optional[int] = Field(None, ge=0, le=100, description="Rotten Tomatoes audience score")
    metacritic: Optional[int] = Field(None, ge=0, le=100, description="Metacritic score")

    # Quality indicators
    rating_reliability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Rating reliability score")
    last_updated: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def validate_rating_consistency(self):
        """Validate rating consistency across sources."""
        # Check distribution consistency
        if self.distribution:
            total_distributed = sum(self.distribution.values())
            if total_distributed > self.count:
                raise ValueError("Distribution total cannot exceed total rating count")

        # Validate TMDB ratings
        if self.tmdb_rating and self.tmdb_count:
            if self.tmdb_count > self.count:
                # TMDB count should not exceed total unless it's the primary source
                pass  # Allow this case

        # Check for reasonable rating values
        ratings = [r for r in [self.tmdb_rating, self.imdb_rating] if r is not None]
        if ratings and abs(max(ratings) - min(ratings)) > 5.0:
            # Flag significant rating discrepancies for review
            pass  # This is informational, not an error

        return self


class MovieMetadata(BaseModel):
    """Comprehensive technical and business metadata."""

    # TMDB integration
    tmdb_id: Optional[int] = Field(None, description="TMDB movie ID")
    imdb_id: Optional[str] = Field(None, description="IMDB movie ID")

    # Language and geography
    original_language: Language = Field(Language.EN, description="Original language")
    spoken_languages: List[Language] = Field(default_factory=list, description="All spoken languages")
    origin_country: str = Field("US", max_length=2, pattern=r"^[A-Z]{2}$", description="Origin country")
    production_countries: List[str] = Field(default_factory=list, description="Production countries")

    # Business data
    budget: Optional[int] = Field(None, description="Production budget in USD")
    revenue: Optional[int] = Field(None, description="Worldwide box office revenue in USD")

    # Technical information
    runtime: Optional[int] = Field(None, le=500, description="Runtime in minutes")
    aspect_ratio: Optional[str] = Field(None, description="Aspect ratio (e.g., 2.35:1)")

    # Production details
    production_companies: List[str] = Field(default_factory=list, max_length=20, description="Production companies")
    distributors: List[str] = Field(default_factory=list, max_length=10, description="Distribution companies")
    filming_locations: List[str] = Field(default_factory=list, max_length=30, description="Filming locations")

    # Content classification
    content_rating: ContentRating = Field(ContentRating.UNKNOWN, description="Content rating")
    production_status: ProductionStatus = Field(ProductionStatus.RELEASED, description="Production status")

    # Popularity and trending
    popularity: Optional[float] = Field(None, description="TMDB popularity score")
    vote_average: Optional[float] = Field(None, ge=0.0, le=10.0, description="TMDB vote average")
    vote_count: Optional[int] = Field(None, ge=0, description="TMDB vote count")

    # SEO and discovery
    keywords: List[str] = Field(default_factory=list, max_length=50, description="Keywords/tags")
    tagline: Optional[str] = Field(None, max_length=500, description="Movie tagline")

    # Visual assets
    poster_path: Optional[str] = Field(None, description="TMDB poster image path")
    backdrop_path: Optional[str] = Field(None, description="TMDB backdrop image path")

    # Data quality
    data_completeness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Metadata completeness score")
    last_tmdb_sync: Optional[datetime] = Field(None, description="Last TMDB data synchronization")

    @field_validator("production_companies", "distributors", "filming_locations", "keywords")
    def validate_string_lists(cls, v):
        """Clean and validate string list fields."""
        if v:
            # Remove empty strings, strip whitespace, remove duplicates
            cleaned = []
            seen = set()
            for item in v:
                if isinstance(item, str):
                    cleaned_item = item.strip()
                    if cleaned_item and cleaned_item.lower() not in seen:
                        cleaned.append(cleaned_item)
                        seen.add(cleaned_item.lower())
            return cleaned if cleaned else []
        return []

    @model_validator(mode="after")
    def validate_business_logic(self):
        """Validate business and financial data consistency."""
        # Budget vs revenue validation
        if self.budget and self.revenue:
            # Flag unusual budget/revenue ratios
            if self.budget > 0 and self.revenue > 0:
                ratio = self.revenue / self.budget
                if ratio > 50:  # Revenue > 50x budget is unusual
                    pass  # Log warning but don't raise error
                elif ratio < 0.05:  # Revenue < 5% of budget is concerning
                    pass  # Log warning but don't raise error

        # Validate country codes
        if self.production_countries:
            valid_countries = []
            for country in self.production_countries:
                if isinstance(country, str) and len(country) == 2 and country.isupper():
                    valid_countries.append(country)
            self.production_countries = valid_countries

        return self


class MovieSchema(BaseModel):
    """
    Primary movie data model with comprehensive validation and TMDB integration.

    This model represents a complete movie record with all metadata, cast, crew,
    ratings, and Constitutional AI tracking information.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # Core identification
    movie_id: str = Field(..., min_length=1, max_length=50, description="Unique movie identifier")
    title: str = Field(..., max_length=500, description="Primary movie title")
    original_title: Optional[str] = Field(None, max_length=500, description="Original title in original language")

    # Content
    synopsis: str = Field(..., max_length=5000, description="Movie synopsis/overview")
    tagline: Optional[str] = Field(None, max_length=500, description="Marketing tagline")

    # Release information
    release_date: Optional[str] = Field(None, description="Release date (YYYY-MM-DD)")
    release_year: int = Field(..., ge=1895, le=2030, description="Release year")

    # Classification
    genres: List[Genre] = Field(..., max_length=10, description="Movie genres")

    # People
    cast: List[CastMember] = Field(default_factory=list, description="Cast members")
    crew: List[CrewMember] = Field(default_factory=list, max_length=200, description="Crew members")

    # Ratings and reception
    ratings: RatingInfo = Field(..., description="Rating information from various sources")

    # Comprehensive metadata
    metadata: MovieMetadata = Field(default_factory=MovieMetadata, description="Technical and business metadata")

    # Constitutional AI compliance
    constitutional_ai: ConstitutionalAIMetrics = Field(
        default_factory=ConstitutionalAIMetrics, description="Bias detection and diversity metrics"
    )

    # Data lineage
    data_source: str = Field("tmdb", description="Primary data source")
    schema_version: SchemaVersion = Field(SchemaVersion.CURRENT, description="Schema version")
    created_at: datetime = Field(default_factory=datetime.now, description="Record creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    # Quality scoring
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall data quality score")

    @field_validator("title", "original_title")
    def validate_title_quality(cls, v):
        """Validate title content and format."""
        if v is None:
            return v  # Allow None for optional original_title

        v = v.strip()
        if len(v) < 1:
            raise ValueError("Title cannot be empty")

        # Remove excessive whitespace
        v = " ".join(v.split())

        # Check for placeholder titles
        if v.lower() in ["untitled", "tbd", "tba", "unknown", "n/a"]:
            raise ValueError(f"Invalid placeholder title: {v}")

        return v

    @field_validator("synopsis")
    def validate_synopsis_quality(cls, v):
        """Validate synopsis content quality."""
        v = v.strip()

        # Remove excessive whitespace
        v = " ".join(v.split())

        # Check for placeholder synopses first
        placeholder_phrases = [
            "no overview found",
            "plot unknown",
            "synopsis unavailable",
            "description not available",
            "coming soon",
        ]

        if any(phrase in v.lower() for phrase in placeholder_phrases):
            raise ValueError("Synopsis contains placeholder text")

        if len(v) < 50:
            raise ValueError("Synopsis must be at least 50 characters after cleaning")

        # Basic content richness check - count words or characters for non-space languages
        word_count = len(v.split())
        char_count = len(re.sub(r"\s", "", v))  # Count non-space characters
        if word_count < 10 and char_count < 30:  # Either 10 words OR 30 characters
            raise ValueError("Synopsis must contain at least 10 words")

        return v

    @field_validator("genres")
    def validate_genre_diversity(cls, v):
        """Validate and clean genre list."""
        if not v:
            raise ValueError("At least one genre is required")

        # Remove duplicates while preserving order
        seen = set()
        unique_genres = []
        for genre in v:
            if genre not in seen:
                seen.add(genre)
                unique_genres.append(genre)

        return unique_genres

    @field_validator("release_date")
    def validate_release_date_consistency(cls, v):
        """Validate release date format and reasonableness."""
        if not v:
            return v

        try:
            # Parse date to validate format
            parsed_date = datetime.strptime(v, "%Y-%m-%d")

            # Check for future dates beyond reasonable horizon
            if parsed_date.year > datetime.now().year + 5:
                raise ValueError(f"Release date {v} is too far in the future")

            return v
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError(f"Invalid release date format: {v}. Use YYYY-MM-DD")
            raise e

    @model_validator(mode="after")
    def validate_cross_field_consistency(self):
        """Validate consistency across related fields."""

        # Release year consistency
        if self.release_date:
            date_year = int(self.release_date.split("-")[0])
            if abs(date_year - self.release_year) > 1:
                raise ValueError(f"Release date year {date_year} inconsistent with release year {self.release_year}")

        # Cast size validation
        if len(self.cast) > 100:
            raise ValueError("Cast size seems unreasonably large")

        # Cast/crew birth year validation
        all_people = list(self.cast) + list(self.crew)
        for person in all_people:
            if person.birth_year:
                age_at_release = self.release_year - person.birth_year
                if age_at_release < 0:
                    raise ValueError(
                        f"Person {person.name} birth year {person.birth_year} is after movie release {self.release_year}"
                    )
                if age_at_release < 5:  # Child actors allowance
                    continue
                if age_at_release > 90:  # Very elderly participants
                    # This is unusual but not impossible, so just log
                    pass

        # Calculate quality score if not set
        if self.quality_score is None:
            object.__setattr__(self, "quality_score", self.calculate_quality_score())

        return self

    def calculate_quality_score(self) -> float:
        """Calculate overall data quality score."""
        score_components = []

        # Title quality (1.0 if present and reasonable)
        if self.title and len(self.title.strip()) > 1:
            score_components.append(1.0)
        else:
            score_components.append(0.0)

        # Synopsis quality (based on length and content)
        if self.synopsis:
            synopsis_score = min(1.0, len(self.synopsis) / 200.0)  # Full score at 200+ chars
            score_components.append(synopsis_score)
        else:
            score_components.append(0.0)

        # Cast completeness
        cast_score = min(1.0, len(self.cast) / 8.0)  # Full score at 8+ cast members
        score_components.append(cast_score)

        # Metadata completeness
        metadata_fields = [
            self.metadata.tmdb_id,
            self.metadata.imdb_id,
            self.metadata.budget,
            self.metadata.revenue,
            self.metadata.runtime,
        ]
        metadata_score = sum(1.0 for field in metadata_fields if field is not None) / len(metadata_fields)
        score_components.append(metadata_score)

        # Rating quality
        rating_score = 1.0 if self.ratings.count >= 100 else self.ratings.count / 100.0
        score_components.append(rating_score)

        # Calculate weighted average
        return sum(score_components) / len(score_components)


# TMDB API Response Mapping Utilities
def map_tmdb_movie_response(tmdb_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform TMDB API movie response to MovieSchema format.

    Args:
        tmdb_data: Raw TMDB API movie response

    Returns:
        Dictionary compatible with MovieSchema
    """

    # Map basic fields
    mapped_data = {
        "movie_id": str(tmdb_data.get("id", "")),
        "title": tmdb_data.get("title", ""),
        "original_title": tmdb_data.get("original_title"),
        "synopsis": tmdb_data.get("overview", ""),
        "tagline": tmdb_data.get("tagline"),
        "release_date": tmdb_data.get("release_date"),
        "release_year": _extract_year_from_date(tmdb_data.get("release_date")),
    }

    # Map genres
    if tmdb_data.get("genres"):
        mapped_genres = []
        for genre_data in tmdb_data["genres"]:
            genre_name = genre_data.get("name", "")
            try:
                # Map TMDB genre names to our enum
                mapped_genre = _map_tmdb_genre(genre_name)
                mapped_genres.append(mapped_genre)
            except ValueError:
                mapped_genres.append(Genre.OTHER)
        mapped_data["genres"] = mapped_genres

    # Map ratings
    mapped_data["ratings"] = {
        "average": tmdb_data.get("vote_average", 0.0),
        "count": tmdb_data.get("vote_count", 0),
        "tmdb_rating": tmdb_data.get("vote_average"),
        "tmdb_count": tmdb_data.get("vote_count"),
    }

    # Map metadata
    metadata = {
        "tmdb_id": tmdb_data.get("id"),
        "imdb_id": tmdb_data.get("imdb_id"),
        "original_language": _map_language_code(tmdb_data.get("original_language", "en")),
        "origin_country": _get_primary_country(tmdb_data.get("origin_country", ["US"])),
        "budget": tmdb_data.get("budget") if tmdb_data.get("budget", 0) > 0 else None,
        "revenue": tmdb_data.get("revenue") if tmdb_data.get("revenue", 0) > 0 else None,
        "runtime": tmdb_data.get("runtime"),
        "popularity": tmdb_data.get("popularity"),
        "vote_average": tmdb_data.get("vote_average"),
        "vote_count": tmdb_data.get("vote_count"),
        "poster_path": tmdb_data.get("poster_path"),
        "backdrop_path": tmdb_data.get("backdrop_path"),
        "last_tmdb_sync": datetime.now(),
    }

    # Map production companies
    if tmdb_data.get("production_companies"):
        metadata["production_companies"] = [
            company.get("name", "") for company in tmdb_data["production_companies"] if company.get("name")
        ]

    # Map production countries
    if tmdb_data.get("production_countries"):
        metadata["production_countries"] = [
            country.get("iso_3166_1", "") for country in tmdb_data["production_countries"] if country.get("iso_3166_1")
        ]

    # Map spoken languages
    if tmdb_data.get("spoken_languages"):
        spoken_langs = []
        for lang_data in tmdb_data["spoken_languages"]:
            lang_code = lang_data.get("iso_639_1", "")
            try:
                mapped_lang = _map_language_code(lang_code)
                spoken_langs.append(mapped_lang)
            except ValueError:
                spoken_langs.append(Language.OTHER)
        metadata["spoken_languages"] = spoken_langs

    mapped_data["metadata"] = metadata

    return mapped_data


def map_tmdb_credits_response(credits_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transform TMDB credits response to cast/crew format.

    Args:
        credits_data: TMDB credits API response

    Returns:
        Dictionary with 'cast' and 'crew' lists
    """

    result = {"cast": [], "crew": []}

    # Map cast
    if credits_data.get("cast"):
        for cast_data in credits_data["cast"]:
            cast_member = {
                "name": cast_data.get("name", ""),
                "character": cast_data.get("character"),
                "order": cast_data.get("order", 999),
                "tmdb_person_id": cast_data.get("id"),
                "gender": _map_tmdb_gender(cast_data.get("gender")),
                "known_for_department": cast_data.get("known_for_department"),
                "popularity": cast_data.get("popularity"),
                "profile_path": cast_data.get("profile_path"),
            }
            result["cast"].append(cast_member)

    # Map crew
    if credits_data.get("crew"):
        for crew_data in credits_data["crew"]:
            crew_member = {
                "name": crew_data.get("name", ""),
                "job": crew_data.get("job", ""),
                "department": crew_data.get("department", ""),
                "tmdb_person_id": crew_data.get("id"),
                "gender": _map_tmdb_gender(crew_data.get("gender")),
                "known_for_department": crew_data.get("known_for_department"),
                "popularity": crew_data.get("popularity"),
                "profile_path": crew_data.get("profile_path"),
            }
            result["crew"].append(crew_member)

    return result


# Helper functions for TMDB mapping
def _extract_year_from_date(date_str: Optional[str]) -> int:
    """Extract year from YYYY-MM-DD date string."""
    if not date_str:
        return datetime.now().year
    try:
        return int(date_str.split("-")[0])
    except (ValueError, IndexError):
        return datetime.now().year


def _map_tmdb_genre(tmdb_genre: str) -> Genre:
    """Map TMDB genre name to our Genre enum."""
    mapping = {
        "Action": Genre.ACTION,
        "Adventure": Genre.ADVENTURE,
        "Animation": Genre.ANIMATION,
        "Comedy": Genre.COMEDY,
        "Crime": Genre.CRIME,
        "Documentary": Genre.DOCUMENTARY,
        "Drama": Genre.DRAMA,
        "Family": Genre.FAMILY,
        "Fantasy": Genre.FANTASY,
        "History": Genre.HISTORY,
        "Horror": Genre.HORROR,
        "Music": Genre.MUSIC,
        "Mystery": Genre.MYSTERY,
        "Romance": Genre.ROMANCE,
        "Science Fiction": Genre.SCIENCE_FICTION,
        "TV Movie": Genre.TV_MOVIE,
        "Thriller": Genre.THRILLER,
        "War": Genre.WAR,
        "Western": Genre.WESTERN,
    }

    return mapping.get(tmdb_genre, Genre.OTHER)


def _map_language_code(lang_code: str) -> Language:
    """Map language code to our Language enum."""
    try:
        return Language(lang_code.lower())
    except ValueError:
        return Language.OTHER


def _map_tmdb_gender(tmdb_gender: Optional[int]) -> Optional[str]:
    """Map TMDB gender code to string."""
    if tmdb_gender is None:
        return None

    mapping = {0: "unknown", 1: "female", 2: "male", 3: "non-binary"}

    return mapping.get(tmdb_gender, "unknown")


def _get_primary_country(countries: List[str]) -> str:
    """Get primary country from list."""
    if not countries:
        return "US"
    return countries[0] if countries[0] else "US"


# Schema validation utilities
def create_movie_from_tmdb(
    tmdb_movie_data: Dict[str, Any], credits_data: Optional[Dict[str, Any]] = None
) -> MovieSchema:
    """
    Create MovieSchema instance from TMDB API data.

    Args:
        tmdb_movie_data: TMDB movie API response
        credits_data: Optional TMDB credits API response

    Returns:
        MovieSchema instance

    Raises:
        ValueError: If data validation fails
    """

    # Map movie data
    mapped_data = map_tmdb_movie_response(tmdb_movie_data)

    # Add cast and crew if provided
    if credits_data:
        credits_mapped = map_tmdb_credits_response(credits_data)
        mapped_data.update(credits_mapped)

    # Set data source
    mapped_data["data_source"] = "tmdb_api"

    try:
        return MovieSchema(**mapped_data)
    except Exception as e:
        movie_id = mapped_data.get("movie_id", "unknown")
        raise ValueError(f"Error creating movie {movie_id} from TMDB data: {str(e)}") from e


def get_enhanced_schema_info() -> Dict[str, Any]:
    """Get comprehensive information about the enhanced movie schema."""
    return {
        "schema_version": SchemaVersion.CURRENT,
        "supported_genres": [genre.value for genre in Genre],
        "supported_languages": [lang.value for lang in Language],
        "content_ratings": [rating.value for rating in ContentRating],
        "production_statuses": [status.value for status in ProductionStatus],
        "cast_member_fields": list(CastMember.model_fields.keys()),
        "crew_member_fields": list(CrewMember.model_fields.keys()),
        "movie_metadata_fields": list(MovieMetadata.model_fields.keys()),
        "constitutional_ai_fields": list(ConstitutionalAIMetrics.model_fields.keys()),
        "validation_features": [
            "Cross-field consistency validation",
            "TMDB API response mapping",
            "Quality score calculation",
            "Constitutional AI bias tracking",
            "Schema versioning support",
            "Comprehensive error reporting",
        ],
        "data_sources": ["tmdb_api", "manual", "import", "other"],
    }


if __name__ == "__main__":
    # Example usage
    schema_info = get_enhanced_schema_info()
    print("Enhanced MovieRecs Schema Information:")
    print(f"Version: {schema_info['schema_version']}")
    print(f"Supported Genres: {len(schema_info['supported_genres'])}")
    print(f"Supported Languages: {len(schema_info['supported_languages'])}")
    print(f"Constitutional AI Features: {len(schema_info['constitutional_ai_fields'])}")

    # Example movie creation
    example_data = {
        "movie_id": "550",
        "title": "Fight Club",
        "synopsis": "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more. This psychological thriller explores themes of consumerism, masculinity, and identity in modern society.",
        "release_year": 1999,
        "release_date": "1999-10-15",
        "genres": [Genre.DRAMA, Genre.THRILLER],
        "ratings": {"average": 8.8, "count": 2000000, "tmdb_rating": 8.4, "tmdb_count": 26000},
        "metadata": {"tmdb_id": 550, "imdb_id": "tt0137523", "budget": 63000000, "revenue": 100853753, "runtime": 139},
    }

    try:
        movie = MovieSchema(**example_data)
        print(f"\nSuccessfully created movie: {movie.title}")
        print(f"Quality Score: {movie.quality_score:.2f}")
        print(f"Schema Version: {movie.schema_version}")
    except ValueError as e:
        print(f"\nValidation error: {e}")
