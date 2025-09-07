"""
Advanced validation rules and utilities for movie data quality assurance.

This module implements sophisticated validation logic for movie datasets including:
- Content quality validation (synopsis richness, metadata completeness)
- Cross-reference validation (cast/crew consistency, role validation)
- Temporal validation (release dates, career timelines, historical accuracy)
- Financial validation (budget/revenue relationships, inflation adjustment)
- Technical validation (runtime vs genre expectations, rating consistency)
- TMDB-specific validation rules
"""

import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .movie_schema import (
    CastMember,
    ContentRating,
    CrewMember,
    Genre,
    Language,
    MovieMetadata,
    MovieSchema,
    ProductionStatus,
    RatingInfo,
)


class ValidationSeverity:
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationResult:
    """Enhanced validation result with detailed reporting."""

    def __init__(self):
        self.is_valid = True
        self.issues = []
        self.summary = {}
        self.quality_score = 1.0

    def add_issue(
        self, severity: str, category: str, field: str, message: str, value: Any = None, suggestion: str = None
    ):
        """Add a validation issue."""
        issue = {
            "severity": severity,
            "category": category,
            "field": field,
            "message": message,
            "value": value,
            "suggestion": suggestion,
            "timestamp": datetime.now().isoformat(),
        }
        self.issues.append(issue)

        # Update validity status
        if severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False

    def get_issues_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue["severity"] == severity]

    def get_issues_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get issues filtered by category."""
        return [issue for issue in self.issues if issue["category"] == category]


class ContentQualityValidator:
    """Validates content quality and richness."""

    @staticmethod
    def validate_synopsis_quality(synopsis: str, movie_id: str, result: ValidationResult) -> None:
        """Perform comprehensive synopsis quality validation."""

        if not synopsis or len(synopsis.strip()) == 0:
            result.add_issue(
                ValidationSeverity.CRITICAL,
                "content_quality",
                "synopsis",
                "Synopsis is empty",
                synopsis,
                "Add a descriptive synopsis of at least 50 characters",
            )
            return

        cleaned_synopsis = synopsis.strip()
        word_count = len(cleaned_synopsis.split())
        char_count = len(cleaned_synopsis)

        # Length checks
        if char_count < 50:
            result.add_issue(
                ValidationSeverity.ERROR,
                "content_quality",
                "synopsis",
                f"Synopsis too short: {char_count} characters (minimum: 50)",
                char_count,
                "Expand synopsis to provide more details about plot and themes",
            )
        elif char_count < 100:
            result.add_issue(
                ValidationSeverity.WARNING,
                "content_quality",
                "synopsis",
                f"Synopsis is quite brief: {char_count} characters (recommended: 100+)",
                char_count,
                "Consider adding more plot details or character information",
            )

        # Word count checks
        if word_count < 10:
            result.add_issue(
                ValidationSeverity.ERROR,
                "content_quality",
                "synopsis",
                f"Synopsis has too few words: {word_count} (minimum: 10)",
                word_count,
                "Add more descriptive content to the synopsis",
            )
        elif word_count < 20:
            result.add_issue(
                ValidationSeverity.WARNING,
                "content_quality",
                "synopsis",
                f"Synopsis word count is low: {word_count} (recommended: 20+)",
                word_count,
                "Expand with more plot or character details",
            )

        # Content quality checks
        ContentQualityValidator._check_synopsis_content_quality(cleaned_synopsis, movie_id, result)

    @staticmethod
    def _check_synopsis_content_quality(synopsis: str, movie_id: str, result: ValidationResult) -> None:
        """Check synopsis content for quality indicators."""

        lower_synopsis = synopsis.lower()

        # Check for placeholder text
        placeholders = [
            "no overview found",
            "plot unknown",
            "synopsis unavailable",
            "description not available",
            "coming soon",
            "plot summary",
            "lorem ipsum",
            "placeholder text",
            "to be announced",
            "tba",
            "synopsis pending",
            "overview missing",
        ]

        for placeholder in placeholders:
            if placeholder in lower_synopsis:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "content_quality",
                    "synopsis",
                    f"Synopsis contains placeholder text: '{placeholder}'",
                    synopsis,
                    "Replace placeholder with actual movie synopsis",
                )

        # Check for repetitive content
        words = synopsis.split()
        if len(words) > 5:
            word_counts = Counter(word.lower().strip(".,!?;:") for word in words)
            repeated_words = [word for word, count in word_counts.items() if count > 3 and len(word) > 3]
            if repeated_words:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "content_quality",
                    "synopsis",
                    f"Synopsis has repetitive words: {repeated_words[:3]}",
                    repeated_words,
                    "Vary language to avoid repetition",
                )

        # Check sentence structure
        sentences = re.split(r"[.!?]+", synopsis)
        if len(sentences) < 2:
            result.add_issue(
                ValidationSeverity.WARNING,
                "content_quality",
                "synopsis",
                "Synopsis appears to be only one sentence",
                len(sentences),
                "Consider using multiple sentences for better readability",
            )

        # Check for very short sentences (may indicate poor quality)
        short_sentences = [s.strip() for s in sentences if len(s.strip()) < 10]
        if len(short_sentences) > len(sentences) * 0.3:
            result.add_issue(
                ValidationSeverity.WARNING,
                "content_quality",
                "synopsis",
                f"Synopsis has many short sentences ({len(short_sentences)}/{len(sentences)})",
                short_sentences,
                "Consider combining short sentences for better flow",
            )

    @staticmethod
    def validate_metadata_completeness(metadata: MovieMetadata, movie_id: str, result: ValidationResult) -> None:
        """Validate metadata completeness and quality."""

        # Core identifiers
        if not metadata.tmdb_id:
            result.add_issue(
                ValidationSeverity.WARNING,
                "completeness",
                "tmdb_id",
                "Missing TMDB ID",
                None,
                "Add TMDB ID for better data linking and validation",
            )

        if not metadata.imdb_id:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "imdb_id",
                "Missing IMDB ID",
                None,
                "Add IMDB ID for cross-platform validation",
            )

        # Financial data
        if metadata.budget is None and metadata.revenue is not None:
            result.add_issue(
                ValidationSeverity.WARNING,
                "completeness",
                "budget",
                "Revenue present but budget missing",
                None,
                "Add budget information for complete financial picture",
            )

        if metadata.budget is not None and metadata.revenue is None:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "revenue",
                "Budget present but revenue missing",
                None,
                "Add box office revenue data if available",
            )

        # Technical details
        if not metadata.runtime:
            result.add_issue(
                ValidationSeverity.WARNING,
                "completeness",
                "runtime",
                "Missing runtime information",
                None,
                "Add movie runtime in minutes",
            )

        # Production information
        if not metadata.production_companies:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "production_companies",
                "No production companies listed",
                None,
                "Add production company information",
            )

        # Content rating
        if metadata.content_rating == ContentRating.UNKNOWN:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "content_rating",
                "Content rating not specified",
                None,
                "Add appropriate content rating (G, PG, PG-13, R, etc.)",
            )


class CrossReferenceValidator:
    """Validates consistency across related data fields."""

    @staticmethod
    def validate_cast_consistency(cast: List[CastMember], movie_id: str, result: ValidationResult) -> None:
        """Validate cast data consistency and quality."""

        if not cast:
            result.add_issue(
                ValidationSeverity.WARNING,
                "completeness",
                "cast",
                "No cast information provided",
                None,
                "Add at least main cast members",
            )
            return

        # Check for duplicate cast members
        names_seen = set()
        for i, member in enumerate(cast):
            name_key = member.name.lower().strip()
            if name_key in names_seen:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "consistency",
                    "cast",
                    f"Duplicate cast member: {member.name}",
                    member.name,
                    "Remove duplicate cast entries",
                )
            names_seen.add(name_key)

        # Validate billing order
        orders = [member.order for member in cast if member.order > 0]
        if orders and len(set(orders)) != len(orders):
            result.add_issue(
                ValidationSeverity.WARNING,
                "consistency",
                "cast",
                "Duplicate billing orders in cast",
                orders,
                "Ensure unique billing order for each cast member",
            )

        # Check for reasonable cast size
        if len(cast) > 100:
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "cast",
                f"Very large cast size: {len(cast)} members",
                len(cast),
                "Verify cast list accuracy - consider including only main cast",
            )
        elif len(cast) < 3:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "cast",
                f"Small cast size: {len(cast)} members",
                len(cast),
                "Consider adding more cast members if available",
            )

        # Check character name quality
        unnamed_characters = sum(1 for member in cast if not member.character)
        if unnamed_characters > len(cast) * 0.5:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "cast",
                f"Many cast members without character names: {unnamed_characters}/{len(cast)}",
                unnamed_characters,
                "Add character names where possible",
            )

    @staticmethod
    def validate_crew_consistency(crew: List[CrewMember], movie_id: str, result: ValidationResult) -> None:
        """Validate crew data consistency and completeness."""

        if not crew:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "crew",
                "No crew information provided",
                None,
                "Add key crew members (director, writer, producer)",
            )
            return

        # Check for essential roles
        essential_roles = {"Director", "Writer", "Producer"}
        crew_roles = {member.job for member in crew}
        missing_roles = essential_roles - crew_roles

        if missing_roles:
            result.add_issue(
                ValidationSeverity.WARNING,
                "completeness",
                "crew",
                f"Missing essential crew roles: {', '.join(missing_roles)}",
                list(missing_roles),
                "Add key crew member information",
            )

        # Check for duplicate crew entries
        crew_signatures = set()
        for member in crew:
            signature = (member.name.lower(), member.job.lower())
            if signature in crew_signatures:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "consistency",
                    "crew",
                    f"Duplicate crew entry: {member.name} - {member.job}",
                    f"{member.name} ({member.job})",
                    "Remove duplicate crew entries",
                )
            crew_signatures.add(signature)

        # Validate department consistency
        dept_job_mapping = {
            "Directing": ["Director", "Assistant Director"],
            "Writing": ["Writer", "Screenplay", "Story"],
            "Production": ["Producer", "Executive Producer", "Co-Producer"],
            "Camera": ["Cinematographer", "Director of Photography"],
            "Editing": ["Editor", "Film Editor"],
            "Sound": ["Composer", "Music", "Sound Designer"],
        }

        for member in crew:
            expected_dept = None
            for dept, jobs in dept_job_mapping.items():
                if any(job.lower() in member.job.lower() for job in jobs):
                    expected_dept = dept
                    break

            if expected_dept and member.department != expected_dept:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "consistency",
                    "crew",
                    f"Job '{member.job}' in unexpected department '{member.department}' (expected: {expected_dept})",
                    f"{member.job} in {member.department}",
                    f"Consider moving to {expected_dept} department",
                )


class TemporalValidator:
    """Validates temporal consistency and historical accuracy."""

    @staticmethod
    def validate_release_date_consistency(movie: MovieSchema, result: ValidationResult) -> None:
        """Validate release date and year consistency."""

        if movie.release_date:
            try:
                parsed_date = datetime.strptime(movie.release_date, "%Y-%m-%d")
                date_year = parsed_date.year

                # Check consistency with release_year
                if abs(date_year - movie.release_year) > 1:
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        "consistency",
                        "release_date",
                        f"Release date year ({date_year}) inconsistent with release year ({movie.release_year})",
                        f"Date: {movie.release_date}, Year: {movie.release_year}",
                        "Ensure release date and year are consistent",
                    )

                # Check for future dates (beyond reasonable horizon)
                future_limit = datetime.now() + timedelta(days=365 * 3)  # 3 years
                if parsed_date > future_limit:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        "temporal_logic",
                        "release_date",
                        f"Release date is far in the future: {movie.release_date}",
                        movie.release_date,
                        "Verify release date accuracy",
                    )

            except ValueError:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "format",
                    "release_date",
                    f"Invalid release date format: {movie.release_date}",
                    movie.release_date,
                    "Use YYYY-MM-DD format",
                )

        # Check release year range
        current_year = datetime.now().year
        if movie.release_year < 1895:  # First movies
            result.add_issue(
                ValidationSeverity.WARNING,
                "temporal_logic",
                "release_year",
                f"Release year {movie.release_year} before cinema invention (1895)",
                movie.release_year,
                "Verify release year accuracy",
            )
        elif movie.release_year > current_year + 5:
            result.add_issue(
                ValidationSeverity.WARNING,
                "temporal_logic",
                "release_year",
                f"Release year {movie.release_year} far in future",
                movie.release_year,
                "Verify release year accuracy",
            )

    @staticmethod
    def validate_career_timelines(movie: MovieSchema, result: ValidationResult) -> None:
        """Validate cast and crew birth years against movie release."""

        all_people = list(movie.cast) + list(movie.crew)

        for person in all_people:
            if person.birth_year:
                age_at_release = movie.release_year - person.birth_year
                person_type = "cast" if person in movie.cast else "crew"

                # Check for reasonable age ranges
                if age_at_release < 0:
                    result.add_issue(
                        ValidationSeverity.ERROR,
                        "temporal_logic",
                        f"{person_type}_birth_year",
                        f"{person.name} birth year ({person.birth_year}) after movie release ({movie.release_year})",
                        f"{person.name}: born {person.birth_year}, movie {movie.release_year}",
                        "Verify birth year and release year accuracy",
                    )
                elif age_at_release < 5:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        "temporal_logic",
                        f"{person_type}_birth_year",
                        f"{person.name} was very young ({age_at_release}) at movie release",
                        f"{person.name}: age {age_at_release}",
                        "Verify this is a child actor/performer",
                    )
                elif age_at_release > 95:
                    result.add_issue(
                        ValidationSeverity.WARNING,
                        "temporal_logic",
                        f"{person_type}_birth_year",
                        f"{person.name} was very old ({age_at_release}) at movie release",
                        f"{person.name}: age {age_at_release}",
                        "Verify birth year accuracy",
                    )

        # Check for historical context in older films
        if movie.release_year < 1930:  # Early cinema
            modern_elements = []
            if "Science Fiction" in [g.value for g in movie.genres]:
                modern_elements.append("science fiction genre")
            if movie.metadata.runtime and movie.metadata.runtime > 180:
                modern_elements.append("very long runtime")

            if modern_elements:
                result.add_issue(
                    ValidationSeverity.INFO,
                    "historical_context",
                    "temporal_consistency",
                    f"Early film ({movie.release_year}) with modern elements: {', '.join(modern_elements)}",
                    modern_elements,
                    "Verify historical accuracy",
                )


class FinancialValidator:
    """Validates financial data and business logic."""

    # Inflation adjustment factors (rough estimates, base year 2020)
    INFLATION_FACTORS = {
        1970: 6.87,
        1975: 4.86,
        1980: 3.16,
        1985: 2.45,
        1990: 2.02,
        1995: 1.71,
        2000: 1.48,
        2005: 1.34,
        2010: 1.21,
        2015: 1.09,
        2020: 1.00,
        2021: 0.96,
        2022: 0.89,
        2023: 0.84,
        2024: 0.81,
    }

    @staticmethod
    def validate_budget_revenue_logic(
        metadata: MovieMetadata, release_year: int, movie_id: str, result: ValidationResult
    ) -> None:
        """Validate budget and revenue relationships."""

        budget = metadata.budget
        revenue = metadata.revenue

        if not budget and not revenue:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "financial_data",
                "No financial data provided",
                None,
                "Add budget and/or revenue information if available",
            )
            return

        # Individual validation
        FinancialValidator._validate_budget(budget, release_year, result)
        FinancialValidator._validate_revenue(revenue, release_year, result)

        # Relationship validation
        if budget and revenue:
            FinancialValidator._validate_budget_revenue_relationship(budget, revenue, release_year, result)

    @staticmethod
    def _validate_budget(budget: Optional[int], release_year: int, result: ValidationResult) -> None:
        """Validate budget reasonableness."""

        if not budget:
            return

        if budget < 0:
            result.add_issue(
                ValidationSeverity.ERROR,
                "data_quality",
                "budget",
                f"Negative budget: ${budget:,}",
                budget,
                "Budget should be positive or zero",
            )
            return

        # Adjust for inflation to get rough modern equivalent
        inflation_factor = FinancialValidator._get_inflation_factor(release_year)
        adjusted_budget = budget * inflation_factor

        # Reasonableness checks (in inflation-adjusted dollars)
        if adjusted_budget < 1000:  # Very low budget
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "budget",
                f"Very low budget: ${budget:,} (${adjusted_budget:,.0f} in 2020 dollars)",
                budget,
                "Verify budget accuracy - may be incomplete or in different currency",
            )
        elif adjusted_budget > 500_000_000:  # Very high budget
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "budget",
                f"Very high budget: ${budget:,} (${adjusted_budget:,.0f} in 2020 dollars)",
                budget,
                "Verify budget accuracy - may include marketing costs",
            )

    @staticmethod
    def _validate_revenue(revenue: Optional[int], release_year: int, result: ValidationResult) -> None:
        """Validate revenue reasonableness."""

        if not revenue:
            return

        if revenue < 0:
            result.add_issue(
                ValidationSeverity.ERROR,
                "data_quality",
                "revenue",
                f"Negative revenue: ${revenue:,}",
                revenue,
                "Revenue should be positive or zero",
            )
            return

        # Adjust for inflation
        inflation_factor = FinancialValidator._get_inflation_factor(release_year)
        adjusted_revenue = revenue * inflation_factor

        # Check for unrealistic revenue figures
        if adjusted_revenue > 3_000_000_000:  # Higher than highest-grossing films
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "revenue",
                f"Extremely high revenue: ${revenue:,} (${adjusted_revenue:,.0f} in 2020 dollars)",
                revenue,
                "Verify revenue accuracy - may be lifetime or global total",
            )

    @staticmethod
    def _validate_budget_revenue_relationship(
        budget: int, revenue: int, release_year: int, result: ValidationResult
    ) -> None:
        """Validate budget vs revenue relationship."""

        if budget == 0 or revenue == 0:
            return

        ratio = revenue / budget

        # Flag unusual ratios
        if ratio > 100:  # Revenue more than 100x budget
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "financial_ratio",
                f"Unusually high revenue/budget ratio: {ratio:.1f}x (${revenue:,} / ${budget:,})",
                {"ratio": ratio, "budget": budget, "revenue": revenue},
                "Verify budget and revenue figures - may indicate data error",
            )
        elif ratio < 0.01:  # Revenue less than 1% of budget
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "financial_ratio",
                f"Unusually low revenue/budget ratio: {ratio:.3f}x (${revenue:,} / ${budget:,})",
                {"ratio": ratio, "budget": budget, "revenue": revenue},
                "Verify financial data - may indicate box office failure or incomplete revenue data",
            )
        elif ratio < 0.5:  # Revenue less than 50% of budget (significant loss)
            result.add_issue(
                ValidationSeverity.INFO,
                "financial_analysis",
                "financial_ratio",
                f"Low revenue/budget ratio suggests box office underperformance: {ratio:.2f}x",
                ratio,
                "This is informational - some films don't recoup production costs",
            )

    @staticmethod
    def _get_inflation_factor(year: int) -> float:
        """Get inflation adjustment factor for given year."""

        # Find closest year in our factors
        closest_year = min(FinancialValidator.INFLATION_FACTORS.keys(), key=lambda x: abs(x - year))

        return FinancialValidator.INFLATION_FACTORS.get(closest_year, 1.0)


class TechnicalValidator:
    """Validates technical aspects and genre-specific expectations."""

    # Expected runtime ranges by genre (in minutes)
    GENRE_RUNTIME_EXPECTATIONS = {
        Genre.DOCUMENTARY: (60, 150),
        Genre.ANIMATION: (70, 120),
        Genre.COMEDY: (80, 120),
        Genre.HORROR: (80, 110),
        Genre.ACTION: (90, 150),
        Genre.DRAMA: (90, 180),
        Genre.THRILLER: (90, 140),
        Genre.ROMANCE: (85, 130),
        Genre.SCIENCE_FICTION: (100, 180),
        Genre.FANTASY: (100, 180),
        Genre.ADVENTURE: (100, 160),
        Genre.WAR: (110, 200),
        Genre.HISTORY: (110, 200),
        Genre.CRIME: (95, 160),
        Genre.MYSTERY: (90, 140),
        Genre.WESTERN: (90, 150),
        Genre.FAMILY: (80, 110),
        Genre.MUSIC: (90, 140),
    }

    @staticmethod
    def validate_runtime_expectations(movie: MovieSchema, result: ValidationResult) -> None:
        """Validate runtime against genre expectations."""

        runtime = movie.metadata.runtime
        if runtime is None:
            result.add_issue(
                ValidationSeverity.INFO,
                "completeness",
                "runtime",
                "Runtime not specified",
                None,
                "Add movie runtime information",
            )
            return

        # Basic runtime validation
        if runtime < 1:
            result.add_issue(
                ValidationSeverity.ERROR,
                "data_quality",
                "runtime",
                f"Invalid runtime: {runtime} minutes",
                runtime,
                "Runtime must be positive",
            )
            return
        elif runtime < 40:
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "runtime",
                f"Very short runtime: {runtime} minutes",
                runtime,
                "Verify runtime - this may be a short film or incomplete data",
            )
        elif runtime > 300:  # 5 hours
            result.add_issue(
                ValidationSeverity.WARNING,
                "data_quality",
                "runtime",
                f"Very long runtime: {runtime} minutes ({runtime//60}h {runtime%60}m)",
                runtime,
                "Verify runtime accuracy - this is unusually long",
            )

        # Genre-specific validation
        primary_genre = movie.genres[0] if movie.genres else None
        if primary_genre and primary_genre in TechnicalValidator.GENRE_RUNTIME_EXPECTATIONS:
            min_expected, max_expected = TechnicalValidator.GENRE_RUNTIME_EXPECTATIONS[primary_genre]

            if runtime < min_expected:
                result.add_issue(
                    ValidationSeverity.INFO,
                    "genre_expectations",
                    "runtime",
                    f"Runtime ({runtime}m) shorter than typical for {primary_genre.value} ({min_expected}-{max_expected}m)",
                    {"runtime": runtime, "genre": primary_genre.value, "expected_range": (min_expected, max_expected)},
                    "This may be normal for some films, but verify accuracy",
                )
            elif runtime > max_expected:
                result.add_issue(
                    ValidationSeverity.INFO,
                    "genre_expectations",
                    "runtime",
                    f"Runtime ({runtime}m) longer than typical for {primary_genre.value} ({min_expected}-{max_expected}m)",
                    {"runtime": runtime, "genre": primary_genre.value, "expected_range": (min_expected, max_expected)},
                    "This may be an epic/extended version, but verify accuracy",
                )

    @staticmethod
    def validate_rating_consistency(
        ratings: RatingInfo, metadata: MovieMetadata, movie_id: str, result: ValidationResult
    ) -> None:
        """Validate rating data consistency and quality."""

        # Basic rating validation
        if ratings.average < 0 or ratings.average > 10:
            result.add_issue(
                ValidationSeverity.ERROR,
                "data_quality",
                "rating_average",
                f"Rating average out of range: {ratings.average} (expected: 0-10)",
                ratings.average,
                "Ensure rating is on 0-10 scale",
            )

        if ratings.count < 0:
            result.add_issue(
                ValidationSeverity.ERROR,
                "data_quality",
                "rating_count",
                f"Negative rating count: {ratings.count}",
                ratings.count,
                "Rating count must be non-negative",
            )

        # Cross-source consistency
        rating_sources = []
        if ratings.tmdb_rating is not None:
            rating_sources.append(("TMDB", ratings.tmdb_rating))
        if ratings.imdb_rating is not None:
            rating_sources.append(("IMDB", ratings.imdb_rating))

        if len(rating_sources) > 1:
            rating_values = [rating for _, rating in rating_sources]
            rating_std = statistics.stdev(rating_values) if len(rating_values) > 1 else 0

            if rating_std > 2.0:  # High deviation between sources
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "consistency",
                    "rating_sources",
                    f"Large rating discrepancy between sources (std: {rating_std:.2f})",
                    {source: rating for source, rating in rating_sources},
                    "Verify rating data accuracy across sources",
                )

        # Rating count reasonableness
        if ratings.count == 0 and ratings.average > 0:
            result.add_issue(
                ValidationSeverity.WARNING,
                "consistency",
                "rating_data",
                "Rating average provided but count is zero",
                {"average": ratings.average, "count": ratings.count},
                "Ensure rating count matches average rating presence",
            )
        elif ratings.count > 0 and ratings.average == 0:
            result.add_issue(
                ValidationSeverity.WARNING,
                "consistency",
                "rating_data",
                "Rating count provided but average is zero",
                {"average": ratings.average, "count": ratings.count},
                "Verify rating average calculation",
            )

        # Popularity vs rating consistency (if TMDB data available)
        if metadata.popularity and ratings.tmdb_rating and metadata.popularity > 100 and ratings.tmdb_rating < 5.0:
            result.add_issue(
                ValidationSeverity.INFO,
                "consistency",
                "popularity_rating",
                f"High popularity ({metadata.popularity}) but low rating ({ratings.tmdb_rating})",
                {"popularity": metadata.popularity, "rating": ratings.tmdb_rating},
                "This may indicate controversial or polarizing content",
            )


class TMDBSpecificValidator:
    """Validators specifically for TMDB API data quality."""

    @staticmethod
    def validate_tmdb_ids(metadata: MovieMetadata, movie_id: str, result: ValidationResult) -> None:
        """Validate TMDB-specific identifier formats and consistency."""

        # TMDB ID validation
        if metadata.tmdb_id:
            if not isinstance(metadata.tmdb_id, int) or metadata.tmdb_id <= 0:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "format",
                    "tmdb_id",
                    f"Invalid TMDB ID format: {metadata.tmdb_id} (must be positive integer)",
                    metadata.tmdb_id,
                    "Use valid TMDB movie ID",
                )
        else:
            result.add_issue(
                ValidationSeverity.WARNING,
                "completeness",
                "tmdb_id",
                "Missing TMDB ID",
                None,
                "Add TMDB ID for data validation and linking",
            )

        # IMDB ID format validation
        if metadata.imdb_id:
            if not re.match(r"^tt\d{7,8}$", metadata.imdb_id):
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "format",
                    "imdb_id",
                    f"Invalid IMDB ID format: {metadata.imdb_id} (expected: tt followed by 7-8 digits)",
                    metadata.imdb_id,
                    "Use correct IMDB ID format (e.g., tt0137523)",
                )

    @staticmethod
    def validate_tmdb_popularity_scores(
        metadata: MovieMetadata, ratings: RatingInfo, movie_id: str, result: ValidationResult
    ) -> None:
        """Validate TMDB popularity and vote data."""

        if metadata.popularity is not None:
            if metadata.popularity < 0:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    "data_quality",
                    "popularity",
                    f"Negative popularity score: {metadata.popularity}",
                    metadata.popularity,
                    "Popularity scores should be non-negative",
                )
            elif metadata.popularity > 1000:  # Very high popularity
                result.add_issue(
                    ValidationSeverity.INFO,
                    "data_quality",
                    "popularity",
                    f"Very high popularity score: {metadata.popularity}",
                    metadata.popularity,
                    "Verify this represents current/peak popularity",
                )

        # Cross-validate TMDB vote data with ratings
        if metadata.vote_count and ratings.tmdb_count:
            if metadata.vote_count != ratings.tmdb_count:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "consistency",
                    "vote_count",
                    f"Inconsistent vote counts: metadata({metadata.vote_count}) vs ratings({ratings.tmdb_count})",
                    {"metadata": metadata.vote_count, "ratings": ratings.tmdb_count},
                    "Ensure vote count consistency across fields",
                )

        if metadata.vote_average and ratings.tmdb_rating:
            if abs(metadata.vote_average - ratings.tmdb_rating) > 0.1:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "consistency",
                    "vote_average",
                    f"Inconsistent vote averages: metadata({metadata.vote_average}) vs ratings({ratings.tmdb_rating})",
                    {"metadata": metadata.vote_average, "ratings": ratings.tmdb_rating},
                    "Ensure vote average consistency across fields",
                )


class ComprehensiveMovieValidator:
    """Main validator that orchestrates all validation types."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.enabled_validators = self.config.get(
            "enabled_validators",
            {
                "content_quality": True,
                "cross_reference": True,
                "temporal": True,
                "financial": True,
                "technical": True,
                "tmdb_specific": True,
            },
        )

    def validate_movie(self, movie: MovieSchema) -> ValidationResult:
        """Perform comprehensive movie validation."""

        result = ValidationResult()

        # Content quality validation
        if self.enabled_validators.get("content_quality", True):
            ContentQualityValidator.validate_synopsis_quality(movie.synopsis, movie.movie_id, result)
            ContentQualityValidator.validate_metadata_completeness(movie.metadata, movie.movie_id, result)

        # Cross-reference validation
        if self.enabled_validators.get("cross_reference", True):
            CrossReferenceValidator.validate_cast_consistency(movie.cast, movie.movie_id, result)
            CrossReferenceValidator.validate_crew_consistency(movie.crew, movie.movie_id, result)

        # Temporal validation
        if self.enabled_validators.get("temporal", True):
            TemporalValidator.validate_release_date_consistency(movie, result)
            TemporalValidator.validate_career_timelines(movie, result)

        # Financial validation
        if self.enabled_validators.get("financial", True):
            FinancialValidator.validate_budget_revenue_logic(movie.metadata, movie.release_year, movie.movie_id, result)

        # Technical validation
        if self.enabled_validators.get("technical", True):
            TechnicalValidator.validate_runtime_expectations(movie, result)
            TechnicalValidator.validate_rating_consistency(movie.ratings, movie.metadata, movie.movie_id, result)

        # TMDB-specific validation
        if self.enabled_validators.get("tmdb_specific", True):
            TMDBSpecificValidator.validate_tmdb_ids(movie.metadata, movie.movie_id, result)
            TMDBSpecificValidator.validate_tmdb_popularity_scores(movie.metadata, movie.ratings, movie.movie_id, result)

        # Calculate overall quality score based on issues
        result.quality_score = self._calculate_quality_score(result)

        # Generate summary
        result.summary = self._generate_validation_summary(result)

        return result

    def _calculate_quality_score(self, result: ValidationResult) -> float:
        """Calculate quality score based on validation issues."""

        if not result.issues:
            return 1.0

        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 0.1,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.ERROR: 0.7,
            ValidationSeverity.CRITICAL: 1.0,
        }

        total_penalty = sum(severity_weights.get(issue["severity"], 0.5) for issue in result.issues)

        # Normalize penalty (assuming 10 issues = 50% quality reduction)
        normalized_penalty = min(0.5, total_penalty / 20)

        return max(0.0, 1.0 - normalized_penalty)

    def _generate_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """Generate summary statistics for validation result."""

        summary = {
            "total_issues": len(result.issues),
            "is_valid": result.is_valid,
            "quality_score": result.quality_score,
            "issues_by_severity": {},
            "issues_by_category": {},
            "validation_timestamp": datetime.now().isoformat(),
        }

        # Count by severity
        for severity in [
            ValidationSeverity.INFO,
            ValidationSeverity.WARNING,
            ValidationSeverity.ERROR,
            ValidationSeverity.CRITICAL,
        ]:
            count = len(result.get_issues_by_severity(severity))
            summary["issues_by_severity"][severity] = count

        # Count by category
        categories = set(issue["category"] for issue in result.issues)
        for category in categories:
            count = len(result.get_issues_by_category(category))
            summary["issues_by_category"][category] = count

        return summary

    def validate_movie_batch(self, movies: List[MovieSchema]) -> Dict[str, ValidationResult]:
        """Validate multiple movies and return results by movie ID."""

        results = {}
        for movie in movies:
            results[movie.movie_id] = self.validate_movie(movie)

        return results


if __name__ == "__main__":
    # Example usage
    from .movie_schema import Genre, MovieSchema, RatingInfo

    # Create validator
    validator = ComprehensiveMovieValidator()

    # Example movie data
    example_movie = MovieSchema(
        movie_id="550",
        title="Fight Club",
        synopsis="An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
        release_year=1999,
        genres=[Genre.DRAMA, Genre.THRILLER],
        ratings=RatingInfo(average=8.8, count=2000000),
    )

    # Validate
    result = validator.validate_movie(example_movie)

    print(f"Validation Result for {example_movie.title}:")
    print(f"Valid: {result.is_valid}")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Total Issues: {result.summary['total_issues']}")

    if result.issues:
        print("\nIssues found:")
        for issue in result.issues[:5]:  # Show first 5 issues
            print(f"  {issue['severity'].upper()}: {issue['message']}")
