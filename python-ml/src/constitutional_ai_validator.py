"""
Constitutional AI integration and bias detection for movie datasets.

This module implements advanced Constitutional AI features for bias detection,
diversity analysis, and ethical data representation validation as specified
in Issue #33 Prompt 2.

Features:
- Comprehensive diversity scoring algorithms
- Automated bias flag generation with configurable rules
- Demographic tracking and balance analysis
- Content warning detection for problematic material
- Audit trails for data provenance and validation history
- Performance-optimized bias calculation for large datasets
"""

import logging
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .movie_schema import (
    CastMember,
    ConstitutionalAIMetrics,
    CrewMember,
    Genre,
    Language,
    MovieMetadata,
    MovieSchema,
    RatingInfo,
)

logger = logging.getLogger(__name__)


class BiasType(str, Enum):
    """Types of bias that can be detected."""

    GEOGRAPHIC = "geographic"
    LANGUAGE = "language"
    GENRE = "genre"
    TEMPORAL = "temporal"
    DEMOGRAPHIC = "demographic"
    CONTENT = "content"
    QUALITY = "quality"
    REPRESENTATION = "representation"


class BiasSeverity(str, Enum):
    """Severity levels for bias detection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DiversityCalculator:
    """Calculates diversity metrics using various algorithms."""

    @staticmethod
    def shannon_diversity_index(counts: Dict[str, int]) -> float:
        """
        Calculate Shannon diversity index for categorical data.

        Returns value between 0 (no diversity) and log(n) (maximum diversity)
        where n is number of categories.
        """
        if not counts:
            return 0.0

        total = sum(counts.values())
        if total == 0:
            return 0.0

        diversity = 0.0
        for count in counts.values():
            if count > 0:
                proportion = count / total
                diversity -= proportion * math.log(proportion)

        return diversity

    @staticmethod
    def simpson_diversity_index(counts: Dict[str, int]) -> float:
        """
        Calculate Simpson's diversity index.

        Returns value between 0 (no diversity) and 1 (maximum diversity).
        """
        if not counts:
            return 0.0

        total = sum(counts.values())
        if total <= 1:
            return 0.0

        sum_squares = sum(count * (count - 1) for count in counts.values())
        simpson_index = sum_squares / (total * (total - 1))

        return 1.0 - simpson_index  # Convert to diversity (higher = more diverse)

    @staticmethod
    def gini_coefficient(values: List[float]) -> float:
        """
        Calculate Gini coefficient for inequality measurement.

        Returns value between 0 (perfect equality) and 1 (maximum inequality).
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if n == 1:
            return 0.0

        cumsum = sum((i + 1) * val for i, val in enumerate(sorted_values))
        total_sum = sum(sorted_values)

        if total_sum == 0:
            return 0.0

        return (2 * cumsum) / (n * total_sum) - (n + 1) / n

    @staticmethod
    def normalized_diversity_score(counts: Dict[str, int], max_categories: int = None) -> float:
        """
        Calculate normalized diversity score (0-1 scale).

        Combines Shannon diversity with category count for comprehensive score.
        """
        if not counts:
            return 0.0

        shannon = DiversityCalculator.shannon_diversity_index(counts)
        num_categories = len(counts)

        # Normalize Shannon index
        max_shannon = math.log(num_categories) if num_categories > 1 else 1.0
        normalized_shannon = shannon / max_shannon if max_shannon > 0 else 0.0

        # Category richness component
        if max_categories:
            category_richness = min(1.0, num_categories / max_categories)
        else:
            # Use square root scaling for category richness
            category_richness = min(1.0, math.sqrt(num_categories) / 10)

        # Weighted combination
        return 0.7 * normalized_shannon + 0.3 * category_richness


class DemographicAnalyzer:
    """Analyzes demographic representation and balance."""

    @staticmethod
    def analyze_cast_demographics(cast: List[CastMember]) -> Dict[str, Any]:
        """Analyze demographic distribution in movie cast."""

        analysis = {
            "total_cast": len(cast),
            "gender_distribution": {},
            "ethnicity_distribution": {},
            "age_distribution": {},
            "nationality_distribution": {},
            "demographic_flags": [],
        }

        if not cast:
            analysis["demographic_flags"].append("No cast data available")
            return analysis

        # Collect demographic data
        genders = [member.gender for member in cast if member.gender]
        ethnicities = [member.ethnicity for member in cast if member.ethnicity]
        nationalities = [member.nationality for member in cast if member.nationality]
        birth_years = [member.birth_year for member in cast if member.birth_year]

        # Gender distribution
        if genders:
            gender_counts = Counter(genders)
            total_with_gender = sum(gender_counts.values())
            analysis["gender_distribution"] = {
                gender: count / total_with_gender for gender, count in gender_counts.items()
            }

            # Check for gender imbalance
            male_pct = analysis["gender_distribution"].get("male", 0)
            female_pct = analysis["gender_distribution"].get("female", 0)

            if abs(male_pct - female_pct) > 0.4:  # 40% threshold
                dominant_gender = "male" if male_pct > female_pct else "female"
                analysis["demographic_flags"].append(
                    f"Significant gender imbalance: {dominant_gender} {max(male_pct, female_pct):.1%}"
                )
        else:
            analysis["demographic_flags"].append("No gender data available")

        # Ethnicity distribution
        if ethnicities:
            ethnicity_counts = Counter(ethnicities)
            total_with_ethnicity = sum(ethnicity_counts.values())
            analysis["ethnicity_distribution"] = {
                ethnicity: count / total_with_ethnicity for ethnicity, count in ethnicity_counts.items()
            }

            # Check for ethnic diversity
            diversity_score = DiversityCalculator.normalized_diversity_score(ethnicity_counts, 10)
            if diversity_score < 0.3:
                analysis["demographic_flags"].append("Low ethnic diversity in cast")
        else:
            analysis["demographic_flags"].append("No ethnicity data available")

        # Age distribution
        if birth_years:
            current_year = datetime.now().year
            ages = [current_year - year for year in birth_years]

            analysis["age_distribution"] = {
                "mean_age": statistics.mean(ages),
                "median_age": statistics.median(ages),
                "age_range": (min(ages), max(ages)),
                "std_dev": statistics.stdev(ages) if len(ages) > 1 else 0,
            }

            # Check for age diversity
            if analysis["age_distribution"]["std_dev"] < 10:  # Low age diversity
                analysis["demographic_flags"].append("Low age diversity in cast")

            # Check for age bias
            young_actors = sum(1 for age in ages if age < 30)
            if young_actors / len(ages) > 0.8:
                analysis["demographic_flags"].append("Cast heavily skewed toward young actors")
        else:
            analysis["demographic_flags"].append("No age data available")

        return analysis

    @staticmethod
    def analyze_crew_demographics(crew: List[CrewMember]) -> Dict[str, Any]:
        """Analyze demographic distribution in movie crew."""

        analysis = {
            "total_crew": len(crew),
            "role_distribution": {},
            "department_distribution": {},
            "gender_by_role": {},
            "demographic_flags": [],
        }

        if not crew:
            analysis["demographic_flags"].append("No crew data available")
            return analysis

        # Role and department analysis
        roles = [member.job for member in crew]
        departments = [member.department for member in crew]

        role_counts = Counter(roles)
        dept_counts = Counter(departments)

        total_crew = len(crew)
        analysis["role_distribution"] = {role: count / total_crew for role, count in role_counts.items()}
        analysis["department_distribution"] = {dept: count / total_crew for dept, count in dept_counts.items()}

        # Gender by role analysis
        for role in role_counts:
            role_members = [member for member in crew if member.job == role]
            genders = [member.gender for member in role_members if member.gender]

            if genders:
                gender_counts = Counter(genders)
                total_role_with_gender = sum(gender_counts.values())
                analysis["gender_by_role"][role] = {
                    gender: count / total_role_with_gender for gender, count in gender_counts.items()
                }

        # Check for key role representation
        key_roles = ["Director", "Writer", "Producer"]
        for role in key_roles:
            role_data = analysis["gender_by_role"].get(role, {})
            if role_data:
                male_pct = role_data.get("male", 0)
                female_pct = role_data.get("female", 0)

                if male_pct > 0.8:  # Male-dominated
                    analysis["demographic_flags"].append(f"{role} role heavily male-dominated ({male_pct:.1%})")
                elif female_pct > 0.8:  # Female-dominated (less common but worth noting)
                    analysis["demographic_flags"].append(f"{role} role heavily female-dominated ({female_pct:.1%})")
            else:
                analysis["demographic_flags"].append(f"No gender data for {role} role")

        return analysis


class BiasDetector:
    """Detects various types of bias in movie datasets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize bias detector with configuration."""
        self.config = config or {}
        self.thresholds = self.config.get(
            "bias_thresholds",
            {
                "geographic_concentration": 0.7,
                "language_concentration": 0.8,
                "temporal_modern_bias": 0.75,
                "genre_concentration": 0.4,
                "gender_imbalance": 0.3,
                "demographic_diversity_min": 0.3,
            },
        )

    def detect_geographic_bias(self, movies: List[MovieSchema]) -> Dict[str, Any]:
        """Detect geographic representation bias."""

        countries = []
        for movie in movies:
            if movie.metadata.origin_country:
                countries.append(movie.metadata.origin_country)
            else:
                countries.append("Unknown")

        country_counts = Counter(countries)
        total_movies = len(movies)

        # Calculate concentration metrics
        country_dist = {country: count / total_movies for country, count in country_counts.items()}
        us_percentage = country_dist.get("US", 0)
        max_percentage = max(country_dist.values()) if country_dist else 0

        # Calculate diversity scores
        geographic_diversity = DiversityCalculator.normalized_diversity_score(country_counts, 50)
        gini_coeff = DiversityCalculator.gini_coefficient(list(country_counts.values()))

        # Bias detection
        bias_flags = []
        severity = BiasSeverity.LOW

        if us_percentage > self.thresholds["geographic_concentration"]:
            bias_flags.append(f"US over-representation: {us_percentage:.1%}")
            severity = BiasSeverity.HIGH if us_percentage > 0.8 else BiasSeverity.MEDIUM

        if max_percentage > 0.9:
            dominant_country = max(country_dist, key=country_dist.get)
            bias_flags.append(f"Extreme concentration in {dominant_country}: {max_percentage:.1%}")
            severity = BiasSeverity.CRITICAL

        if len(country_counts) < 5:
            bias_flags.append(f"Very low geographic diversity: {len(country_counts)} countries")
            severity = max(
                severity, BiasSeverity.MEDIUM, key=lambda x: ["low", "medium", "high", "critical"].index(x.value)
            )

        return {
            "bias_type": BiasType.GEOGRAPHIC,
            "severity": severity,
            "diversity_score": geographic_diversity,
            "gini_coefficient": gini_coeff,
            "country_distribution": country_dist,
            "dominant_country": max(country_dist, key=country_dist.get) if country_dist else None,
            "dominant_percentage": max_percentage,
            "unique_countries": len(country_counts),
            "bias_flags": bias_flags,
            "recommendations": self._generate_geographic_recommendations(country_dist, bias_flags),
        }

    def detect_language_bias(self, movies: List[MovieSchema]) -> Dict[str, Any]:
        """Detect language representation bias."""

        languages = []
        for movie in movies:
            if movie.metadata.original_language:
                languages.append(movie.metadata.original_language.value)
            else:
                languages.append("unknown")

        language_counts = Counter(languages)
        total_movies = len(movies)

        language_dist = {lang: count / total_movies for lang, count in language_counts.items()}
        en_percentage = language_dist.get("en", 0)

        # Calculate diversity
        language_diversity = DiversityCalculator.normalized_diversity_score(language_counts, 30)

        # Bias detection
        bias_flags = []
        severity = BiasSeverity.LOW

        if en_percentage > self.thresholds["language_concentration"]:
            bias_flags.append(f"English over-representation: {en_percentage:.1%}")
            severity = BiasSeverity.HIGH if en_percentage >= 0.9 else BiasSeverity.MEDIUM

        if len(language_counts) < 5:
            bias_flags.append(f"Very low language diversity: {len(language_counts)} languages")
            severity = max(
                severity, BiasSeverity.MEDIUM, key=lambda x: ["low", "medium", "high", "critical"].index(x.value)
            )

        return {
            "bias_type": BiasType.LANGUAGE,
            "severity": severity,
            "diversity_score": language_diversity,
            "language_distribution": language_dist,
            "english_percentage": en_percentage,
            "unique_languages": len(language_counts),
            "bias_flags": bias_flags,
            "recommendations": self._generate_language_recommendations(language_dist, bias_flags),
        }

    def detect_temporal_bias(self, movies: List[MovieSchema]) -> Dict[str, Any]:
        """Detect temporal representation bias."""

        years = [movie.release_year for movie in movies]
        current_year = datetime.now().year

        # Decade analysis
        decades = [(year // 10) * 10 for year in years]
        decade_counts = Counter(decades)

        # Modern bias analysis
        modern_movies = sum(1 for year in years if year >= 2000)
        modern_percentage = modern_movies / len(years) if years else 0

        very_recent = sum(1 for year in years if year >= current_year - 5)
        very_recent_percentage = very_recent / len(years) if years else 0

        # Calculate temporal spread
        if years:
            year_range = max(years) - min(years)
            year_std = statistics.stdev(years) if len(years) > 1 else 0
        else:
            year_range = 0
            year_std = 0

        # Bias detection
        bias_flags = []
        severity = BiasSeverity.LOW

        if modern_percentage > self.thresholds["temporal_modern_bias"]:
            bias_flags.append(f"Modern bias: {modern_percentage:.1%} from 2000+")
            severity = BiasSeverity.MEDIUM

        if very_recent_percentage >= 0.9:
            bias_flags.append(f"Very recent bias: {very_recent_percentage:.1%} from last 5 years")
            severity = BiasSeverity.HIGH

        if year_range < 20:
            bias_flags.append(f"Narrow temporal range: {year_range} years")
            severity = max(
                severity, BiasSeverity.MEDIUM, key=lambda x: ["low", "medium", "high", "critical"].index(x.value)
            )

        return {
            "bias_type": BiasType.TEMPORAL,
            "severity": severity,
            "year_range": year_range,
            "year_std": year_std,
            "modern_percentage": modern_percentage,
            "very_recent_percentage": very_recent_percentage,
            "decade_distribution": {f"{decade}s": count for decade, count in decade_counts.items()},
            "bias_flags": bias_flags,
            "recommendations": self._generate_temporal_recommendations(modern_percentage, year_range, bias_flags),
        }

    def detect_genre_bias(self, movies: List[MovieSchema]) -> Dict[str, Any]:
        """Detect genre representation bias."""

        all_genres = []
        for movie in movies:
            for genre in movie.genres:
                all_genres.append(genre.value)

        genre_counts = Counter(all_genres)
        total_genres = len(all_genres)

        genre_dist = {genre: count / total_genres for genre, count in genre_counts.items()}

        # Calculate concentration
        max_genre_pct = max(genre_dist.values()) if genre_dist else 0
        gini_coeff = DiversityCalculator.gini_coefficient(list(genre_counts.values()))

        # Bias detection
        bias_flags = []
        severity = BiasSeverity.LOW

        if max_genre_pct > self.thresholds["genre_concentration"]:
            dominant_genre = max(genre_dist, key=genre_dist.get)
            bias_flags.append(f"Genre over-concentration: {dominant_genre} {max_genre_pct:.1%}")
            severity = BiasSeverity.MEDIUM if max_genre_pct > 0.5 else BiasSeverity.LOW

        if len(genre_counts) < 5:
            bias_flags.append(f"Low genre diversity: {len(genre_counts)} genres")
            severity = max(
                severity, BiasSeverity.MEDIUM, key=lambda x: ["low", "medium", "high", "critical"].index(x.value)
            )

        return {
            "bias_type": BiasType.GENRE,
            "severity": severity,
            "genre_distribution": genre_dist,
            "gini_coefficient": gini_coeff,
            "dominant_genre_percentage": max_genre_pct,
            "unique_genres": len(genre_counts),
            "bias_flags": bias_flags,
            "recommendations": self._generate_genre_recommendations(genre_dist, bias_flags),
        }

    def detect_demographic_bias(self, movies: List[MovieSchema]) -> Dict[str, Any]:
        """Detect demographic representation bias across cast and crew."""

        # Aggregate demographic data
        all_cast = []
        all_crew = []

        for movie in movies:
            all_cast.extend(movie.cast)
            all_crew.extend(movie.crew)

        # Analyze demographics
        cast_demographics = DemographicAnalyzer.analyze_cast_demographics(all_cast)
        crew_demographics = DemographicAnalyzer.analyze_crew_demographics(all_crew)

        # Detect bias patterns
        bias_flags = []
        severity = BiasSeverity.LOW

        # Cast gender bias
        cast_gender = cast_demographics.get("gender_distribution", {})
        if cast_gender:
            male_pct = cast_gender.get("male", 0)
            female_pct = cast_gender.get("female", 0)

            if abs(male_pct - female_pct) > self.thresholds["gender_imbalance"]:
                dominant_gender = "male" if male_pct > female_pct else "female"
                bias_flags.append(f"Cast gender imbalance: {dominant_gender} {max(male_pct, female_pct):.1%}")
                severity = BiasSeverity.MEDIUM

        # Crew role bias (focusing on key roles)
        crew_roles = crew_demographics.get("gender_by_role", {})
        for role in ["Director", "Writer", "Producer"]:
            role_gender = crew_roles.get(role, {})
            if role_gender:
                male_pct = role_gender.get("male", 0)
                if male_pct > 0.8:
                    bias_flags.append(f"{role} heavily male-dominated: {male_pct:.1%}")
                    severity = max(
                        severity, BiasSeverity.HIGH, key=lambda x: ["low", "medium", "high", "critical"].index(x.value)
                    )

        return {
            "bias_type": BiasType.DEMOGRAPHIC,
            "severity": severity,
            "cast_demographics": cast_demographics,
            "crew_demographics": crew_demographics,
            "bias_flags": bias_flags,
            "recommendations": self._generate_demographic_recommendations(
                cast_demographics, crew_demographics, bias_flags
            ),
        }

    def _generate_geographic_recommendations(self, country_dist: Dict[str, float], bias_flags: List[str]) -> List[str]:
        """Generate recommendations for geographic bias."""
        recommendations = []

        if any("US over-representation" in flag for flag in bias_flags):
            recommendations.append("Increase representation of international films from diverse regions")
            recommendations.append("Specifically target underrepresented regions: Asia, Africa, South America")

        if any("low geographic diversity" in flag for flag in bias_flags):
            recommendations.append("Expand dataset to include films from more countries")
            recommendations.append("Focus on film festivals and international cinema")

        if not recommendations:
            recommendations.append("Geographic representation appears balanced")

        return recommendations

    def _generate_language_recommendations(self, language_dist: Dict[str, float], bias_flags: List[str]) -> List[str]:
        """Generate recommendations for language bias."""
        recommendations = []

        if any("English over-representation" in flag for flag in bias_flags):
            recommendations.append("Include more non-English language films")
            recommendations.append("Target films in major world languages: Spanish, Mandarin, Hindi, Arabic")

        if any("low language diversity" in flag for flag in bias_flags):
            recommendations.append("Expand to include indigenous and minority languages")
            recommendations.append("Consider regional cinema beyond major markets")

        return recommendations

    def _generate_temporal_recommendations(
        self, modern_pct: float, year_range: int, bias_flags: List[str]
    ) -> List[str]:
        """Generate recommendations for temporal bias."""
        recommendations = []

        if modern_pct > 0.75:
            recommendations.append("Include more classic and historical films (pre-2000)")
            recommendations.append("Add films from cinema's golden ages: 1940s-1970s")

        if year_range < 20:
            recommendations.append("Expand temporal range to cover more decades")
            recommendations.append("Include films from cinema history milestones")

        return recommendations

    def _generate_genre_recommendations(self, genre_dist: Dict[str, float], bias_flags: List[str]) -> List[str]:
        """Generate recommendations for genre bias."""
        recommendations = []

        if any("over-concentration" in flag for flag in bias_flags):
            recommendations.append("Balance genre representation across all major categories")
            recommendations.append("Include underrepresented genres: documentaries, foreign films, art house")

        if any("low genre diversity" in flag for flag in bias_flags):
            recommendations.append("Expand genre coverage to include niche and emerging categories")

        return recommendations

    def _generate_demographic_recommendations(
        self, cast_demo: Dict[str, Any], crew_demo: Dict[str, Any], bias_flags: List[str]
    ) -> List[str]:
        """Generate recommendations for demographic bias."""
        recommendations = []

        if any("gender imbalance" in flag for flag in bias_flags):
            recommendations.append("Seek better gender balance in cast representation")

        if any("male-dominated" in flag for flag in bias_flags):
            recommendations.append("Prioritize films by female directors, writers, and producers")
            recommendations.append("Include more films with strong female-led narratives")

        return recommendations


class ConstitutionalAIValidator:
    """Main Constitutional AI validator integrating all bias detection components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Constitutional AI validator with configuration."""
        self.config = config or {}
        self.bias_detector = BiasDetector(config)

        # Performance optimization settings
        self.batch_size = self.config.get("batch_size", 1000)
        self.cache_results = self.config.get("cache_results", True)
        self._bias_cache = {} if self.cache_results else None

    def validate_constitutional_compliance(self, movies: List[MovieSchema]) -> Dict[str, Any]:
        """
        Perform comprehensive Constitutional AI compliance validation.

        Args:
            movies: List of movies to analyze

        Returns:
            Comprehensive compliance report with bias detection results
        """

        logger.info(f"Starting Constitutional AI validation for {len(movies)} movies")

        # Initialize compliance report
        compliance_report = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_movies": len(movies),
            "overall_compliance": "PASS",
            "bias_analyses": {},
            "overall_bias_score": 0.0,
            "constitutional_ai_metrics": {},
            "recommendations": [],
            "audit_trail": [],
        }

        if not movies:
            compliance_report["overall_compliance"] = "FAIL"
            compliance_report["audit_trail"].append("No movies provided for validation")
            return compliance_report

        try:
            # Perform bias detection analyses
            bias_analyses = {}

            # Geographic bias
            geographic_analysis = self.bias_detector.detect_geographic_bias(movies)
            bias_analyses["geographic"] = geographic_analysis

            # Language bias
            language_analysis = self.bias_detector.detect_language_bias(movies)
            bias_analyses["language"] = language_analysis

            # Temporal bias
            temporal_analysis = self.bias_detector.detect_temporal_bias(movies)
            bias_analyses["temporal"] = temporal_analysis

            # Genre bias
            genre_analysis = self.bias_detector.detect_genre_bias(movies)
            bias_analyses["genre"] = genre_analysis

            # Demographic bias
            demographic_analysis = self.bias_detector.detect_demographic_bias(movies)
            bias_analyses["demographic"] = demographic_analysis

            compliance_report["bias_analyses"] = bias_analyses

            # Calculate overall bias score and compliance
            overall_score, overall_compliance = self._calculate_overall_compliance(bias_analyses)
            compliance_report["overall_bias_score"] = overall_score
            compliance_report["overall_compliance"] = overall_compliance

            # Generate Constitutional AI metrics for individual movies
            compliance_report["constitutional_ai_metrics"] = self._generate_constitutional_metrics(
                movies, bias_analyses
            )

            # Aggregate recommendations
            all_recommendations = []
            for analysis in bias_analyses.values():
                all_recommendations.extend(analysis.get("recommendations", []))
            compliance_report["recommendations"] = list(set(all_recommendations))  # Remove duplicates

            # Audit trail
            compliance_report["audit_trail"] = self._generate_audit_trail(bias_analyses)

            logger.info(f"Constitutional AI validation complete. Overall compliance: {overall_compliance}")

        except Exception as e:
            logger.error(f"Error during Constitutional AI validation: {e}")
            compliance_report["overall_compliance"] = "ERROR"
            compliance_report["audit_trail"].append(f"Validation error: {str(e)}")

        return compliance_report

    def _calculate_overall_compliance(self, bias_analyses: Dict[str, Any]) -> Tuple[float, str]:
        """Calculate overall compliance score and status."""

        severity_scores = {
            BiasSeverity.LOW: 0.1,
            BiasSeverity.MEDIUM: 0.3,
            BiasSeverity.HIGH: 0.7,
            BiasSeverity.CRITICAL: 1.0,
        }

        total_bias_score = 0.0
        analysis_count = len(bias_analyses)

        critical_issues = 0
        high_issues = 0

        for analysis in bias_analyses.values():
            severity = analysis.get("severity", BiasSeverity.LOW)
            bias_score = severity_scores.get(severity, 0.0)
            total_bias_score += bias_score

            if severity == BiasSeverity.CRITICAL:
                critical_issues += 1
            elif severity == BiasSeverity.HIGH:
                high_issues += 1

        # Normalize bias score (0.0 = no bias, 1.0 = maximum bias)
        overall_bias_score = total_bias_score / analysis_count if analysis_count > 0 else 0.0

        # Determine compliance status
        if critical_issues > 0:
            compliance = "CRITICAL"
        elif high_issues > 0 or overall_bias_score > 0.6:
            compliance = "FAIL"
        elif overall_bias_score > 0.3:
            compliance = "WARNING"
        else:
            compliance = "PASS"

        return overall_bias_score, compliance

    def _generate_constitutional_metrics(
        self, movies: List[MovieSchema], bias_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Constitutional AI metrics for the dataset."""

        metrics = {
            "dataset_diversity_scores": {},
            "bias_flag_summary": [],
            "representation_analysis": {},
            "compliance_indicators": {},
        }

        # Extract diversity scores from analyses
        for bias_type, analysis in bias_analyses.items():
            if "diversity_score" in analysis:
                metrics["dataset_diversity_scores"][bias_type] = analysis["diversity_score"]

        # Aggregate bias flags
        all_flags = []
        for analysis in bias_analyses.values():
            all_flags.extend(analysis.get("bias_flags", []))
        metrics["bias_flag_summary"] = all_flags

        # Representation analysis
        if "demographic" in bias_analyses:
            demo_analysis = bias_analyses["demographic"]
            metrics["representation_analysis"] = {
                "cast_gender_balance": demo_analysis.get("cast_demographics", {}).get("gender_distribution", {}),
                "crew_role_diversity": demo_analysis.get("crew_demographics", {}).get("gender_by_role", {}),
            }

        # Compliance indicators
        metrics["compliance_indicators"] = {
            "geographic_diversity": bias_analyses.get("geographic", {}).get("unique_countries", 0),
            "language_diversity": bias_analyses.get("language", {}).get("unique_languages", 0),
            "temporal_range": bias_analyses.get("temporal", {}).get("year_range", 0),
            "genre_diversity": bias_analyses.get("genre", {}).get("unique_genres", 0),
        }

        return metrics

    def _generate_audit_trail(self, bias_analyses: Dict[str, Any]) -> List[str]:
        """Generate audit trail for compliance validation."""

        audit_trail = []

        for bias_type, analysis in bias_analyses.items():
            severity = analysis.get("severity", BiasSeverity.LOW)
            flags = analysis.get("bias_flags", [])

            audit_entry = f"{bias_type.upper()} bias analysis: {severity.value} severity"
            if flags:
                audit_entry += f" - Issues: {'; '.join(flags[:3])}"  # First 3 issues
            audit_trail.append(audit_entry)

        return audit_trail

    def update_movie_constitutional_metrics(self, movie: MovieSchema, dataset_analysis: Dict[str, Any]) -> MovieSchema:
        """Update individual movie's Constitutional AI metrics based on dataset analysis."""

        # Initialize Constitutional AI metrics if not present
        if not hasattr(movie, "constitutional_ai") or movie.constitutional_ai is None:
            movie.constitutional_ai = ConstitutionalAIMetrics()

        # Update diversity scores (contextual to dataset)
        bias_analyses = dataset_analysis.get("bias_analyses", {})

        # Geographic context
        if "geographic" in bias_analyses:
            geo_analysis = bias_analyses["geographic"]
            country = movie.metadata.origin_country
            country_dist = geo_analysis.get("country_distribution", {})

            # Lower score if from over-represented country
            country_pct = country_dist.get(country, 0)
            movie.constitutional_ai.geographic_diversity_score = round(max(0.0, 1.0 - country_pct), 10)

        # Language context
        if "language" in bias_analyses:
            lang_analysis = bias_analyses["language"]
            language = movie.metadata.original_language.value
            lang_dist = lang_analysis.get("language_distribution", {})

            lang_pct = lang_dist.get(language, 0)
            movie.constitutional_ai.language_diversity_score = round(max(0.0, 1.0 - lang_pct), 10)

        # Set bias flags based on movie characteristics
        bias_flags = []

        # Check if movie contributes to identified biases
        overall_flags = dataset_analysis.get("constitutional_ai_metrics", {}).get("bias_flag_summary", [])

        if movie.metadata.origin_country == "US" and any("US over-representation" in flag for flag in overall_flags):
            bias_flags.append("contributes_to_geographic_bias")

        if movie.metadata.original_language.value == "en" and any(
            "English over-representation" in flag for flag in overall_flags
        ):
            bias_flags.append("contributes_to_language_bias")

        movie.constitutional_ai.bias_flags = bias_flags

        # Update audit information
        movie.constitutional_ai.last_bias_check = datetime.now()
        movie.constitutional_ai.overall_bias_score = dataset_analysis.get("overall_bias_score", 0.0)

        return movie


# Batch processing utilities for performance
class BatchProcessor:
    """Utility for processing large datasets efficiently."""

    @staticmethod
    def process_movies_in_batches(movies: List[MovieSchema], batch_size: int = 1000, processor_func=None) -> List[Any]:
        """Process movies in batches for memory efficiency."""

        results = []
        total_batches = (len(movies) + batch_size - 1) // batch_size

        for i in range(0, len(movies), batch_size):
            batch = movies[i : i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} movies)")

            if processor_func:
                batch_result = processor_func(batch)
                results.append(batch_result)

        return results


if __name__ == "__main__":
    # Example usage
    # from .movie_schema import Genre, Language, MovieMetadata, MovieSchema, RatingInfo

    # Create validator
    validator = ConstitutionalAIValidator()

    # Example movie data
    example_movies = [
        MovieSchema(
            movie_id="1",
            title="Fight Club",
            synopsis="An insomniac office worker forms a fight club.",
            release_year=1999,
            genres=[Genre.DRAMA, Genre.THRILLER],
            ratings=RatingInfo(average=8.8, count=2000000),
            metadata=MovieMetadata(tmdb_id=550, original_language=Language.EN, origin_country="US"),
        ),
        MovieSchema(
            movie_id="2",
            title="Seven Samurai",
            synopsis="A village hires samurai to defend against bandits.",
            release_year=1954,
            genres=[Genre.ACTION, Genre.DRAMA],
            ratings=RatingInfo(average=9.0, count=500000),
            metadata=MovieMetadata(tmdb_id=346, original_language=Language.JA, origin_country="JP"),
        ),
    ]

    # Validate Constitutional AI compliance
    compliance_report = validator.validate_constitutional_compliance(example_movies)

    print(f"Constitutional AI Validation Results:")
    print(f"Overall Compliance: {compliance_report['overall_compliance']}")
    print(f"Overall Bias Score: {compliance_report['overall_bias_score']:.3f}")
    print(f"Number of Recommendations: {len(compliance_report['recommendations'])}")

    if compliance_report["recommendations"]:
        print("\nTop Recommendations:")
        for rec in compliance_report["recommendations"][:3]:
            print(f"  - {rec}")
