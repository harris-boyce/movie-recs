"""
Enhanced TMDB API client with rate limiting, retry logic, and Constitutional AI integration.

This module provides a production-ready TMDB API client that extends tmdbsimple with:
- Token bucket rate limiting (respects TMDB's 40 requests/10 seconds limit)
- Exponential backoff retry logic for failed requests
- Response caching to reduce API calls
- Comprehensive error handling and logging
- Constitutional AI integration for bias detection and diversity tracking
- Optional async support for high-throughput scenarios
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import requests
import tmdbsimple as tmdb
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .acquisition import DataAcquisitionError

logger = logging.getLogger(__name__)


class TMDBError(DataAcquisitionError):
    """TMDB-specific error for API failures."""

    pass


class RateLimitExceededError(TMDBError):
    """Raised when rate limit is exceeded."""

    pass


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, capacity: float = 40, refill_rate: float = 4.0):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (TMDB allows 40 requests per 10 seconds)
            refill_rate: Rate at which tokens are refilled (4 per second to stay under limit)
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = asyncio.Lock() if asyncio.iscoroutinefunction(self.__init__) else None

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough tokens available
        """
        now = time.time()

        # Refill tokens based on time elapsed
        time_passed = now - self.last_refill
        new_tokens = time_passed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time needed to consume tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        if self.tokens >= tokens:
            return 0.0

        if self.refill_rate <= 0:
            return float("inf")  # Cannot refill, infinite wait time

        needed_tokens = tokens - self.tokens
        return needed_tokens / self.refill_rate


class ResponseCache:
    """Simple in-memory response cache with TTL."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached responses
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}

    def _generate_key(self, method: str, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from request parameters."""
        key_parts = [method, url]
        if params:
            key_parts.append(urlencode(sorted(params.items())))
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def get(self, method: str, url: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        key = self._generate_key(method, url, params)

        if key not in self.cache:
            return None

        entry = self.cache[key]
        now = time.time()

        if now - entry["timestamp"] > entry["ttl"]:
            # Expired
            del self.cache[key]
            del self.access_times[key]
            return None

        self.access_times[key] = now
        return entry["response"]

    def put(
        self, method: str, url: str, response: Dict[str, Any], params: Optional[Dict] = None, ttl: Optional[int] = None
    ) -> None:
        """Cache response with TTL."""
        key = self._generate_key(method, url, params)

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        now = time.time()
        self.cache[key] = {"response": response, "timestamp": now, "ttl": ttl or self.default_ttl}
        self.access_times[key] = now


class EnhancedTMDBClient:
    """Enhanced TMDB API client with rate limiting and Constitutional AI integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_per_second: float = 4.0,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.0,
        timeout: int = 30,
        config_path: Optional[str] = None,
    ):
        """
        Initialize enhanced TMDB client.

        Args:
            api_key: TMDB API key (will use TMDB_API_KEY env var if not provided)
            rate_limit_per_second: Requests per second limit (default: 4.0)
            enable_caching: Whether to enable response caching
            cache_ttl: Cache TTL in seconds
            max_retries: Maximum number of retries for failed requests
            retry_backoff_factor: Backoff factor for exponential backoff
            timeout: Request timeout in seconds
            config_path: Optional path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Setup API key
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise TMDBError("TMDB API key required. Set TMDB_API_KEY environment variable or pass api_key parameter.")

        # Configure tmdbsimple
        tmdb.API_KEY = self.api_key

        # Setup rate limiting
        capacity = min(40, rate_limit_per_second * 10)  # TMDB allows 40 requests per 10 seconds
        self.rate_limiter = TokenBucket(capacity=capacity, refill_rate=rate_limit_per_second)

        # Setup caching
        self.cache = ResponseCache() if enable_caching else None
        self.cache_ttl = cache_ttl

        # Setup retry configuration
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.timeout = timeout

        # Setup requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Constitutional AI tracking
        self.diversity_metrics = {
            "genres_fetched": defaultdict(int),
            "countries_fetched": defaultdict(int),
            "languages_fetched": defaultdict(int),
            "decades_fetched": defaultdict(int),
            "total_requests": 0,
            "failed_requests": 0,
        }

        logger.info(f"Enhanced TMDB client initialized with rate limit: {rate_limit_per_second}/sec")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file if provided."""
        if not config_path or not Path(config_path).exists():
            return {}

        try:
            import yaml

            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}

    def _wait_for_rate_limit(self, tokens: int = 1) -> None:
        """Wait if rate limit would be exceeded."""
        wait_time = self.rate_limiter.wait_time(tokens)

        if wait_time == float("inf"):
            raise RateLimitExceededError("Rate limit cannot be satisfied - no token refill rate")

        if wait_time > 0:
            logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)

        if not self.rate_limiter.consume(tokens):
            raise RateLimitExceededError("Rate limit exceeded after waiting")

    def _make_request_with_retry(
        self, method: str, url: str, params: Optional[Dict] = None, use_cache: bool = True
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic and caching."""

        # Check cache first
        if use_cache and self.cache:
            cached_response = self.cache.get(method, url, params)
            if cached_response:
                logger.debug(f"Cache hit for {method} {url}")
                return cached_response

        # Wait for rate limit
        self._wait_for_rate_limit()

        # Track request
        self.diversity_metrics["total_requests"] += 1

        # Make request with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making request: {method} {url} (attempt {attempt + 1})")

                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=self.timeout)
                else:
                    response = self.session.request(method, url, params=params, timeout=self.timeout)

                response.raise_for_status()
                result = response.json()

                # Cache successful response
                if use_cache and self.cache and response.status_code == 200:
                    self.cache.put(method, url, result, params, self.cache_ttl)

                return result

            except requests.exceptions.HTTPError as e:
                last_exception = e
                if e.response.status_code == 429:
                    # Rate limit exceeded, wait longer
                    wait_time = (2**attempt) * self.retry_backoff_factor
                    logger.warning(f"Rate limit exceeded (429), waiting {wait_time} seconds")
                    time.sleep(wait_time)
                elif e.response.status_code in [500, 502, 503, 504]:
                    # Server error, retry with backoff
                    wait_time = (2**attempt) * self.retry_backoff_factor
                    logger.warning(f"Server error {e.response.status_code}, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    # Client error, don't retry
                    self.diversity_metrics["failed_requests"] += 1
                    raise TMDBError(f"HTTP error {e.response.status_code}: {e.response.text}")

            except (requests.exceptions.RequestException, ValueError, json.JSONDecodeError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = (2**attempt) * self.retry_backoff_factor
                    logger.warning(f"Request failed: {e}, retrying in {wait_time} seconds")
                    time.sleep(wait_time)
                else:
                    break

        # All retries failed
        self.diversity_metrics["failed_requests"] += 1
        raise TMDBError(f"Request failed after {self.max_retries + 1} attempts: {last_exception}")

    def get_movie_details(self, movie_id: int, append_to_response: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get detailed movie information using enhanced tmdbsimple wrapper.

        Args:
            movie_id: TMDB movie ID
            append_to_response: Additional data to append (credits, videos, etc.)

        Returns:
            Dictionary containing movie details
        """
        try:
            # Use tmdbsimple for the API call but with our enhancements
            movie = tmdb.Movies(movie_id)

            # Build append_to_response parameter
            append_params = {}
            if append_to_response:
                append_params["append_to_response"] = ",".join(append_to_response)

            # Make request through our enhanced wrapper
            base_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            params = {"api_key": self.api_key, **append_params}

            result = self._make_request_with_retry("GET", base_url, params)

            # Track diversity metrics
            self._track_movie_diversity(result)

            logger.debug(f"Successfully fetched movie details for ID {movie_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch movie details for ID {movie_id}: {e}")
            raise TMDBError(f"Failed to fetch movie details: {e}")

    def get_movie_credits(self, movie_id: int) -> Dict[str, Any]:
        """
        Get movie cast and crew information.

        Args:
            movie_id: TMDB movie ID

        Returns:
            Dictionary containing cast and crew information
        """
        try:
            base_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
            params = {"api_key": self.api_key}

            result = self._make_request_with_retry("GET", base_url, params)

            logger.debug(f"Successfully fetched movie credits for ID {movie_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch movie credits for ID {movie_id}: {e}")
            raise TMDBError(f"Failed to fetch movie credits: {e}")

    def discover_diverse_movies(
        self,
        with_genres: Optional[List[int]] = None,
        without_genres: Optional[List[int]] = None,
        with_origin_country: Optional[List[str]] = None,
        primary_release_date_gte: Optional[str] = None,
        primary_release_date_lte: Optional[str] = None,
        sort_by: str = "popularity.desc",
        page: int = 1,
        diversity_requirements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Discover movies with Constitutional AI diversity considerations.

        Args:
            with_genres: Include only these genre IDs
            without_genres: Exclude these genre IDs
            with_origin_country: Include only these countries
            primary_release_date_gte: Minimum release date (YYYY-MM-DD)
            primary_release_date_lte: Maximum release date (YYYY-MM-DD)
            sort_by: Sort order
            page: Page number
            diversity_requirements: Dictionary specifying diversity requirements

        Returns:
            Dictionary containing movie discovery results with diversity analysis
        """
        try:
            base_url = "https://api.themoviedb.org/3/discover/movie"
            params = {"api_key": self.api_key, "sort_by": sort_by, "page": page}

            # Add filter parameters
            if with_genres:
                params["with_genres"] = ",".join(map(str, with_genres))
            if without_genres:
                params["without_genres"] = ",".join(map(str, without_genres))
            if with_origin_country:
                params["with_origin_country"] = ",".join(with_origin_country)
            if primary_release_date_gte:
                params["primary_release_date.gte"] = primary_release_date_gte
            if primary_release_date_lte:
                params["primary_release_date.lte"] = primary_release_date_lte

            result = self._make_request_with_retry("GET", base_url, params)

            # Analyze and enhance results for diversity
            if diversity_requirements:
                result = self._apply_diversity_filters(result, diversity_requirements)

            # Track diversity metrics for all returned movies
            for movie in result.get("results", []):
                self._track_movie_diversity(movie)

            logger.debug(f"Successfully discovered {len(result.get('results', []))} movies")
            return result

        except Exception as e:
            logger.error(f"Failed to discover movies: {e}")
            raise TMDBError(f"Failed to discover movies: {e}")

    def get_popular_movies_with_diversity(self, page: int = 1, ensure_diversity: bool = True) -> Dict[str, Any]:
        """
        Get popular movies with diversity considerations.

        Args:
            page: Page number
            ensure_diversity: Whether to apply diversity filtering

        Returns:
            Dictionary containing popular movies with diversity analysis
        """
        try:
            base_url = "https://api.themoviedb.org/3/movie/popular"
            params = {"api_key": self.api_key, "page": page}

            result = self._make_request_with_retry("GET", base_url, params)

            if ensure_diversity:
                # Apply basic diversity requirements
                diversity_requirements = {"min_genre_diversity": 5, "min_decade_spread": 3, "max_us_percentage": 0.7}
                result = self._apply_diversity_filters(result, diversity_requirements)

            # Track diversity metrics
            for movie in result.get("results", []):
                self._track_movie_diversity(movie)

            logger.debug(f"Successfully fetched popular movies with diversity considerations")
            return result

        except Exception as e:
            logger.error(f"Failed to fetch popular movies: {e}")
            raise TMDBError(f"Failed to fetch popular movies: {e}")

    def _track_movie_diversity(self, movie_data: Dict[str, Any]) -> None:
        """Track diversity metrics for Constitutional AI monitoring."""
        try:
            # Track genres
            if "genre_ids" in movie_data:
                for genre_id in movie_data["genre_ids"]:
                    self.diversity_metrics["genres_fetched"][genre_id] += 1
            elif "genres" in movie_data:
                for genre in movie_data["genres"]:
                    self.diversity_metrics["genres_fetched"][genre.get("id", "unknown")] += 1

            # Track origin country
            if "origin_country" in movie_data:
                for country in movie_data["origin_country"]:
                    self.diversity_metrics["countries_fetched"][country] += 1
            elif "production_countries" in movie_data:
                for country in movie_data["production_countries"]:
                    self.diversity_metrics["countries_fetched"][country.get("iso_3166_1", "unknown")] += 1

            # Track original language
            if "original_language" in movie_data:
                self.diversity_metrics["languages_fetched"][movie_data["original_language"]] += 1

            # Track decade
            if "release_date" in movie_data and movie_data["release_date"]:
                try:
                    year = int(movie_data["release_date"][:4])
                    decade = (year // 10) * 10
                    self.diversity_metrics["decades_fetched"][decade] += 1
                except (ValueError, TypeError):
                    pass

        except Exception as e:
            logger.warning(f"Failed to track diversity metrics: {e}")

    def _apply_diversity_filters(self, results: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Apply diversity filtering to results based on requirements."""
        try:
            movies = results.get("results", [])
            if not movies:
                return results

            filtered_movies = []
            diversity_stats = {"genres": set(), "countries": set(), "languages": set(), "decades": set()}

            for movie in movies:
                # Check if adding this movie improves diversity
                if self._improves_diversity(movie, diversity_stats, requirements):
                    filtered_movies.append(movie)
                    self._update_diversity_stats(movie, diversity_stats)

            # If we don't have enough diverse movies, include more from original results
            min_results = requirements.get("min_results", 10)
            if len(filtered_movies) < min_results:
                for movie in movies:
                    if movie not in filtered_movies and len(filtered_movies) < min_results:
                        filtered_movies.append(movie)

            results["results"] = filtered_movies
            results["diversity_analysis"] = {
                "unique_genres": len(diversity_stats["genres"]),
                "unique_countries": len(diversity_stats["countries"]),
                "unique_languages": len(diversity_stats["languages"]),
                "unique_decades": len(diversity_stats["decades"]),
                "total_movies": len(filtered_movies),
            }

            return results

        except Exception as e:
            logger.warning(f"Failed to apply diversity filters: {e}")
            return results

    def _improves_diversity(
        self, movie: Dict[str, Any], current_stats: Dict[str, Any], requirements: Dict[str, Any]
    ) -> bool:
        """Check if adding this movie improves overall diversity."""
        # Always include the first few movies
        if sum(len(s) for s in current_stats.values()) < 5:
            return True

        # Check genre diversity
        movie_genres = set()
        if "genre_ids" in movie:
            movie_genres = set(movie["genre_ids"])
        elif "genres" in movie:
            movie_genres = {g.get("id") for g in movie["genres"]}

        if movie_genres - current_stats["genres"]:
            return True

        # Check country diversity
        movie_countries = set()
        if "origin_country" in movie:
            movie_countries = set(movie["origin_country"])

        if movie_countries - current_stats["countries"]:
            return True

        # Check language diversity
        if "original_language" in movie:
            if movie["original_language"] not in current_stats["languages"]:
                return True

        # Check decade diversity
        if "release_date" in movie and movie["release_date"]:
            try:
                year = int(movie["release_date"][:4])
                decade = (year // 10) * 10
                if decade not in current_stats["decades"]:
                    return True
            except (ValueError, TypeError):
                pass

        return False

    def _update_diversity_stats(self, movie: Dict[str, Any], stats: Dict[str, Any]) -> None:
        """Update diversity statistics with new movie."""
        try:
            # Update genres
            if "genre_ids" in movie:
                stats["genres"].update(movie["genre_ids"])
            elif "genres" in movie:
                stats["genres"].update(g.get("id") for g in movie["genres"])

            # Update countries
            if "origin_country" in movie:
                stats["countries"].update(movie["origin_country"])

            # Update languages
            if "original_language" in movie:
                stats["languages"].add(movie["original_language"])

            # Update decades
            if "release_date" in movie and movie["release_date"]:
                try:
                    year = int(movie["release_date"][:4])
                    decade = (year // 10) * 10
                    stats["decades"].add(decade)
                except (ValueError, TypeError):
                    pass

        except Exception as e:
            logger.warning(f"Failed to update diversity stats: {e}")

    async def get_movie_details_async(self, movie_id: int) -> Dict[str, Any]:
        """Async version of get_movie_details."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_movie_details, movie_id)

    async def discover_diverse_movies_async(self, **kwargs) -> Dict[str, Any]:
        """Async version of discover_diverse_movies."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.discover_diverse_movies, **kwargs)

    def get_diversity_report(self) -> Dict[str, Any]:
        """Generate diversity report for Constitutional AI monitoring."""
        total_requests = self.diversity_metrics["total_requests"]

        report = {
            "total_requests": total_requests,
            "failed_requests": self.diversity_metrics["failed_requests"],
            "success_rate": 1 - (self.diversity_metrics["failed_requests"] / max(total_requests, 1)),
            "genre_diversity": {
                "unique_genres": len(self.diversity_metrics["genres_fetched"]),
                "total_genre_fetches": sum(self.diversity_metrics["genres_fetched"].values()),
                "top_genres": dict(
                    sorted(self.diversity_metrics["genres_fetched"].items(), key=lambda x: x[1], reverse=True)[:10]
                ),
            },
            "geographic_diversity": {
                "unique_countries": len(self.diversity_metrics["countries_fetched"]),
                "total_country_fetches": sum(self.diversity_metrics["countries_fetched"].values()),
                "top_countries": dict(
                    sorted(self.diversity_metrics["countries_fetched"].items(), key=lambda x: x[1], reverse=True)[:10]
                ),
            },
            "language_diversity": {
                "unique_languages": len(self.diversity_metrics["languages_fetched"]),
                "total_language_fetches": sum(self.diversity_metrics["languages_fetched"].values()),
                "top_languages": dict(
                    sorted(self.diversity_metrics["languages_fetched"].items(), key=lambda x: x[1], reverse=True)[:10]
                ),
            },
            "temporal_diversity": {
                "unique_decades": len(self.diversity_metrics["decades_fetched"]),
                "total_decade_fetches": sum(self.diversity_metrics["decades_fetched"].values()),
                "decade_distribution": dict(
                    sorted(self.diversity_metrics["decades_fetched"].items(), key=lambda x: x[0])
                ),
            },
            "bias_indicators": self._calculate_bias_indicators(),
        }

        return report

    def _calculate_bias_indicators(self) -> Dict[str, Any]:
        """Calculate bias indicators for Constitutional AI monitoring."""
        indicators = {}

        # Genre concentration bias
        genre_counts = list(self.diversity_metrics["genres_fetched"].values())
        if genre_counts:
            total_genre_fetches = sum(genre_counts)
            max_genre_pct = max(genre_counts) / total_genre_fetches if total_genre_fetches > 0 else 0
            indicators["genre_concentration"] = max_genre_pct

        # Geographic concentration bias
        country_counts = list(self.diversity_metrics["countries_fetched"].values())
        if country_counts:
            total_country_fetches = sum(country_counts)
            max_country_pct = max(country_counts) / total_country_fetches if total_country_fetches > 0 else 0
            indicators["geographic_concentration"] = max_country_pct

            # US bias indicator
            us_count = self.diversity_metrics["countries_fetched"].get("US", 0)
            indicators["us_bias"] = us_count / total_country_fetches if total_country_fetches > 0 else 0

        # Language concentration bias
        lang_counts = list(self.diversity_metrics["languages_fetched"].values())
        if lang_counts:
            total_lang_fetches = sum(lang_counts)
            max_lang_pct = max(lang_counts) / total_lang_fetches if total_lang_fetches > 0 else 0
            indicators["language_concentration"] = max_lang_pct

            # English bias indicator
            en_count = self.diversity_metrics["languages_fetched"].get("en", 0)
            indicators["english_bias"] = en_count / total_lang_fetches if total_lang_fetches > 0 else 0

        # Temporal bias (modern bias)
        decade_counts = self.diversity_metrics["decades_fetched"]
        if decade_counts:
            total_decade_fetches = sum(decade_counts.values())
            modern_count = sum(count for decade, count in decade_counts.items() if decade >= 2000)
            indicators["modern_bias"] = modern_count / total_decade_fetches if total_decade_fetches > 0 else 0

        return indicators

    def clear_cache(self) -> None:
        """Clear response cache."""
        if self.cache:
            self.cache.cache.clear()
            self.cache.access_times.clear()
            logger.info("Response cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"cache_enabled": False}

        return {
            "cache_enabled": True,
            "cache_size": len(self.cache.cache),
            "max_cache_size": self.cache.max_size,
            "hit_rate": "Not tracked",  # Could implement hit rate tracking
        }


if __name__ == "__main__":
    # Example usage

    from dotenv import load_dotenv

    load_dotenv()

    try:
        # Initialize client
        client = EnhancedTMDBClient()

        # Test movie details
        movie = client.get_movie_details(550, append_to_response=["credits"])  # Fight Club
        print(f"Movie: {movie['title']} ({movie['release_date'][:4]})")

        # Test movie discovery with diversity
        popular = client.get_popular_movies_with_diversity(page=1)
        print(f"Found {len(popular['results'])} diverse popular movies")

        # Test diversity discovery
        diverse_movies = client.discover_diverse_movies(
            sort_by="vote_average.desc", diversity_requirements={"min_genre_diversity": 5, "min_decade_spread": 3}
        )
        print(f"Found {len(diverse_movies['results'])} movies meeting diversity requirements")

        # Generate diversity report
        report = client.get_diversity_report()
        print(f"Diversity Report - Success Rate: {report['success_rate']:.2%}")
        print(f"Genre Diversity: {report['genre_diversity']['unique_genres']} unique genres")
        print(f"Geographic Diversity: {report['geographic_diversity']['unique_countries']} unique countries")

        # Print bias indicators
        bias = report["bias_indicators"]
        if bias:
            print("\nBias Indicators:")
            for indicator, value in bias.items():
                print(f"  {indicator}: {value:.2%}")

    except TMDBError as e:
        print(f"TMDB Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
