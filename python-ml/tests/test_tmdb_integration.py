"""
Integration tests for Enhanced TMDB Client against real TMDB API.

These tests require a valid TMDB API key and internet connection.
They test the actual integration with TMDB's API endpoints.

Note: These tests may be skipped in CI/CD environments without API keys.
"""

import os
import time
import unittest
from unittest.mock import patch

import pytest

from src.tmdb_client import EnhancedTMDBClient, TMDBError


@pytest.mark.integration
class TestTMDBIntegration(unittest.TestCase):
    """Integration tests requiring real TMDB API access."""
    
    @classmethod
    def setUpClass(cls):
        """Set up integration test class."""
        cls.api_key = os.getenv('TMDB_API_KEY')
        if not cls.api_key:
            pytest.skip("TMDB_API_KEY environment variable not set")
        
        # Use conservative rate limiting for real API
        cls.client = EnhancedTMDBClient(
            api_key=cls.api_key,
            rate_limit_per_second=2.0,  # Conservative for real API
            enable_caching=True,
            max_retries=3
        )
    
    def setUp(self):
        """Set up each test."""
        # Reset diversity metrics for clean test state
        self.client.diversity_metrics = {
            'genres_fetched': {},
            'countries_fetched': {},
            'languages_fetched': {},
            'decades_fetched': {},
            'total_requests': 0,
            'failed_requests': 0
        }
    
    def test_get_movie_details_real_api(self):
        """Test fetching real movie details from TMDB."""
        # Test with Fight Club (movie ID 550)
        movie_id = 550
        
        result = self.client.get_movie_details(movie_id)
        
        # Verify expected fields are present
        self.assertIn('id', result)
        self.assertIn('title', result)
        self.assertIn('overview', result)
        self.assertIn('release_date', result)
        self.assertIn('genres', result)
        self.assertIn('vote_average', result)
        
        # Verify correct movie
        self.assertEqual(result['id'], movie_id)
        self.assertEqual(result['title'], 'Fight Club')
        
        # Verify diversity tracking worked
        self.assertEqual(self.client.diversity_metrics['total_requests'], 1)
        self.assertEqual(self.client.diversity_metrics['failed_requests'], 0)
        self.assertGreater(len(self.client.diversity_metrics['genres_fetched']), 0)
    
    def test_get_movie_details_with_credits(self):
        """Test fetching movie details with additional data."""
        movie_id = 550  # Fight Club
        
        result = self.client.get_movie_details(
            movie_id,
            append_to_response=['credits', 'videos']
        )
        
        # Should have additional sections
        self.assertIn('credits', result)
        self.assertIn('videos', result)
        
        # Verify credits structure
        credits = result['credits']
        self.assertIn('cast', credits)
        self.assertIn('crew', credits)
        self.assertIsInstance(credits['cast'], list)
        self.assertIsInstance(credits['crew'], list)
    
    def test_get_movie_credits_real_api(self):
        """Test fetching real movie credits."""
        movie_id = 550  # Fight Club
        
        result = self.client.get_movie_credits(movie_id)
        
        # Verify structure
        self.assertIn('cast', result)
        self.assertIn('crew', result)
        self.assertIsInstance(result['cast'], list)
        self.assertIsInstance(result['crew'], list)
        
        # Verify cast has expected fields
        if result['cast']:
            cast_member = result['cast'][0]
            self.assertIn('name', cast_member)
            self.assertIn('character', cast_member)
            self.assertIn('profile_path', cast_member)
        
        # Verify crew has expected fields
        if result['crew']:
            crew_member = result['crew'][0]
            self.assertIn('name', crew_member)
            self.assertIn('job', crew_member)
    
    def test_discover_movies_real_api(self):
        """Test movie discovery with real API."""
        result = self.client.discover_diverse_movies(
            sort_by='popularity.desc',
            page=1
        )
        
        # Verify structure
        self.assertIn('results', result)
        self.assertIn('total_pages', result)
        self.assertIn('total_results', result)
        self.assertIsInstance(result['results'], list)
        
        # Verify we got movies
        self.assertGreater(len(result['results']), 0)
        
        # Verify movie structure
        if result['results']:
            movie = result['results'][0]
            self.assertIn('id', movie)
            self.assertIn('title', movie)
            self.assertIn('overview', movie)
            self.assertIn('genre_ids', movie)
            self.assertIn('original_language', movie)
            self.assertIn('release_date', movie)
    
    def test_discover_with_genre_filter(self):
        """Test discovery with genre filtering."""
        # Discover action movies (genre ID 28)
        result = self.client.discover_diverse_movies(
            with_genres=[28],  # Action
            sort_by='vote_average.desc',
            page=1
        )
        
        self.assertIn('results', result)
        self.assertGreater(len(result['results']), 0)
        
        # Verify all movies have action genre
        for movie in result['results']:
            self.assertIn(28, movie['genre_ids'])
    
    def test_discover_with_diversity_requirements(self):
        """Test discovery with diversity requirements."""
        result = self.client.discover_diverse_movies(
            sort_by='popularity.desc',
            page=1,
            diversity_requirements={
                'min_genre_diversity': 3,
                'min_decade_spread': 2
            }
        )
        
        # Should have diversity analysis
        self.assertIn('diversity_analysis', result)
        
        analysis = result['diversity_analysis']
        self.assertIn('unique_genres', analysis)
        self.assertIn('unique_countries', analysis)
        self.assertIn('unique_languages', analysis)
        self.assertIn('total_movies', analysis)
    
    def test_popular_movies_with_diversity(self):
        """Test getting popular movies with diversity considerations."""
        result = self.client.get_popular_movies_with_diversity(page=1)
        
        self.assertIn('results', result)
        self.assertGreater(len(result['results']), 0)
        
        # Should apply diversity filtering
        self.assertIn('diversity_analysis', result)
    
    def test_rate_limiting_compliance(self):
        """Test that rate limiting respects TMDB limits."""
        start_time = time.time()
        
        # Make multiple requests
        for i in range(5):
            self.client.get_movie_details(550 + i)  # Different movies
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should have taken at least some time due to rate limiting
        # With 2.0 requests/second, 5 requests should take at least 2 seconds
        expected_min_time = 4.0 / 2.0  # 4 requests need waiting time
        self.assertGreater(elapsed, expected_min_time * 0.8)  # Allow some tolerance
    
    def test_caching_functionality(self):
        """Test that response caching works with real API."""
        movie_id = 550
        
        # First request (should hit API)
        start_time = time.time()
        result1 = self.client.get_movie_details(movie_id)
        first_request_time = time.time() - start_time
        
        # Second request (should use cache)
        start_time = time.time()
        result2 = self.client.get_movie_details(movie_id)
        second_request_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(result1['id'], result2['id'])
        self.assertEqual(result1['title'], result2['title'])
        
        # Second request should be much faster (cached)
        self.assertLess(second_request_time, first_request_time * 0.5)
    
    def test_error_handling_invalid_movie_id(self):
        """Test error handling for invalid movie ID."""
        with self.assertRaises(TMDBError):
            self.client.get_movie_details(999999999)  # Non-existent movie
    
    def test_diversity_tracking_real_data(self):
        """Test diversity tracking with real TMDB data."""
        # Get diverse set of movies
        popular = self.client.get_popular_movies_with_diversity(page=1)
        
        # Should have tracked diversity metrics
        self.assertGreater(self.client.diversity_metrics['total_requests'], 0)
        self.assertGreater(len(self.client.diversity_metrics['genres_fetched']), 0)
        
        # Generate diversity report
        report = self.client.get_diversity_report()
        
        # Verify report structure
        self.assertIn('total_requests', report)
        self.assertIn('success_rate', report)
        self.assertIn('genre_diversity', report)
        self.assertIn('geographic_diversity', report)
        self.assertIn('language_diversity', report)
        self.assertIn('temporal_diversity', report)
        self.assertIn('bias_indicators', report)
        
        # Verify we have real diversity data
        self.assertGreater(report['genre_diversity']['unique_genres'], 1)
        self.assertGreater(report['success_rate'], 0.8)  # Should have high success rate
    
    def test_bias_detection_real_data(self):
        """Test bias detection with real TMDB data."""
        # Get movies from different sources to generate bias data
        popular = self.client.get_popular_movies_with_diversity(page=1)
        action_movies = self.client.discover_diverse_movies(with_genres=[28], page=1)
        drama_movies = self.client.discover_diverse_movies(with_genres=[18], page=1)
        
        report = self.client.get_diversity_report()
        bias_indicators = report['bias_indicators']
        
        # Should have calculated bias indicators
        self.assertIn('genre_concentration', bias_indicators)
        self.assertIn('geographic_concentration', bias_indicators)
        self.assertIn('us_bias', bias_indicators)
        self.assertIn('english_bias', bias_indicators)
        
        # Bias values should be between 0 and 1
        for indicator, value in bias_indicators.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_cache_stats_real_usage(self):
        """Test cache statistics with real usage."""
        # Make some requests to populate cache
        self.client.get_movie_details(550)
        self.client.get_movie_details(551)
        
        stats = self.client.get_cache_stats()
        
        self.assertTrue(stats['cache_enabled'])
        self.assertGreater(stats['cache_size'], 0)
        self.assertIsInstance(stats['max_cache_size'], int)
        
        # Clear cache and verify
        self.client.clear_cache()
        stats_after_clear = self.client.get_cache_stats()
        self.assertEqual(stats_after_clear['cache_size'], 0)
    
    def test_comprehensive_movie_data_extraction(self):
        """Test comprehensive extraction of movie data for ML pipeline."""
        movie_id = 550  # Fight Club
        
        # Get full movie details with all appendable data
        movie = self.client.get_movie_details(
            movie_id,
            append_to_response=[
                'credits', 'videos', 'images', 'keywords',
                'reviews', 'similar', 'translations'
            ]
        )
        
        # Verify we have comprehensive data suitable for ML pipeline
        required_fields = [
            'id', 'title', 'overview', 'release_date', 'genres',
            'vote_average', 'vote_count', 'popularity', 'runtime'
        ]
        
        for field in required_fields:
            self.assertIn(field, movie)
        
        # Verify additional data sections
        additional_sections = ['credits', 'videos', 'keywords']
        for section in additional_sections:
            if section in movie:  # Some sections might be empty
                self.assertIsInstance(movie[section], (dict, list))
        
        # Verify credits have cast and crew
        if 'credits' in movie:
            self.assertIn('cast', movie['credits'])
            self.assertIn('crew', movie['credits'])


@pytest.mark.integration
@pytest.mark.slow
class TestTMDBPerformance(unittest.TestCase):
    """Performance tests for TMDB integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up performance test class."""
        cls.api_key = os.getenv('TMDB_API_KEY')
        if not cls.api_key:
            pytest.skip("TMDB_API_KEY environment variable not set")
        
        cls.client = EnhancedTMDBClient(
            api_key=cls.api_key,
            rate_limit_per_second=3.0
        )
    
    def test_bulk_movie_fetching_performance(self):
        """Test performance of fetching multiple movies."""
        movie_ids = [550, 551, 552, 553, 554]  # 5 movies
        
        start_time = time.time()
        results = []
        
        for movie_id in movie_ids:
            try:
                movie = self.client.get_movie_details(movie_id)
                results.append(movie)
            except TMDBError:
                pass  # Some IDs might not exist
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 30.0)  # Less than 30 seconds for 5 movies
        self.assertGreater(len(results), 0)  # Should get at least some results
        
        print(f"Fetched {len(results)} movies in {elapsed:.2f} seconds")
    
    def test_diversity_analysis_performance(self):
        """Test performance of diversity analysis."""
        start_time = time.time()
        
        # Get movies with diversity requirements
        result = self.client.discover_diverse_movies(
            sort_by='popularity.desc',
            page=1,
            diversity_requirements={
                'min_genre_diversity': 5,
                'min_decade_spread': 3
            }
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete analysis in reasonable time
        self.assertLess(elapsed, 10.0)
        self.assertIn('diversity_analysis', result)
        
        print(f"Diversity analysis completed in {elapsed:.2f} seconds")


if __name__ == '__main__':
    # Run integration tests only if API key is available
    if os.getenv('TMDB_API_KEY'):
        unittest.main(verbosity=2)
    else:
        print("Skipping integration tests: TMDB_API_KEY not set")
        print("Set TMDB_API_KEY environment variable to run integration tests")
