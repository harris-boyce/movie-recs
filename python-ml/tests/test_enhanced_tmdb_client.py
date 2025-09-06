"""
Comprehensive test suite for Enhanced TMDB Client.

Tests cover:
- Rate limiting functionality
- Retry logic and error handling
- Response caching
- Constitutional AI integration
- Mock TMDB responses for consistent testing
"""

import json
import time
import unittest
from unittest.mock import MagicMock, Mock, patch
from datetime import datetime, timedelta

import pytest
import requests
from requests.exceptions import HTTPError, RequestException

from src.tmdb_client import (
    EnhancedTMDBClient,
    RateLimitExceededError,
    ResponseCache,
    TMDBError,
    TokenBucket,
)


class TestTokenBucket(unittest.TestCase):
    """Test token bucket rate limiting implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.bucket = TokenBucket(capacity=10, refill_rate=5.0)
    
    def test_initial_capacity(self):
        """Test that bucket starts at full capacity."""
        self.assertEqual(self.bucket.tokens, 10)
        self.assertTrue(self.bucket.consume(10))
        self.assertEqual(self.bucket.tokens, 0)
    
    def test_token_consumption(self):
        """Test basic token consumption."""
        self.assertTrue(self.bucket.consume(5))
        self.assertAlmostEqual(self.bucket.tokens, 5, places=1)
        self.assertTrue(self.bucket.consume(5))
        self.assertAlmostEqual(self.bucket.tokens, 0, places=1)
        self.assertFalse(self.bucket.consume(1))
    
    def test_token_refill(self):
        """Test that tokens refill at correct rate."""
        # Consume all tokens
        self.bucket.consume(10)
        self.assertEqual(self.bucket.tokens, 0)
        
        # Wait for refill (mock time passage)
        original_time = self.bucket.last_refill
        self.bucket.last_refill = original_time - 2.0  # 2 seconds ago
        
        # Should refill 10 tokens (5 per second * 2 seconds)
        self.assertTrue(self.bucket.consume(10))
    
    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        self.bucket.consume(10)  # Empty bucket
        
        # Need 5 tokens, refill rate is 5/second, so should wait 1 second
        wait_time = self.bucket.wait_time(5)
        self.assertEqual(wait_time, 1.0)
    
    def test_capacity_limit(self):
        """Test that bucket doesn't exceed capacity."""
        # Start with empty bucket
        self.bucket.tokens = 0
        self.bucket.last_refill = time.time() - 100  # Long time ago
        
        # Should refill to capacity, not more
        self.assertTrue(self.bucket.consume(10))
        self.assertEqual(self.bucket.tokens, 0)


class TestResponseCache(unittest.TestCase):
    """Test response cache functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = ResponseCache(max_size=3, default_ttl=3600)
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        result = self.cache.get('GET', '/test', {'param': 'value'})
        self.assertIsNone(result)
    
    def test_cache_hit(self):
        """Test cache hit returns stored response."""
        response = {'data': 'test'}
        self.cache.put('GET', '/test', response, {'param': 'value'})
        
        result = self.cache.get('GET', '/test', {'param': 'value'})
        self.assertEqual(result, response)
    
    def test_cache_expiry(self):
        """Test that expired entries are removed."""
        response = {'data': 'test'}
        
        # Mock time to make entry expired
        with patch('time.time', return_value=0):
            self.cache.put('GET', '/test', response, ttl=1)
        
        with patch('time.time', return_value=2):  # 2 seconds later
            result = self.cache.get('GET', '/test')
            self.assertIsNone(result)
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(3):
            self.cache.put('GET', f'/test{i}', {'data': i})
        
        # Access first entry to make it recently used
        self.cache.get('GET', '/test0')
        
        # Add new entry, should evict least recently used (/test1)
        self.cache.put('GET', '/test3', {'data': 3})
        
        self.assertIsNotNone(self.cache.get('GET', '/test0'))  # Should still exist
        self.assertIsNone(self.cache.get('GET', '/test1'))     # Should be evicted
        self.assertIsNotNone(self.cache.get('GET', '/test2'))  # Should still exist
        self.assertIsNotNone(self.cache.get('GET', '/test3'))  # Should be added
    
    def test_key_generation(self):
        """Test that cache keys are generated consistently."""
        key1 = self.cache._generate_key('GET', '/test', {'a': '1', 'b': '2'})
        key2 = self.cache._generate_key('GET', '/test', {'b': '2', 'a': '1'})
        self.assertEqual(key1, key2)  # Order shouldn't matter


class TestEnhancedTMDBClient(unittest.TestCase):
    """Test Enhanced TMDB Client functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = 'test_api_key'
        
        # Mock environment variable
        with patch.dict('os.environ', {'TMDB_API_KEY': self.api_key}):
            self.client = EnhancedTMDBClient(
                rate_limit_per_second=10.0,  # Higher limit for testing
                enable_caching=True,
                max_retries=2
            )
    
    @patch('src.tmdb_client.requests.Session.get')
    def test_successful_request(self, mock_get):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'title': 'Test Movie', 'id': 123}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client._make_request_with_retry('GET', 'http://test.com')
        
        self.assertEqual(result['title'], 'Test Movie')
        self.assertEqual(result['id'], 123)
        mock_get.assert_called_once()
    
    @patch('src.tmdb_client.requests.Session.get')
    def test_rate_limiting(self, mock_get):
        """Test that rate limiting works correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Exhaust rate limit
        self.client.rate_limiter.tokens = 0
        
        # Should not raise exception but should work after refill
        start_time = time.time()
        result = self.client._make_request_with_retry('GET', 'http://test.com')
        end_time = time.time()
        
        # Should have waited for rate limit
        self.assertGreater(end_time - start_time, 0)
        self.assertEqual(result['data'], 'test')
    
    @patch('src.tmdb_client.requests.Session.get')
    def test_retry_on_server_error(self, mock_get):
        """Test retry logic on server errors."""
        # First call returns 503, second succeeds
        error_response = Mock()
        error_response.status_code = 503
        error_response.raise_for_status.side_effect = HTTPError(response=error_response)
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {'data': 'success'}
        success_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [HTTPError(response=error_response), success_response]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = self.client._make_request_with_retry('GET', 'http://test.com')
        
        self.assertEqual(result['data'], 'success')
        self.assertEqual(mock_get.call_count, 2)
    
    @patch('src.tmdb_client.requests.Session.get')
    def test_retry_exhaustion(self, mock_get):
        """Test that retries eventually give up."""
        error_response = Mock()
        error_response.status_code = 500
        error_response.text = 'Internal Server Error'
        error_response.raise_for_status.side_effect = HTTPError(response=error_response)
        
        mock_get.side_effect = HTTPError(response=error_response)
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with self.assertRaises(TMDBError):
                self.client._make_request_with_retry('GET', 'http://test.com')
        
        # Should have tried max_retries + 1 times
        self.assertEqual(mock_get.call_count, 3)
    
    @patch('src.tmdb_client.requests.Session.get')
    def test_caching_behavior(self, mock_get):
        """Test response caching."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'cached': 'data'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        url = 'http://test.com'
        params = {'param': 'value'}
        
        # First call should hit API
        result1 = self.client._make_request_with_retry('GET', url, params)
        self.assertEqual(mock_get.call_count, 1)
        
        # Second call should use cache
        result2 = self.client._make_request_with_retry('GET', url, params)
        self.assertEqual(mock_get.call_count, 1)  # No additional API call
        
        self.assertEqual(result1, result2)
    
    @patch('src.tmdb_client.requests.Session.get')
    def test_get_movie_details(self, mock_get):
        """Test get_movie_details method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 550,
            'title': 'Fight Club',
            'release_date': '1999-10-15',
            'genres': [{'id': 18, 'name': 'Drama'}],
            'origin_country': ['US'],
            'original_language': 'en'
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.get_movie_details(550)
        
        self.assertEqual(result['title'], 'Fight Club')
        self.assertEqual(result['id'], 550)
        # Check that diversity tracking was called
        self.assertGreater(self.client.diversity_metrics['total_requests'], 0)
    
    @patch('src.tmdb_client.requests.Session.get')
    def test_discover_diverse_movies(self, mock_get):
        """Test discover_diverse_movies method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {
                    'id': 1,
                    'title': 'Movie 1',
                    'genre_ids': [18, 35],  # Drama, Comedy
                    'origin_country': ['US'],
                    'original_language': 'en',
                    'release_date': '2020-01-01'
                },
                {
                    'id': 2,
                    'title': 'Movie 2',
                    'genre_ids': [28, 12],  # Action, Adventure
                    'origin_country': ['FR'],
                    'original_language': 'fr',
                    'release_date': '1985-06-15'
                }
            ],
            'total_pages': 1,
            'total_results': 2
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.client.discover_diverse_movies(
            with_genres=[18, 35],
            diversity_requirements={'min_genre_diversity': 2}
        )
        
        self.assertEqual(len(result['results']), 2)
        self.assertIn('diversity_analysis', result)
        
        # Check diversity tracking
        self.assertGreater(len(self.client.diversity_metrics['genres_fetched']), 0)
        self.assertGreater(len(self.client.diversity_metrics['countries_fetched']), 0)
    
    def test_diversity_tracking(self):
        """Test Constitutional AI diversity tracking."""
        movie_data = {
            'id': 123,
            'title': 'Test Movie',
            'genre_ids': [18, 35],  # Drama, Comedy
            'origin_country': ['US'],
            'original_language': 'en',
            'release_date': '2020-05-15'
        }
        
        self.client._track_movie_diversity(movie_data)
        
        # Check that metrics were updated
        self.assertEqual(self.client.diversity_metrics['genres_fetched'][18], 1)
        self.assertEqual(self.client.diversity_metrics['genres_fetched'][35], 1)
        self.assertEqual(self.client.diversity_metrics['countries_fetched']['US'], 1)
        self.assertEqual(self.client.diversity_metrics['languages_fetched']['en'], 1)
        self.assertEqual(self.client.diversity_metrics['decades_fetched'][2020], 1)
    
    def test_diversity_report_generation(self):
        """Test diversity report generation."""
        # Add some mock data
        self.client.diversity_metrics['total_requests'] = 10
        self.client.diversity_metrics['failed_requests'] = 1
        self.client.diversity_metrics['genres_fetched'][18] = 5  # Drama
        self.client.diversity_metrics['genres_fetched'][35] = 3  # Comedy
        self.client.diversity_metrics['countries_fetched']['US'] = 6
        self.client.diversity_metrics['countries_fetched']['FR'] = 2
        self.client.diversity_metrics['languages_fetched']['en'] = 7
        self.client.diversity_metrics['languages_fetched']['fr'] = 1
        
        report = self.client.get_diversity_report()
        
        self.assertEqual(report['total_requests'], 10)
        self.assertEqual(report['success_rate'], 0.9)
        self.assertEqual(report['genre_diversity']['unique_genres'], 2)
        self.assertEqual(report['geographic_diversity']['unique_countries'], 2)
        
        # Check bias indicators
        bias = report['bias_indicators']
        self.assertIn('genre_concentration', bias)
        self.assertIn('us_bias', bias)
        self.assertIn('english_bias', bias)
    
    def test_bias_indicators_calculation(self):
        """Test bias indicators calculation."""
        # Setup biased data
        self.client.diversity_metrics['genres_fetched'][18] = 80  # Drama dominant
        self.client.diversity_metrics['genres_fetched'][35] = 20  # Comedy
        self.client.diversity_metrics['countries_fetched']['US'] = 90  # US dominant
        self.client.diversity_metrics['countries_fetched']['FR'] = 10
        self.client.diversity_metrics['languages_fetched']['en'] = 85  # English dominant
        self.client.diversity_metrics['languages_fetched']['fr'] = 15
        self.client.diversity_metrics['decades_fetched'][2020] = 70  # Modern bias
        self.client.diversity_metrics['decades_fetched'][1990] = 30
        
        bias = self.client._calculate_bias_indicators()
        
        self.assertAlmostEqual(bias['genre_concentration'], 0.8, places=2)
        self.assertAlmostEqual(bias['geographic_concentration'], 0.9, places=2)
        self.assertAlmostEqual(bias['us_bias'], 0.9, places=2)
        self.assertAlmostEqual(bias['language_concentration'], 0.85, places=2)
        self.assertAlmostEqual(bias['english_bias'], 0.85, places=2)
        self.assertAlmostEqual(bias['modern_bias'], 0.7, places=2)
    
    def test_cache_management(self):
        """Test cache management methods."""
        # Add something to cache
        self.client.cache.put('GET', '/test', {'data': 'test'})
        self.assertEqual(len(self.client.cache.cache), 1)
        
        # Clear cache
        self.client.clear_cache()
        self.assertEqual(len(self.client.cache.cache), 0)
        
        # Check cache stats
        stats = self.client.get_cache_stats()
        self.assertTrue(stats['cache_enabled'])
        self.assertEqual(stats['cache_size'], 0)
    
    def test_error_handling(self):
        """Test various error conditions."""
        # Test missing API key
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(TMDBError):
                EnhancedTMDBClient()
        
        # Test rate limit exceeded
        self.client.rate_limiter.tokens = 0
        self.client.rate_limiter.refill_rate = 0  # No refill
        
        # Should raise RateLimitExceededError when wait time is infinite
        with self.assertRaises(RateLimitExceededError):
            self.client._wait_for_rate_limit()


@pytest.mark.asyncio
async def test_async_methods():
    """Test async method functionality."""
    api_key = 'test_api_key'
    
    with patch.dict('os.environ', {'TMDB_API_KEY': api_key}):
        client = EnhancedTMDBClient()
    
    with patch.object(client, 'get_movie_details', return_value={'id': 123}):
        result = await client.get_movie_details_async(123)
        assert result['id'] == 123
    
    with patch.object(client, 'discover_diverse_movies', return_value={'results': []}):
        result = await client.discover_diverse_movies_async()
        assert 'results' in result


class TestDiversityFiltering(unittest.TestCase):
    """Test diversity filtering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = 'test_api_key'
        
        with patch.dict('os.environ', {'TMDB_API_KEY': self.api_key}):
            self.client = EnhancedTMDBClient()
    
    def test_diversity_improvement_detection(self):
        """Test detection of diversity improvements."""
        current_stats = {
            'genres': {18, 35},      # Drama, Comedy
            'countries': {'US'},
            'languages': {'en'},
            'decades': {2020}
        }
        
        # Movie with new genre should improve diversity
        movie_new_genre = {
            'genre_ids': [28],  # Action
            'origin_country': ['US'],
            'original_language': 'en',
            'release_date': '2020-01-01'
        }
        
        self.assertTrue(
            self.client._improves_diversity(movie_new_genre, current_stats, {})
        )
        
        # Movie with same genres should not improve diversity (much)
        movie_same = {
            'genre_ids': [18, 35],  # Drama, Comedy
            'origin_country': ['US'],
            'original_language': 'en',
            'release_date': '2020-01-01'
        }
        
        self.assertFalse(
            self.client._improves_diversity(movie_same, current_stats, {})
        )
    
    def test_diversity_stats_update(self):
        """Test diversity statistics updates."""
        stats = {
            'genres': set(),
            'countries': set(),
            'languages': set(),
            'decades': set()
        }
        
        movie = {
            'genre_ids': [18, 35],
            'origin_country': ['US', 'CA'],
            'original_language': 'en',
            'release_date': '2020-05-15'
        }
        
        self.client._update_diversity_stats(movie, stats)
        
        self.assertEqual(stats['genres'], {18, 35})
        self.assertEqual(stats['countries'], {'US', 'CA'})
        self.assertEqual(stats['languages'], {'en'})
        self.assertEqual(stats['decades'], {2020})
    
    def test_diversity_filter_application(self):
        """Test application of diversity filters to results."""
        results = {
            'results': [
                {
                    'id': 1,
                    'genre_ids': [18],  # Drama
                    'origin_country': ['US'],
                    'original_language': 'en',
                    'release_date': '2020-01-01'
                },
                {
                    'id': 2,
                    'genre_ids': [18],  # Drama (same genre)
                    'origin_country': ['US'],
                    'original_language': 'en',
                    'release_date': '2020-01-01'
                },
                {
                    'id': 3,
                    'genre_ids': [35],  # Comedy (different genre)
                    'origin_country': ['FR'],
                    'original_language': 'fr',
                    'release_date': '1990-01-01'
                }
            ]
        }
        
        requirements = {
            'min_genre_diversity': 2,
            'min_results': 2
        }
        
        filtered = self.client._apply_diversity_filters(results, requirements)
        
        # Should prefer movies that add diversity
        self.assertGreater(len(filtered['results']), 0)
        self.assertIn('diversity_analysis', filtered)
        
        # Check that analysis was added
        analysis = filtered['diversity_analysis']
        self.assertIn('unique_genres', analysis)
        self.assertIn('unique_countries', analysis)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
