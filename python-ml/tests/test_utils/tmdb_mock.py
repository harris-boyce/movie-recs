"""
Mock utilities for TMDB API testing.

Provides mock responses, fixtures, and utilities for consistent testing
of the Enhanced TMDB Client without requiring real API calls.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock

# Load test fixtures
FIXTURES_PATH = Path(__file__).parent.parent / 'fixtures' / 'tmdb_responses.json'

try:
    with open(FIXTURES_PATH, 'r') as f:
        MOCK_RESPONSES = json.load(f)
except FileNotFoundError:
    MOCK_RESPONSES = {}


class MockTMDBResponse:
    """Mock response object that mimics requests.Response."""
    
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        """Initialize mock response."""
        self.json_data = json_data
        self.status_code = status_code
        self.headers = {
            'content-type': 'application/json',
            'x-ratelimit-limit': '40',
            'x-ratelimit-remaining': '39',
            'x-ratelimit-reset': '1638360000'
        }
        self.text = json.dumps(json_data)
    
    def json(self):
        """Return JSON data."""
        return self.json_data
    
    def raise_for_status(self):
        """Raise HTTPError for bad status codes."""
        if self.status_code >= 400:
            from requests.exceptions import HTTPError
            response = Mock()
            response.status_code = self.status_code
            response.text = self.text
            raise HTTPError(response=response)


def get_mock_movie_details(movie_id: int) -> Dict[str, Any]:
    """Get mock movie details by ID."""
    movie_details = MOCK_RESPONSES.get('movie_details', {})
    return movie_details.get(str(movie_id), {
        'id': movie_id,
        'title': f'Mock Movie {movie_id}',
        'overview': f'This is a mock movie with ID {movie_id}',
        'release_date': '2020-01-01',
        'vote_average': 7.5,
        'vote_count': 1000,
        'popularity': 50.0,
        'genres': [{'id': 18, 'name': 'Drama'}],
        'origin_country': ['US'],
        'original_language': 'en'
    })


def get_mock_movie_credits(movie_id: int) -> Dict[str, Any]:
    """Get mock movie credits by ID."""
    movie_credits = MOCK_RESPONSES.get('movie_credits', {})
    return movie_credits.get(str(movie_id), {
        'id': movie_id,
        'cast': [
            {
                'id': 1,
                'name': 'Mock Actor 1',
                'character': 'Main Character',
                'order': 0,
                'profile_path': '/mock_profile.jpg',
                'gender': 2
            },
            {
                'id': 2,
                'name': 'Mock Actor 2',
                'character': 'Supporting Character',
                'order': 1,
                'profile_path': '/mock_profile2.jpg',
                'gender': 1
            }
        ],
        'crew': [
            {
                'id': 3,
                'name': 'Mock Director',
                'job': 'Director',
                'department': 'Directing',
                'profile_path': '/mock_director.jpg',
                'gender': 2
            },
            {
                'id': 4,
                'name': 'Mock Producer',
                'job': 'Producer',
                'department': 'Production',
                'profile_path': None,
                'gender': 1
            }
        ]
    })


def get_mock_discover_movies(discover_type: str = 'popular') -> Dict[str, Any]:
    """Get mock discover movies response."""
    discover_movies = MOCK_RESPONSES.get('discover_movies', {})
    return discover_movies.get(discover_type, {
        'page': 1,
        'total_pages': 100,
        'total_results': 2000,
        'results': [
            {
                'id': 1,
                'title': 'Mock Popular Movie 1',
                'overview': 'This is a mock popular movie',
                'release_date': '2022-01-01',
                'vote_average': 8.0,
                'vote_count': 5000,
                'popularity': 100.0,
                'genre_ids': [28, 12],  # Action, Adventure
                'origin_country': ['US'],
                'original_language': 'en',
                'poster_path': '/mock_poster1.jpg',
                'backdrop_path': '/mock_backdrop1.jpg',
                'adult': False,
                'video': False
            },
            {
                'id': 2,
                'title': 'Mock Popular Movie 2',
                'overview': 'This is another mock popular movie',
                'release_date': '2021-06-15',
                'vote_average': 7.5,
                'vote_count': 3000,
                'popularity': 85.0,
                'genre_ids': [18, 10749],  # Drama, Romance
                'origin_country': ['FR'],
                'original_language': 'fr',
                'poster_path': '/mock_poster2.jpg',
                'backdrop_path': '/mock_backdrop2.jpg',
                'adult': False,
                'video': False
            }
        ]
    })


def get_mock_error_response(error_type: str = '404') -> Dict[str, Any]:
    """Get mock error response."""
    errors = MOCK_RESPONSES.get('errors', {})
    return errors.get(error_type, {
        'status_code': 34,
        'status_message': 'The resource you requested could not be found.'
    })


class MockTMDBSession:
    """Mock session that simulates TMDB API responses."""
    
    def __init__(self, simulate_rate_limits: bool = False, simulate_errors: bool = False):
        """Initialize mock session."""
        self.simulate_rate_limits = simulate_rate_limits
        self.simulate_errors = simulate_errors
        self.request_count = 0
        self.rate_limit_threshold = 40
    
    def get(self, url: str, params: Optional[Dict] = None, **kwargs) -> MockTMDBResponse:
        """Mock GET request."""
        self.request_count += 1
        
        # Simulate rate limiting
        if self.simulate_rate_limits and self.request_count > self.rate_limit_threshold:
            return MockTMDBResponse(
                get_mock_error_response('429'),
                status_code=429
            )
        
        # Simulate random errors
        if self.simulate_errors and self.request_count % 10 == 0:
            return MockTMDBResponse(
                get_mock_error_response('500'),
                status_code=500
            )
        
        # Parse URL to determine response type
        if '/movie/' in url and '/credits' in url:
            # Movie credits endpoint
            movie_id = self._extract_movie_id(url)
            return MockTMDBResponse(get_mock_movie_credits(movie_id))
        
        elif '/movie/' in url:
            # Movie details endpoint
            movie_id = self._extract_movie_id(url)
            return MockTMDBResponse(get_mock_movie_details(movie_id))
        
        elif '/discover/movie' in url:
            # Discover movies endpoint
            if params and 'with_genres' in params:
                if '28' in str(params['with_genres']):  # Action genre
                    return MockTMDBResponse(get_mock_discover_movies('action_genre'))
            return MockTMDBResponse(get_mock_discover_movies('popular'))
        
        elif '/movie/popular' in url:
            # Popular movies endpoint
            return MockTMDBResponse(get_mock_discover_movies('popular'))
        
        else:
            # Default response
            return MockTMDBResponse({'message': 'Mock response for unknown endpoint'})
    
    def request(self, method: str, url: str, **kwargs) -> MockTMDBResponse:
        """Mock request method."""
        if method.upper() == 'GET':
            return self.get(url, **kwargs)
        else:
            return MockTMDBResponse({'message': f'Mock {method} response'})
    
    def _extract_movie_id(self, url: str) -> int:
        """Extract movie ID from URL."""
        parts = url.split('/')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 550  # Default to Fight Club ID


def create_diverse_mock_data() -> Dict[str, Any]:
    """Create diverse mock data for Constitutional AI testing."""
    return {
        'results': [
            {
                'id': 1,
                'title': 'US Action Movie',
                'genre_ids': [28],  # Action
                'origin_country': ['US'],
                'original_language': 'en',
                'release_date': '2022-01-01'
            },
            {
                'id': 2,
                'title': 'French Drama',
                'genre_ids': [18],  # Drama
                'origin_country': ['FR'],
                'original_language': 'fr',
                'release_date': '1995-05-15'
            },
            {
                'id': 3,
                'title': 'Japanese Animation',
                'genre_ids': [16],  # Animation
                'origin_country': ['JP'],
                'original_language': 'ja',
                'release_date': '2010-07-20'
            },
            {
                'id': 4,
                'title': 'Korean Comedy',
                'genre_ids': [35],  # Comedy
                'origin_country': ['KR'],
                'original_language': 'ko',
                'release_date': '2019-03-10'
            },
            {
                'id': 5,
                'title': 'Indian Romance',
                'genre_ids': [10749],  # Romance
                'origin_country': ['IN'],
                'original_language': 'hi',
                'release_date': '2000-12-08'
            },
            {
                'id': 6,
                'title': 'German Thriller',
                'genre_ids': [53],  # Thriller
                'origin_country': ['DE'],
                'original_language': 'de',
                'release_date': '1985-09-22'
            }
        ]
    }


def create_biased_mock_data() -> Dict[str, Any]:
    """Create biased mock data for bias detection testing."""
    return {
        'results': [
            {
                'id': 1,
                'title': 'US Action Movie 1',
                'genre_ids': [28],  # Action
                'origin_country': ['US'],
                'original_language': 'en',
                'release_date': '2022-01-01'
            },
            {
                'id': 2,
                'title': 'US Action Movie 2',
                'genre_ids': [28],  # Action
                'origin_country': ['US'],
                'original_language': 'en',
                'release_date': '2021-06-15'
            },
            {
                'id': 3,
                'title': 'US Action Movie 3',
                'genre_ids': [28],  # Action
                'origin_country': ['US'],
                'original_language': 'en',
                'release_date': '2020-11-30'
            },
            {
                'id': 4,
                'title': 'US Drama',
                'genre_ids': [18],  # Drama (some variety in genre)
                'origin_country': ['US'],
                'original_language': 'en',
                'release_date': '2023-03-15'
            },
            {
                'id': 5,
                'title': 'British Action',
                'genre_ids': [28],  # Action
                'origin_country': ['GB'],  # Some geographic variety
                'original_language': 'en',
                'release_date': '2019-08-20'
            }
        ]
    }


class MockTMDBClient:
    """Mock TMDB client for testing without real API calls."""
    
    def __init__(self, **kwargs):
        """Initialize mock client."""
        self.api_key = kwargs.get('api_key', 'mock_api_key')
        self.diversity_metrics = {
            'genres_fetched': {},
            'countries_fetched': {},
            'languages_fetched': {},
            'decades_fetched': {},
            'total_requests': 0,
            'failed_requests': 0
        }
        self.mock_session = MockTMDBSession(
            simulate_rate_limits=kwargs.get('simulate_rate_limits', False),
            simulate_errors=kwargs.get('simulate_errors', False)
        )
    
    def get_movie_details(self, movie_id: int, **kwargs) -> Dict[str, Any]:
        """Mock get movie details."""
        self.diversity_metrics['total_requests'] += 1
        return get_mock_movie_details(movie_id)
    
    def get_movie_credits(self, movie_id: int) -> Dict[str, Any]:
        """Mock get movie credits."""
        self.diversity_metrics['total_requests'] += 1
        return get_mock_movie_credits(movie_id)
    
    def discover_diverse_movies(self, **kwargs) -> Dict[str, Any]:
        """Mock discover diverse movies."""
        self.diversity_metrics['total_requests'] += 1
        
        if kwargs.get('bias_test_mode') == 'diverse':
            return create_diverse_mock_data()
        elif kwargs.get('bias_test_mode') == 'biased':
            return create_biased_mock_data()
        else:
            return get_mock_discover_movies('popular')
    
    def get_popular_movies_with_diversity(self, **kwargs) -> Dict[str, Any]:
        """Mock get popular movies with diversity."""
        self.diversity_metrics['total_requests'] += 1
        result = get_mock_discover_movies('popular')
        
        # Add mock diversity analysis
        result['diversity_analysis'] = {
            'unique_genres': 5,
            'unique_countries': 3,
            'unique_languages': 2,
            'unique_decades': 4,
            'total_movies': len(result.get('results', []))
        }
        
        return result


def patch_requests_for_tmdb(test_instance, mock_responses: Optional[Dict] = None):
    """
    Patch requests for TMDB testing.
    
    Usage:
        @patch_requests_for_tmdb
        def test_method(self, mock_session):
            # Test code here
    """
    def decorator(test_method):
        def wrapper(*args, **kwargs):
            with patch('requests.Session') as mock_session_class:
                mock_session = MockTMDBSession()
                if mock_responses:
                    # Update mock responses if provided
                    global MOCK_RESPONSES
                    MOCK_RESPONSES.update(mock_responses)
                
                mock_session_class.return_value = mock_session
                kwargs['mock_session'] = mock_session
                return test_method(*args, **kwargs)
        return wrapper
    return decorator


# Pytest fixtures for easy testing
try:
    import pytest
    
    @pytest.fixture
    def mock_tmdb_responses():
        """Pytest fixture for mock TMDB responses."""
        return MOCK_RESPONSES
    
    @pytest.fixture
    def mock_tmdb_session():
        """Pytest fixture for mock TMDB session."""
        return MockTMDBSession()
    
    @pytest.fixture
    def mock_tmdb_client():
        """Pytest fixture for mock TMDB client."""
        return MockTMDBClient()
    
    @pytest.fixture
    def diverse_movie_data():
        """Pytest fixture for diverse movie data."""
        return create_diverse_mock_data()
    
    @pytest.fixture
    def biased_movie_data():
        """Pytest fixture for biased movie data."""
        return create_biased_mock_data()

except ImportError:
    # Pytest not available, skip fixtures
    pass


if __name__ == '__main__':
    # Example usage
    print("Mock TMDB Utilities")
    print("===================")
    
    # Show mock movie details
    movie = get_mock_movie_details(550)
    print(f"Mock movie: {movie['title']}")
    
    # Show mock discover results
    discover = get_mock_discover_movies('popular')
    print(f"Mock discover: {len(discover['results'])} movies")
    
    # Show diverse data
    diverse = create_diverse_mock_data()
    print(f"Diverse data: {len(diverse['results'])} movies from different countries")
    
    countries = {movie['origin_country'][0] for movie in diverse['results']}
    print(f"Countries represented: {sorted(countries)}")