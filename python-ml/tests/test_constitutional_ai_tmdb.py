"""
Tests for Constitutional AI integration with TMDB client.

Tests cover:
- Bias detection in movie data
- Diversity tracking and analysis
- Constitutional AI compliance monitoring
- Representation analysis
"""

import unittest
from unittest.mock import Mock, patch
from collections import defaultdict

from src.tmdb_client import EnhancedTMDBClient


class TestConstitutionalAIIntegration(unittest.TestCase):
    """Test Constitutional AI features of TMDB client."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = 'test_api_key'
        
        with patch.dict('os.environ', {'TMDB_API_KEY': self.api_key}):
            self.client = EnhancedTMDBClient()
    
    def test_genre_bias_detection(self):
        """Test detection of genre bias in fetched data."""
        # Simulate heavily biased towards action movies
        self.client.diversity_metrics['genres_fetched'][28] = 80  # Action
        self.client.diversity_metrics['genres_fetched'][18] = 15  # Drama
        self.client.diversity_metrics['genres_fetched'][35] = 5   # Comedy
        
        bias_indicators = self.client._calculate_bias_indicators()
        
        # Should detect high genre concentration
        self.assertGreater(bias_indicators['genre_concentration'], 0.7)
    
    def test_geographic_bias_detection(self):
        """Test detection of geographic bias."""
        # Simulate US-heavy bias
        self.client.diversity_metrics['countries_fetched']['US'] = 85
        self.client.diversity_metrics['countries_fetched']['GB'] = 10
        self.client.diversity_metrics['countries_fetched']['FR'] = 5
        
        bias_indicators = self.client._calculate_bias_indicators()
        
        # Should detect high US bias
        self.assertGreater(bias_indicators['us_bias'], 0.8)
        self.assertGreater(bias_indicators['geographic_concentration'], 0.8)
    
    def test_language_bias_detection(self):
        """Test detection of language bias."""
        # Simulate English-heavy bias
        self.client.diversity_metrics['languages_fetched']['en'] = 90
        self.client.diversity_metrics['languages_fetched']['fr'] = 5
        self.client.diversity_metrics['languages_fetched']['es'] = 3
        self.client.diversity_metrics['languages_fetched']['de'] = 2
        
        bias_indicators = self.client._calculate_bias_indicators()
        
        # Should detect high English bias
        self.assertGreater(bias_indicators['english_bias'], 0.85)
        self.assertGreater(bias_indicators['language_concentration'], 0.85)
    
    def test_temporal_bias_detection(self):
        """Test detection of temporal bias towards modern movies."""
        # Simulate modern bias
        self.client.diversity_metrics['decades_fetched'][2020] = 40
        self.client.diversity_metrics['decades_fetched'][2010] = 35
        self.client.diversity_metrics['decades_fetched'][2000] = 15
        self.client.diversity_metrics['decades_fetched'][1990] = 8
        self.client.diversity_metrics['decades_fetched'][1980] = 2
        
        bias_indicators = self.client._calculate_bias_indicators()
        
        # Should detect high modern bias (movies from 2000+)
        self.assertGreater(bias_indicators['modern_bias'], 0.8)
    
    def test_balanced_diversity_metrics(self):
        """Test metrics calculation with balanced, diverse data."""
        # Simulate balanced data
        genres = [18, 28, 35, 27, 53, 12, 16, 80, 99, 878]  # Various genres
        for genre in genres:
            self.client.diversity_metrics['genres_fetched'][genre] = 10
        
        countries = ['US', 'GB', 'FR', 'DE', 'JP', 'KR', 'IN', 'BR', 'MX', 'CA']
        for country in countries:
            self.client.diversity_metrics['countries_fetched'][country] = 10
        
        languages = ['en', 'fr', 'es', 'de', 'ja', 'ko', 'hi', 'pt', 'it', 'ru']
        for lang in languages:
            self.client.diversity_metrics['languages_fetched'][lang] = 10
        
        decades = [1970, 1980, 1990, 2000, 2010, 2020]
        for decade in decades:
            self.client.diversity_metrics['decades_fetched'][decade] = 15
        
        bias_indicators = self.client._calculate_bias_indicators()
        
        # Should show low bias across all dimensions
        self.assertLess(bias_indicators['genre_concentration'], 0.2)
        self.assertLess(bias_indicators['geographic_concentration'], 0.2)
        self.assertLess(bias_indicators['us_bias'], 0.2)
        self.assertLess(bias_indicators['language_concentration'], 0.2)
        self.assertLess(bias_indicators['english_bias'], 0.2)
        self.assertLess(bias_indicators['modern_bias'], 0.7)  # Some modern bias expected
    
    def test_diversity_tracking_movie_data(self):
        """Test diversity tracking with various movie data formats."""
        # Test with genre_ids format (from discover API)
        movie1 = {
            'id': 1,
            'genre_ids': [18, 35],
            'origin_country': ['US'],
            'original_language': 'en',
            'release_date': '2020-05-15'
        }
        
        self.client._track_movie_diversity(movie1)
        
        self.assertEqual(self.client.diversity_metrics['genres_fetched'][18], 1)
        self.assertEqual(self.client.diversity_metrics['genres_fetched'][35], 1)
        self.assertEqual(self.client.diversity_metrics['countries_fetched']['US'], 1)
        self.assertEqual(self.client.diversity_metrics['languages_fetched']['en'], 1)
        self.assertEqual(self.client.diversity_metrics['decades_fetched'][2020], 1)
        
        # Test with genres object format (from movie details API)
        movie2 = {
            'id': 2,
            'genres': [{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}],
            'production_countries': [{'iso_3166_1': 'FR', 'name': 'France'}],
            'original_language': 'fr',
            'release_date': '1985-06-20'
        }
        
        self.client._track_movie_diversity(movie2)
        
        self.assertEqual(self.client.diversity_metrics['genres_fetched'][28], 1)
        self.assertEqual(self.client.diversity_metrics['genres_fetched'][12], 1)
        self.assertEqual(self.client.diversity_metrics['countries_fetched']['FR'], 1)
        self.assertEqual(self.client.diversity_metrics['languages_fetched']['fr'], 1)
        self.assertEqual(self.client.diversity_metrics['decades_fetched'][1980], 1)
    
    def test_diversity_improvement_algorithm(self):
        """Test algorithm that determines if movie improves diversity."""
        current_stats = {
            'genres': {18, 35},      # Drama, Comedy
            'countries': {'US'},
            'languages': {'en'},
            'decades': {2020}
        }
        
        requirements = {'min_genre_diversity': 3}
        
        # Movie that adds new genre should improve diversity
        movie_new_genre = {
            'genre_ids': [28],  # Action - new genre
            'origin_country': ['US'],
            'original_language': 'en',
            'release_date': '2020-01-01'
        }
        
        self.assertTrue(
            self.client._improves_diversity(movie_new_genre, current_stats, requirements)
        )
        
        # Movie that adds new country should improve diversity
        movie_new_country = {
            'genre_ids': [18],  # Drama - existing genre
            'origin_country': ['FR'],  # France - new country
            'original_language': 'en',
            'release_date': '2020-01-01'
        }
        
        self.assertTrue(
            self.client._improves_diversity(movie_new_country, current_stats, requirements)
        )
        
        # Movie that adds new language should improve diversity
        movie_new_language = {
            'genre_ids': [18],  # Drama - existing genre
            'origin_country': ['US'],  # US - existing country
            'original_language': 'fr',  # French - new language
            'release_date': '2020-01-01'
        }
        
        self.assertTrue(
            self.client._improves_diversity(movie_new_language, current_stats, requirements)
        )
        
        # Movie that adds new decade should improve diversity
        movie_new_decade = {
            'genre_ids': [18],  # Drama - existing genre
            'origin_country': ['US'],  # US - existing country
            'original_language': 'en',  # English - existing language
            'release_date': '1985-01-01'  # 1980s - new decade
        }
        
        self.assertTrue(
            self.client._improves_diversity(movie_new_decade, current_stats, requirements)
        )
        
        # Movie that doesn't improve diversity
        movie_no_improvement = {
            'genre_ids': [18, 35],  # Same genres
            'origin_country': ['US'],  # Same country
            'original_language': 'en',  # Same language
            'release_date': '2020-01-01'  # Same decade
        }
        
        # Should still return True for first few movies
        empty_stats = {
            'genres': set(),
            'countries': set(),
            'languages': set(),
            'decades': set()
        }
        
        self.assertTrue(
            self.client._improves_diversity(movie_no_improvement, empty_stats, requirements)
        )
        
        # But False when stats are already populated
        self.assertFalse(
            self.client._improves_diversity(movie_no_improvement, current_stats, requirements)
        )
    
    def test_diversity_filter_application(self):
        """Test application of diversity filters to movie results."""
        # Create mock results with varying diversity
        mock_results = {
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
                    'genre_ids': [28],  # Action (same genre)
                    'origin_country': ['US'],  # US (same country)
                    'original_language': 'en',  # English (same language)
                    'release_date': '2021-01-01'  # Recent (same decade)
                },
                {
                    'id': 3,
                    'title': 'French Drama',
                    'genre_ids': [18],  # Drama (new genre)
                    'origin_country': ['FR'],  # France (new country)
                    'original_language': 'fr',  # French (new language)
                    'release_date': '1995-01-01'  # 1990s (new decade)
                },
                {
                    'id': 4,
                    'title': 'Japanese Anime',
                    'genre_ids': [16],  # Animation (new genre)
                    'origin_country': ['JP'],  # Japan (new country)
                    'original_language': 'ja',  # Japanese (new language)
                    'release_date': '2010-01-01'  # 2010s (new decade)
                },
                {
                    'id': 5,
                    'title': 'US Comedy',
                    'genre_ids': [35],  # Comedy (new genre)
                    'origin_country': ['US'],  # US (existing country)
                    'original_language': 'en',  # English (existing language)
                    'release_date': '2020-01-01'  # Recent (existing decade)
                }
            ]
        }
        
        requirements = {
            'min_genre_diversity': 4,
            'min_results': 3
        }
        
        filtered = self.client._apply_diversity_filters(mock_results, requirements)
        
        # Should have diversity analysis
        self.assertIn('diversity_analysis', filtered)
        
        analysis = filtered['diversity_analysis']
        self.assertGreater(analysis['unique_genres'], 1)
        self.assertGreater(analysis['unique_countries'], 1)
        self.assertGreater(analysis['unique_languages'], 1)
        self.assertGreater(analysis['unique_decades'], 1)
        
        # Should prefer diverse movies
        filtered_ids = {movie['id'] for movie in filtered['results']}
        
        # Should include the French drama and Japanese anime for diversity
        self.assertIn(3, filtered_ids)  # French Drama adds country, language, decade, genre diversity
        self.assertIn(4, filtered_ids)  # Japanese Anime adds country, language, decade, genre diversity
    
    def test_comprehensive_diversity_report(self):
        """Test generation of comprehensive diversity report."""
        # Set up diverse metrics
        self.client.diversity_metrics = {
            'total_requests': 100,
            'failed_requests': 5,
            'genres_fetched': {18: 25, 28: 20, 35: 15, 27: 10, 53: 8, 12: 7, 16: 5, 80: 5, 99: 3, 878: 2},
            'countries_fetched': {'US': 40, 'GB': 15, 'FR': 12, 'DE': 8, 'JP': 7, 'KR': 6, 'IN': 5, 'BR': 4, 'MX': 2, 'CA': 1},
            'languages_fetched': {'en': 45, 'fr': 12, 'de': 10, 'ja': 8, 'ko': 7, 'es': 6, 'hi': 5, 'pt': 4, 'it': 2, 'ru': 1},
            'decades_fetched': {2020: 30, 2010: 25, 2000: 20, 1990: 15, 1980: 8, 1970: 2}
        }
        
        report = self.client.get_diversity_report()
        
        # Verify report structure
        self.assertEqual(report['total_requests'], 100)
        self.assertEqual(report['failed_requests'], 5)
        self.assertAlmostEqual(report['success_rate'], 0.95, places=2)
        
        # Verify genre diversity section
        genre_div = report['genre_diversity']
        self.assertEqual(genre_div['unique_genres'], 10)
        self.assertEqual(genre_div['total_genre_fetches'], 100)
        self.assertEqual(genre_div['top_genres'][18], 25)  # Drama should be top
        
        # Verify geographic diversity section
        geo_div = report['geographic_diversity']
        self.assertEqual(geo_div['unique_countries'], 10)
        self.assertEqual(geo_div['total_country_fetches'], 100)
        self.assertEqual(geo_div['top_countries']['US'], 40)  # US should be top
        
        # Verify language diversity section
        lang_div = report['language_diversity']
        self.assertEqual(lang_div['unique_languages'], 10)
        self.assertEqual(lang_div['total_language_fetches'], 100)
        self.assertEqual(lang_div['top_languages']['en'], 45)  # English should be top
        
        # Verify temporal diversity section
        temp_div = report['temporal_diversity']
        self.assertEqual(temp_div['unique_decades'], 6)
        self.assertEqual(temp_div['total_decade_fetches'], 100)
        self.assertEqual(temp_div['decade_distribution'][2020], 30)  # 2020s should be highest
        
        # Verify bias indicators
        bias = report['bias_indicators']
        self.assertIn('genre_concentration', bias)
        self.assertIn('geographic_concentration', bias)
        self.assertIn('us_bias', bias)
        self.assertIn('language_concentration', bias)
        self.assertIn('english_bias', bias)
        self.assertIn('modern_bias', bias)
        
        # Check specific bias calculations
        self.assertAlmostEqual(bias['us_bias'], 0.4, places=2)
        self.assertAlmostEqual(bias['english_bias'], 0.45, places=2)
        self.assertAlmostEqual(bias['modern_bias'], 0.75, places=2)  # 2000+ = 75%
    
    def test_constitutional_ai_compliance_monitoring(self):
        """Test Constitutional AI compliance monitoring features."""
        # Test bias threshold violations
        self.client.diversity_metrics['countries_fetched']['US'] = 95
        self.client.diversity_metrics['countries_fetched']['Others'] = 5
        
        bias_indicators = self.client._calculate_bias_indicators()
        
        # Should flag high US bias as potential Constitutional AI violation
        us_bias = bias_indicators.get('us_bias', 0)
        self.assertGreater(us_bias, 0.9)
        
        # In a real implementation, this would trigger alerts or corrective actions
        if us_bias > 0.8:
            # Simulated corrective action
            corrective_action = "Recommend increasing international movie representation"
            self.assertEqual(corrective_action, "Recommend increasing international movie representation")
    
    def test_representation_analysis(self):
        """Test analysis of representation in fetched movies."""
        # Simulate fetching movies from different demographics
        movies_data = [
            {
                'id': 1,
                'genre_ids': [18],  # Drama
                'origin_country': ['US'],
                'original_language': 'en',
                'release_date': '2020-01-01'
            },
            {
                'id': 2,
                'genre_ids': [28],  # Action
                'origin_country': ['IN'],
                'original_language': 'hi',
                'release_date': '2019-01-01'
            },
            {
                'id': 3,
                'genre_ids': [35],  # Comedy
                'origin_country': ['KR'],
                'original_language': 'ko',
                'release_date': '2021-01-01'
            }
        ]
        
        # Track diversity for all movies
        for movie in movies_data:
            self.client._track_movie_diversity(movie)
        
        report = self.client.get_diversity_report()
        
        # Should show good representation
        self.assertEqual(report['geographic_diversity']['unique_countries'], 3)
        self.assertEqual(report['language_diversity']['unique_languages'], 3)
        self.assertEqual(report['genre_diversity']['unique_genres'], 3)
        
        # No single country should dominate
        bias = report['bias_indicators']
        self.assertLessEqual(bias.get('geographic_concentration', 1.0), 0.5)


class TestBiasAlertSystem(unittest.TestCase):
    """Test bias alerting and monitoring system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = 'test_api_key'
        
        with patch.dict('os.environ', {'TMDB_API_KEY': self.api_key}):
            self.client = EnhancedTMDBClient()
    
    def test_bias_threshold_detection(self):
        """Test detection of bias threshold violations."""
        # Set up biased data that should trigger alerts
        bias_scenarios = [
            {
                'name': 'Genre Bias',
                'setup': lambda: self.client.diversity_metrics.update({'genres_fetched': {28: 90, 18: 10}}),
                'expected_indicator': 'genre_concentration',
                'expected_threshold': 0.8
            },
            {
                'name': 'Geographic Bias',
                'setup': lambda: self.client.diversity_metrics.update({'countries_fetched': {'US': 92, 'FR': 8}}),
                'expected_indicator': 'us_bias',
                'expected_threshold': 0.8
            },
            {
                'name': 'Language Bias',
                'setup': lambda: self.client.diversity_metrics.update({'languages_fetched': {'en': 88, 'fr': 12}}),
                'expected_indicator': 'english_bias',
                'expected_threshold': 0.8
            },
            {
                'name': 'Temporal Bias',
                'setup': lambda: self.client.diversity_metrics.update({'decades_fetched': {2020: 85, 1990: 15}}),
                'expected_indicator': 'modern_bias',
                'expected_threshold': 0.7
            }
        ]
        
        for scenario in bias_scenarios:
            with self.subTest(scenario=scenario['name']):
                # Reset metrics
                self.client.diversity_metrics = {
                    'genres_fetched': defaultdict(int),
                    'countries_fetched': defaultdict(int),
                    'languages_fetched': defaultdict(int),
                    'decades_fetched': defaultdict(int),
                    'total_requests': 0,
                    'failed_requests': 0
                }
                
                # Apply scenario setup
                scenario['setup']()
                
                # Calculate bias indicators
                bias_indicators = self.client._calculate_bias_indicators()
                
                # Verify threshold violation
                indicator_value = bias_indicators.get(scenario['expected_indicator'], 0)
                self.assertGreater(
                    indicator_value,
                    scenario['expected_threshold'],
                    f"{scenario['name']}: {scenario['expected_indicator']} = {indicator_value} should exceed {scenario['expected_threshold']}"
                )


if __name__ == '__main__':
    unittest.main(verbosity=2)
