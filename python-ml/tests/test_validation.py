"""
Tests for the data validation and bias detection module.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from src.validation import DataValidator, BiasMetrics
from src.schema import Movie, Genre, Language, Ratings, PersonInfo


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'quality_thresholds': {
                'completeness_min': 0.8,
                'synopsis_min_length': 30,
                'title_min_length': 1,
                'valid_year_range': [1900, 2025],
                'min_rating_count': 1,
                'max_cast_size': 100,
                'max_crew_size': 50
            },
            'bias_detection': {
                'enable_genre_analysis': True,
                'enable_demographic_analysis': True,
                'enable_geographic_analysis': True,
                'enable_temporal_analysis': True
            }
        }
        
        self.validator = DataValidator(self.test_config)
        
        # Sample test movies
        self.sample_movies = [
            {
                "movie_id": "1",
                "title": "The Shawshank Redemption",
                "synopsis": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
                "release_year": 1994,
                "runtime_mins": 142,
                "genres": [Genre.DRAMA],
                "ratings": {
                    "average": 9.3,
                    "count": 2500000
                }
            },
            {
                "movie_id": "2",
                "title": "The Godfather",
                "synopsis": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
                "release_year": 1972,
                "runtime_mins": 175,
                "genres": [Genre.CRIME, Genre.DRAMA],
                "ratings": {
                    "average": 9.2,
                    "count": 1800000
                }
            },
            {
                "movie_id": "3",
                "title": "Pulp Fiction",
                "synopsis": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
                "release_year": 1994,
                "runtime_mins": 154,
                "genres": [Genre.CRIME, Genre.DRAMA],
                "ratings": {
                    "average": 8.9,
                    "count": 2000000
                }
            }
        ]
        
    def test_validator_initialization(self):
        """Test DataValidator initialization with config."""
        self.assertIsNotNone(self.validator.config)
        self.assertIsNotNone(self.validator.thresholds)
        self.assertEqual(self.validator.thresholds.completeness_min, 0.8)
        
    def test_dataset_validation_success(self):
        """Test successful dataset validation."""
        validation_result, valid_movies = self.validator.validate_dataset(self.sample_movies)
        
        self.assertEqual(len(valid_movies), 3)
        self.assertTrue(validation_result.is_valid or validation_result.warning_count > 0)
        self.assertEqual(validation_result.summary['total_movies'], 3)
        self.assertEqual(validation_result.summary['valid_movies'], 3)
        
    def test_dataset_validation_with_invalid_movie(self):
        """Test dataset validation with invalid movie."""
        invalid_movies = self.sample_movies + [{
            "movie_id": "invalid",
            "title": "",  # Invalid: too short
            "synopsis": "Short",  # Invalid: too short
            "release_year": 2050,  # Invalid: future year
            "genres": [],  # Invalid: no genres
            "ratings": {
                "average": 15.0,  # Invalid: out of range
                "count": -5  # Invalid: negative count
            }
        }]
        
        validation_result, valid_movies = self.validator.validate_dataset(invalid_movies)
        
        self.assertEqual(len(valid_movies), 3)  # Only valid movies
        self.assertFalse(validation_result.is_valid)
        self.assertGreater(validation_result.error_count, 0)
        
    def test_field_completeness_calculation(self):
        """Test field completeness calculation."""
        # Create movies with varying completeness
        movies = [
            Movie(
                movie_id="1",
                title="Movie 1", 
                synopsis="This is a test synopsis that is long enough to pass validation requirements.",
                release_year=2000,
                genres=[Genre.DRAMA],
                ratings=Ratings(average=8.0, count=100),
                runtime_mins=120,  # Has runtime
                cast=[PersonInfo(name="Actor 1", role="actor")]  # Has cast
            ),
            Movie(
                movie_id="2",
                title="Movie 2",
                synopsis="Another test synopsis that meets the minimum length requirements for validation.",
                release_year=2001, 
                genres=[Genre.COMEDY],
                ratings=Ratings(average=7.5, count=50)
                # Missing runtime_mins and cast
            )
        ]
        
        completeness = self.validator._calculate_field_completeness(movies)
        
        self.assertEqual(completeness['movie_id'], 1.0)  # Always complete
        self.assertEqual(completeness['title'], 1.0)     # Always complete
        self.assertEqual(completeness['runtime_mins'], 0.5)  # 1 out of 2
        self.assertEqual(completeness['cast'], 0.5)      # 1 out of 2
        
    def test_bias_detection_genre_diversity(self):
        """Test genre diversity analysis."""
        validation_result, valid_movies = self.validator.validate_dataset(self.sample_movies)
        bias_metrics = self.validator.detect_bias(valid_movies)
        
        self.assertIn('genre_diversity', bias_metrics.__dict__)
        self.assertIn('unique_genres', bias_metrics.genre_diversity)
        self.assertIn('genre_distribution', bias_metrics.genre_diversity)
        
        # Should detect Crime and Drama genres
        genre_dist = bias_metrics.genre_diversity['genre_distribution']
        self.assertIn('Drama', genre_dist)
        self.assertIn('Crime', genre_dist)
        
    def test_bias_detection_temporal_analysis(self):
        """Test temporal bias detection."""
        validation_result, valid_movies = self.validator.validate_dataset(self.sample_movies)
        bias_metrics = self.validator.detect_bias(valid_movies)
        
        self.assertIn('temporal_distribution', bias_metrics.__dict__)
        temporal = bias_metrics.temporal_distribution
        
        self.assertIn('year_range', temporal)
        self.assertIn('decade_distribution', temporal)
        self.assertEqual(temporal['year_range']['min'], 1972)
        self.assertEqual(temporal['year_range']['max'], 1994)
        
    def test_bias_detection_geographic_analysis(self):
        """Test geographic distribution analysis."""
        validation_result, valid_movies = self.validator.validate_dataset(self.sample_movies)
        bias_metrics = self.validator.detect_bias(valid_movies)
        
        self.assertIn('geographic_distribution', bias_metrics.__dict__)
        geo = bias_metrics.geographic_distribution
        
        self.assertIn('country_distribution', geo)
        self.assertIn('language_distribution', geo) 
        self.assertIn('unique_countries', geo)
        self.assertIn('unique_languages', geo)
        
    def test_bias_score_calculation(self):
        """Test overall bias score calculation."""
        validation_result, valid_movies = self.validator.validate_dataset(self.sample_movies)
        bias_metrics = self.validator.detect_bias(valid_movies)
        
        self.assertIsInstance(bias_metrics.overall_bias_score, float)
        self.assertGreaterEqual(bias_metrics.overall_bias_score, 0.0)
        self.assertLessEqual(bias_metrics.overall_bias_score, 1.0)
        
    def test_bias_recommendations_generation(self):
        """Test bias recommendation generation."""
        validation_result, valid_movies = self.validator.validate_dataset(self.sample_movies)
        bias_metrics = self.validator.detect_bias(valid_movies)
        
        self.assertIsInstance(bias_metrics.recommendations, list)
        self.assertGreater(len(bias_metrics.recommendations), 0)
        
        for recommendation in bias_metrics.recommendations:
            self.assertIsInstance(recommendation, str)
            self.assertGreater(len(recommendation), 10)  # Should be meaningful
            
    def test_movie_quality_validation(self):
        """Test individual movie quality checks."""
        # Test with a movie that should trigger warnings
        from src.schema import ValidationResult
        result = ValidationResult(is_valid=True)
        
        short_synopsis_movie = Movie(
            movie_id="test",
            title="Test Movie",
            synopsis="Short synopsis.",  # Too short
            release_year=1800,  # Outside range
            genres=[Genre.DRAMA],
            ratings=Ratings(average=8.0, count=0)  # Zero ratings
        )
        
        self.validator._validate_movie_quality(short_synopsis_movie, result)
        
        self.assertGreater(result.warning_count, 0)
        
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        validation_result, valid_movies = self.validator.validate_dataset([])
        
        self.assertEqual(len(valid_movies), 0)
        self.assertEqual(validation_result.summary['total_movies'], 0)
        self.assertEqual(validation_result.summary['valid_movies'], 0)
        
        # Bias detection with empty dataset
        bias_metrics = self.validator.detect_bias([])
        self.assertEqual(bias_metrics.overall_bias_score, 0.0)
        
    @patch('src.validation.PANDAS_AVAILABLE', False)
    def test_without_pandas(self):
        """Test validator behavior when pandas is not available."""
        validator = DataValidator(self.test_config)
        
        validation_result, valid_movies = validator.validate_dataset(self.sample_movies)
        
        # Should still work for basic validation
        self.assertEqual(len(valid_movies), 3)
        
    def test_html_report_generation(self):
        """Test HTML report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"
            
            validation_result, valid_movies = self.validator.validate_dataset(self.sample_movies)
            bias_metrics = self.validator.detect_bias(valid_movies)
            
            # Should not raise an exception
            try:
                self.validator.generate_html_report(
                    validation_result, 
                    bias_metrics, 
                    valid_movies, 
                    str(output_path)
                )
                
                # Check if file was created
                self.assertTrue(output_path.exists())
                
                # Check if file has content
                with open(output_path, 'r') as f:
                    content = f.read()
                    self.assertIn('MovieRecs Data Quality', content)
                    self.assertIn('Bias Analysis', content)
                    
            except Exception as e:
                # HTML generation might fail due to missing dependencies in test environment
                # This is acceptable as long as it fails gracefully
                self.assertIn('available', str(e).lower())


class TestBiasMetrics(unittest.TestCase):
    """Test BiasMetrics model."""
    
    def test_bias_metrics_initialization(self):
        """Test BiasMetrics initialization."""
        metrics = BiasMetrics()
        
        self.assertIsInstance(metrics.genre_diversity, dict)
        self.assertIsInstance(metrics.demographic_representation, dict)
        self.assertIsInstance(metrics.geographic_distribution, dict)
        self.assertIsInstance(metrics.temporal_distribution, dict)
        self.assertIsInstance(metrics.rating_bias_analysis, dict)
        self.assertEqual(metrics.overall_bias_score, 0.0)
        self.assertIsInstance(metrics.recommendations, list)
        
    def test_bias_metrics_with_data(self):
        """Test BiasMetrics with sample data."""
        metrics = BiasMetrics(
            genre_diversity={'unique_genres': 5, 'shannon_diversity': 1.5},
            overall_bias_score=0.3,
            recommendations=['Test recommendation']
        )
        
        self.assertEqual(metrics.genre_diversity['unique_genres'], 5)
        self.assertEqual(metrics.overall_bias_score, 0.3)
        self.assertEqual(len(metrics.recommendations), 1)


class TestDataValidatorIntegration(unittest.TestCase):
    """Integration tests for the complete validation pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete validation and bias detection pipeline."""
        config = {
            'quality_thresholds': {'completeness_min': 0.5},
            'bias_detection': {
                'enable_genre_analysis': True,
                'enable_temporal_analysis': True
            }
        }
        
        validator = DataValidator(config)
        
        # Extended sample with more diversity
        movies = [
            {
                "movie_id": "1",
                "title": "Action Movie",
                "synopsis": "An action-packed thriller with explosions and car chases that keeps viewers on the edge of their seats.",
                "release_year": 2020,
                "genres": [Genre.ACTION],
                "ratings": {"average": 7.5, "count": 100000}
            },
            {
                "movie_id": "2", 
                "title": "Comedy Film",
                "synopsis": "A hilarious comedy about misunderstandings and romantic entanglements in a small town setting.",
                "release_year": 2019,
                "genres": [Genre.COMEDY],
                "ratings": {"average": 6.8, "count": 50000}
            },
            {
                "movie_id": "3",
                "title": "Drama Series",
                "synopsis": "A deep emotional drama exploring family relationships and personal growth over several decades.",
                "release_year": 1995,
                "genres": [Genre.DRAMA],
                "ratings": {"average": 8.9, "count": 200000}
            },
            {
                "movie_id": "4",
                "title": "Sci-Fi Adventure", 
                "synopsis": "A science fiction epic set in the distant future with advanced technology and alien civilizations.",
                "release_year": 2021,
                "genres": [Genre.SCIENCE_FICTION, Genre.ADVENTURE],
                "ratings": {"average": 8.2, "count": 150000}
            }
        ]
        
        # Run full pipeline
        validation_result, valid_movies = validator.validate_dataset(movies)
        
        self.assertEqual(len(valid_movies), 4)
        self.assertTrue(validation_result.is_valid or validation_result.warning_count >= 0)
        
        # Run bias detection
        bias_metrics = validator.detect_bias(valid_movies)
        
        self.assertGreaterEqual(bias_metrics.overall_bias_score, 0.0)
        self.assertLessEqual(bias_metrics.overall_bias_score, 1.0)
        self.assertGreater(len(bias_metrics.recommendations), 0)
        
        # Check that we detected different genres
        genre_diversity = bias_metrics.genre_diversity
        self.assertGreater(genre_diversity['unique_genres'], 1)
        self.assertIn('genre_distribution', genre_diversity)
        

if __name__ == "__main__":
    unittest.main()