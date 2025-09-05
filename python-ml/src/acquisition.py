"""
Data acquisition module for movie datasets with caching and versioning.

This module provides robust data download capabilities with:
- Multiple data source support (MovieLens, TMDB, local fallback)
- Intelligent caching to avoid re-downloading
- Dataset version tracking and metadata
- Checksum validation for data integrity
- License compliance checking and attribution
"""

import hashlib
import json
import logging
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataAcquisitionError(Exception):
    """Custom exception for data acquisition failures."""

    pass


class DatasetDownloader:
    """Handles dataset downloading with caching and version management."""

    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """Initialize downloader with configuration."""
        self.config = self._load_config(config_path)

        # Set up directories with defaults
        download_config = self.config.get("download", {})
        versioning_config = self.config.get("versioning", {})

        self.cache_dir = Path(download_config.get("cache_dir", "data/cache"))
        self.temp_dir = Path(download_config.get("temp_dir", "data/temp"))
        self.metadata_file = Path(versioning_config.get("metadata_file", "data/metadata.json"))

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default configuration if file not found
            logger.warning(f"Configuration file not found: {config_path}, using defaults")
            return {
                "download": {"cache_dir": "data/cache", "temp_dir": "data/temp"},
                "versioning": {"metadata_file": "data/metadata.json"},
                "logging": {
                    "level": "INFO",
                    "file": "data/logs/acquisition.log",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            }
        except yaml.YAMLError as e:
            raise DataAcquisitionError(f"Error parsing configuration: {e}")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get(
            "logging",
            {
                "level": "INFO",
                "file": "data/logs/acquisition.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        )

        # Create log directory if it doesn't exist
        log_file = Path(log_config.get("file", "data/logs/acquisition.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def download_dataset(self, source_name: Optional[str] = None, force_refresh: bool = False) -> Path:
        """
        Download dataset from specified source with caching.

        Args:
            source_name: Name of data source to use (defaults to primary)
            force_refresh: Force re-download even if cached version exists

        Returns:
            Path to downloaded dataset file

        Raises:
            DataAcquisitionError: If download fails from all sources
        """
        source_name = source_name or self.config["data_sources"]["primary"]

        logger.info(f"Starting dataset download from source: {source_name}")

        try:
            # Try primary source
            return self._download_from_source(source_name, force_refresh)

        except DataAcquisitionError as e:
            logger.warning(f"Primary source failed: {e}")

            # Try backup source if different from primary
            backup_source = self.config["data_sources"]["backup"]
            if backup_source != source_name:
                logger.info(f"Attempting backup source: {backup_source}")
                try:
                    return self._download_from_source(backup_source, force_refresh)
                except DataAcquisitionError as backup_error:
                    logger.error(f"Backup source also failed: {backup_error}")

            # All sources failed
            raise DataAcquisitionError(f"Failed to download from all available sources")

    def _download_from_source(self, source_name: Optional[str], force_refresh: bool) -> Path:
        """Download from specific data source."""
        if source_name not in self.config["data_sources"]:
            raise DataAcquisitionError(f"Unknown data source: {source_name}")

        source_config = self.config["data_sources"][source_name]

        if source_name == "local_fallback":
            return self._handle_local_fallback(source_config)
        elif source_name == "movielens":
            return self._download_movielens(source_config, force_refresh)
        elif source_name == "tmdb":
            return self._download_tmdb(source_config, force_refresh)
        else:
            raise DataAcquisitionError(f"Unsupported data source: {source_name}")

    def _handle_local_fallback(self, source_config: Dict[str, Any]) -> Path:
        """Handle local fallback dataset."""
        fallback_path = Path(source_config["path"])

        if not fallback_path.exists():
            # Create a minimal fallback dataset
            logger.warning("Local fallback dataset not found, creating minimal sample")
            self._create_minimal_fallback(fallback_path)

        return fallback_path

    def _create_minimal_fallback(self, fallback_path: Path) -> None:
        """Create a minimal fallback dataset for testing."""
        minimal_data = [
            {
                "movie_id": "1",
                "title": "The Shawshank Redemption",
                "synopsis": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
                "release_year": 1994,
                "runtime_mins": 142,
                "genres": ["Drama"],
                "ratings": {"average": 9.3, "count": 2500000},
            },
            {
                "movie_id": "2",
                "title": "The Godfather",
                "synopsis": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
                "release_year": 1972,
                "runtime_mins": 175,
                "genres": ["Crime", "Drama"],
                "ratings": {"average": 9.2, "count": 1800000},
            },
        ]

        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fallback_path, "w") as f:
            json.dump(minimal_data, f, indent=2)

        logger.info(f"Created minimal fallback dataset at {fallback_path}")

    def _download_movielens(self, source_config: Dict[str, Any], force_refresh: bool) -> Path:
        """Download MovieLens dataset."""
        url = source_config["url"]
        filename = source_config["filename"]
        expected_checksum = source_config.get("checksum", "").replace("md5:", "")

        cached_file = self.cache_dir / filename

        # Check if cached version exists and is valid
        if not force_refresh and cached_file.exists():
            if self._verify_file_integrity(cached_file, expected_checksum):
                logger.info(f"Using cached dataset: {cached_file}")
                self._update_metadata(source_config, cached_file, "cached")
                return cached_file
            else:
                logger.warning("Cached file failed integrity check, re-downloading")

        # Download the file
        logger.info(f"Downloading MovieLens dataset from: {url}")

        try:
            downloaded_file = self._download_with_progress(url, cached_file)

            # Verify checksum if provided
            if expected_checksum and not self._verify_file_integrity(downloaded_file, expected_checksum):
                raise DataAcquisitionError("Downloaded file failed checksum verification")

            # Update metadata
            self._update_metadata(source_config, downloaded_file, "downloaded")

            logger.info(f"Successfully downloaded dataset to: {downloaded_file}")
            return downloaded_file

        except (URLError, HTTPError, requests.RequestException) as e:
            raise DataAcquisitionError(f"Network error downloading MovieLens data: {e}")

    def _download_tmdb(self, source_config: Dict[str, Any], force_refresh: bool) -> Path:
        """Download data from TMDB API using Enhanced TMDB Client."""
        try:
            from .tmdb_client import EnhancedTMDBClient, TMDBError
        except ImportError:
            logger.error("Enhanced TMDB Client not available, using local fallback")
            return self._handle_local_fallback(self.config["data_sources"]["local_fallback"])
        
        # Determine output file path
        filename = source_config.get("filename", "tmdb_movies.json")
        output_file = self.cache_dir / filename
        
        # Check if cached version exists and is valid (not forcing refresh)
        if not force_refresh and output_file.exists():
            file_age = datetime.now().timestamp() - output_file.stat().st_mtime
            max_age = source_config.get("cache_ttl_hours", 24) * 3600  # Convert to seconds
            
            if file_age < max_age:
                logger.info(f"Using cached TMDB dataset: {output_file}")
                self._update_metadata(source_config, output_file, "cached")
                return output_file
            else:
                logger.info("Cached TMDB dataset expired, refreshing")
        
        logger.info("Fetching movie data from TMDB API")
        
        try:
            # Initialize Enhanced TMDB Client
            tmdb_client = EnhancedTMDBClient(
                rate_limit_per_second=source_config.get("rate_limit_per_second", 3.0),
                enable_caching=source_config.get("enable_caching", True),
                max_retries=source_config.get("max_retries", 3)
            )
            
            # Fetch diverse movies based on configuration
            movies_data = self._fetch_diverse_tmdb_movies(tmdb_client, source_config)
            
            # Convert to our movie dataset format
            processed_movies = self._convert_tmdb_to_movie_format(movies_data)
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_movies, f, indent=2, ensure_ascii=False)
            
            # Generate diversity report
            diversity_report = tmdb_client.get_diversity_report()
            self._save_tmdb_diversity_report(diversity_report, output_file.parent)
            
            # Update metadata
            self._update_metadata(source_config, output_file, "downloaded")
            
            logger.info(f"Successfully downloaded {len(processed_movies)} movies from TMDB to: {output_file}")
            logger.info(f"Diversity report - Unique genres: {diversity_report['genre_diversity']['unique_genres']}, "
                       f"Countries: {diversity_report['geographic_diversity']['unique_countries']}, "
                       f"Success rate: {diversity_report['success_rate']:.2%}")
            
            return output_file
            
        except TMDBError as e:
            logger.error(f"TMDB API error: {e}")
            raise DataAcquisitionError(f"Failed to fetch data from TMDB API: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching TMDB data: {e}")
            raise DataAcquisitionError(f"Unexpected error during TMDB data acquisition: {e}")
    
    def _fetch_diverse_tmdb_movies(self, tmdb_client, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch diverse movie data from TMDB API."""
        all_movies = []
        
        # Configuration for diverse fetching
        fetch_config = source_config.get("fetch_config", {})
        max_movies = fetch_config.get("max_movies", 1000)
        min_diversity_score = fetch_config.get("min_diversity_score", 0.7)
        
        # Diversity requirements
        diversity_requirements = {
            "min_genre_diversity": fetch_config.get("min_genre_diversity", 10),
            "min_decade_spread": fetch_config.get("min_decade_spread", 4),
            "max_us_percentage": fetch_config.get("max_us_percentage", 0.6),
            "min_results": fetch_config.get("min_results_per_fetch", 20)
        }
        
        logger.info(f"Fetching up to {max_movies} diverse movies from TMDB")
        
        # Strategy 1: Get popular movies with diversity filtering
        try:
            popular_movies = tmdb_client.get_popular_movies_with_diversity(page=1, ensure_diversity=True)
            movies_from_popular = popular_movies.get('results', [])
            all_movies.extend(movies_from_popular)
            logger.info(f"Fetched {len(movies_from_popular)} movies from popular endpoint")
        except Exception as e:
            logger.warning(f"Error fetching popular movies: {e}")
        
        # Strategy 2: Discover movies with specific diversity criteria
        discover_strategies = [
            # High-rated international films
            {
                "sort_by": "vote_average.desc", 
                "with_origin_country": ["FR", "DE", "JP", "KR", "IT", "ES", "IN"],
                "primary_release_date_gte": "1990-01-01"
            },
            # Recent diverse films
            {
                "sort_by": "popularity.desc",
                "primary_release_date_gte": "2015-01-01",
                "without_genres": [28],  # Avoid too much action
                "diversity_requirements": diversity_requirements
            },
            # Classic films for temporal diversity
            {
                "sort_by": "vote_count.desc",
                "primary_release_date_lte": "2000-12-31",
                "primary_release_date_gte": "1970-01-01",
                "diversity_requirements": diversity_requirements
            },
            # Genre-specific diverse searches
            {
                "with_genres": [18],  # Drama
                "sort_by": "vote_average.desc",
                "diversity_requirements": diversity_requirements
            },
            {
                "with_genres": [16],  # Animation 
                "sort_by": "popularity.desc",
                "diversity_requirements": diversity_requirements
            },
            {
                "with_genres": [99],  # Documentary
                "sort_by": "vote_average.desc",
                "diversity_requirements": diversity_requirements
            }
        ]
        
        for i, strategy in enumerate(discover_strategies):
            if len(all_movies) >= max_movies:
                break
                
            try:
                logger.info(f"Applying discovery strategy {i+1}: {strategy.get('sort_by', 'custom')}")
                discovered = tmdb_client.discover_diverse_movies(**strategy)
                new_movies = discovered.get('results', [])
                
                # Avoid duplicates
                existing_ids = {movie.get('id') for movie in all_movies}
                unique_new_movies = [movie for movie in new_movies if movie.get('id') not in existing_ids]
                
                all_movies.extend(unique_new_movies)
                logger.info(f"Added {len(unique_new_movies)} unique movies from strategy {i+1}")
                
            except Exception as e:
                logger.warning(f"Error with discovery strategy {i+1}: {e}")
                continue
        
        # Strategy 3: Fetch detailed data for top movies to ensure quality
        detailed_movies = []
        for movie in all_movies[:max_movies]:
            try:
                # Get full movie details with credits
                movie_details = tmdb_client.get_movie_details(
                    movie['id'], 
                    append_to_response=['credits', 'keywords', 'videos']
                )
                detailed_movies.append(movie_details)
                
                # Rate limiting - don't overwhelm the API
                if len(detailed_movies) % 50 == 0:
                    logger.info(f"Fetched detailed data for {len(detailed_movies)} movies...")
                    
            except Exception as e:
                logger.warning(f"Error fetching details for movie {movie.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully fetched detailed data for {len(detailed_movies)} movies")
        return detailed_movies
    
    def _convert_tmdb_to_movie_format(self, tmdb_movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert TMDB API response format to our movie dataset format."""
        processed_movies = []
        
        for tmdb_movie in tmdb_movies:
            try:
                # Extract basic information
                movie_data = {
                    "movie_id": str(tmdb_movie.get('id', '')),
                    "title": tmdb_movie.get('title', ''),
                    "synopsis": tmdb_movie.get('overview', ''),
                    "release_year": self._extract_year_from_date(tmdb_movie.get('release_date', '')),
                    "runtime_mins": tmdb_movie.get('runtime'),
                    "genres": [genre.get('name', '') for genre in tmdb_movie.get('genres', [])],
                    "ratings": {
                        "average": tmdb_movie.get('vote_average', 0.0),
                        "count": tmdb_movie.get('vote_count', 0)
                    }
                }
                
                # Add cast information if available
                if 'credits' in tmdb_movie and 'cast' in tmdb_movie['credits']:
                    movie_data['cast'] = [
                        {
                            "name": person.get('name', ''),
                            "character": person.get('character', ''),
                            "order": person.get('order', 999),
                            "gender": person.get('gender'),
                            "profile_path": person.get('profile_path')
                        }
                        for person in tmdb_movie['credits']['cast'][:20]  # Limit to top 20
                    ]
                
                # Add crew information if available
                if 'credits' in tmdb_movie and 'crew' in tmdb_movie['credits']:
                    movie_data['crew'] = [
                        {
                            "name": person.get('name', ''),
                            "job": person.get('job', ''),
                            "department": person.get('department', ''),
                            "gender": person.get('gender'),
                            "profile_path": person.get('profile_path')
                        }
                        for person in tmdb_movie['credits']['crew']
                        if person.get('job') in ['Director', 'Producer', 'Writer', 'Screenplay', 'Executive Producer']
                    ]
                
                # Add metadata
                movie_data['metadata'] = {
                    "tmdb_id": tmdb_movie.get('id'),
                    "imdb_id": tmdb_movie.get('imdb_id'),
                    "popularity": tmdb_movie.get('popularity'),
                    "budget": tmdb_movie.get('budget'),
                    "revenue": tmdb_movie.get('revenue'),
                    "language": tmdb_movie.get('original_language', 'en'),
                    "country": tmdb_movie.get('origin_country', ['US'])[0] if tmdb_movie.get('origin_country') else 'US',
                    "status": tmdb_movie.get('status', 'Released'),
                    "tagline": tmdb_movie.get('tagline', ''),
                    "keywords": [kw.get('name', '') for kw in tmdb_movie.get('keywords', {}).get('keywords', [])]
                }
                
                # Only add movies with sufficient data quality
                if (movie_data['title'] and 
                    movie_data['synopsis'] and 
                    len(movie_data['synopsis']) > 50 and
                    movie_data['release_year'] and 
                    movie_data['genres']):
                    processed_movies.append(movie_data)
                
            except Exception as e:
                logger.warning(f"Error processing TMDB movie {tmdb_movie.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_movies)} movies from TMDB data")
        return processed_movies
    
    def _extract_year_from_date(self, date_string: str) -> Optional[int]:
        """Extract year from date string (YYYY-MM-DD format)."""
        if not date_string:
            return None
        try:
            return int(date_string.split('-')[0])
        except (ValueError, IndexError):
            return None
    
    def _save_tmdb_diversity_report(self, diversity_report: Dict[str, Any], output_dir: Path) -> None:
        """Save TMDB diversity report for Constitutional AI monitoring."""
        try:
            report_file = output_dir / "tmdb_diversity_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(diversity_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved TMDB diversity report to: {report_file}")
            
            # Log key diversity metrics
            bias_indicators = diversity_report.get('bias_indicators', {})
            if bias_indicators:
                logger.info("Constitutional AI Bias Indicators:")
                for indicator, value in bias_indicators.items():
                    level = "HIGH" if value > 0.8 else "MEDIUM" if value > 0.5 else "LOW"
                    logger.info(f"  {indicator}: {value:.2%} ({level})")
                    
        except Exception as e:
            logger.warning(f"Failed to save diversity report: {e}")

    def _download_with_progress(self, url: str, target_path: Path) -> Path:
        """Download file with progress bar."""
        config = self.config["download"]

        # Use temporary file during download
        with tempfile.NamedTemporaryFile(delete=False, dir=self.temp_dir) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            response = requests.get(url, stream=True, timeout=config["timeout_seconds"])
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(temp_path, "wb") as f:
                with tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=config["chunk_size"]):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Move completed download to final location
            temp_path.replace(target_path)
            return target_path

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def _verify_file_integrity(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file integrity using MD5 checksum."""
        if not expected_checksum:
            return True  # Skip verification if no checksum provided

        try:
            file_hash = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)

            actual_checksum = file_hash.hexdigest()
            is_valid = actual_checksum == expected_checksum

            if not is_valid:
                logger.error(f"Checksum mismatch - expected: {expected_checksum}, actual: {actual_checksum}")

            return is_valid

        except Exception as e:
            logger.error(f"Error verifying file integrity: {e}")
            return False

    def _update_metadata(self, source_config: Dict[str, Any], file_path: Path, action: str) -> None:
        """Update dataset metadata with download information."""
        metadata = self._load_metadata()

        file_stat = file_path.stat()

        entry = {
            "source": source_config,
            "file_path": str(file_path),
            "file_size": file_stat.st_size,
            "download_timestamp": datetime.now().isoformat(),
            "action": action,
            "attribution": source_config.get("attribution", ""),
            "license": source_config.get("license", ""),
        }

        metadata[f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = entry

        # Save updated metadata
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Updated dataset metadata")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading metadata, creating new: {e}")

        return {}

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about available datasets."""
        metadata = self._load_metadata()

        info = {
            "available_sources": list(self.config["data_sources"].keys()),
            "primary_source": self.config["data_sources"]["primary"],
            "cached_datasets": len(metadata),
            "cache_directory": str(self.cache_dir),
            "last_download": None,
        }

        # Find most recent download
        if metadata:
            latest_entry = max(metadata.values(), key=lambda x: x["download_timestamp"])
            info["last_download"] = latest_entry["download_timestamp"]

        return info

    def extract_dataset(self, dataset_path: Path, extract_dir: Optional[Path] = None) -> Path:
        """Extract dataset if it's a compressed file."""
        if not extract_dir:
            extract_dir = dataset_path.parent / "extracted"

        extract_dir.mkdir(parents=True, exist_ok=True)

        if dataset_path.suffix.lower() == ".zip":
            logger.info(f"Extracting ZIP file: {dataset_path}")

            with zipfile.ZipFile(dataset_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            logger.info(f"Extracted to: {extract_dir}")
            return extract_dir

        else:
            # Not a compressed file, return as-is
            return dataset_path


def validate_license_compliance(source_config: Dict[str, Any]) -> List[str]:
    """
    Validate license compliance for dataset usage.

    Returns list of required attributions/acknowledgments.
    """
    requirements = []

    if "attribution" in source_config:
        requirements.append(f"Attribution required: {source_config['attribution']}")

    if "license" in source_config:
        requirements.append(f"License terms: {source_config['license']}")

    return requirements


def generate_attribution_file(metadata: Dict[str, Any], output_path: str = "DATASET_ATTRIBUTIONS.md") -> None:
    """Generate attribution file for dataset compliance."""
    with open(output_path, "w") as f:
        f.write("# Dataset Attributions\n\n")
        f.write("This project uses the following datasets:\n\n")

        for dataset_id, info in metadata.items():
            f.write(f"## {dataset_id}\n")

            if "attribution" in info and info["attribution"]:
                f.write(f"**Attribution:** {info['attribution']}\n\n")

            if "license" in info and info["license"]:
                f.write(f"**License:** {info['license']}\n\n")

            f.write(f"**Downloaded:** {info['download_timestamp']}\n")
            f.write(f"**File Size:** {info['file_size']} bytes\n\n")

    logger.info(f"Generated attribution file: {output_path}")


if __name__ == "__main__":
    # Example usage
    try:
        downloader = DatasetDownloader()

        # Get dataset information
        info = downloader.get_dataset_info()
        print("Dataset Info:", json.dumps(info, indent=2))

        # Download dataset
        dataset_path = downloader.download_dataset()
        print(f"Dataset downloaded to: {dataset_path}")

        # Extract if needed
        extracted_path = downloader.extract_dataset(dataset_path)
        print(f"Dataset available at: {extracted_path}")

        # Generate attribution file
        metadata = downloader._load_metadata()
        generate_attribution_file(metadata)

    except DataAcquisitionError as e:
        logger.error(f"Data acquisition failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
