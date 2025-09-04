"""
Feature utilities for movie recommendation system.

This module provides helper functions for feature engineering, including
feature name mapping, scaling utilities, category encoding, and feature selection.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureNameMapper:
    """Maps between feature names and their descriptions/metadata."""
    
    def __init__(self):
        """Initialize feature name mapper with standard mappings."""
        self.name_mappings = {
            # Text features
            'text_': 'Text/Synopsis features (TF-IDF)',
            
            # Genre features
            'genre_action': 'Action genre indicator',
            'genre_adventure': 'Adventure genre indicator',
            'genre_animation': 'Animation genre indicator',
            'genre_comedy': 'Comedy genre indicator',
            'genre_crime': 'Crime genre indicator',
            'genre_documentary': 'Documentary genre indicator',
            'genre_drama': 'Drama genre indicator',
            'genre_family': 'Family genre indicator',
            'genre_fantasy': 'Fantasy genre indicator',
            'genre_history': 'History genre indicator',
            'genre_horror': 'Horror genre indicator',
            'genre_music': 'Music genre indicator',
            'genre_mystery': 'Mystery genre indicator',
            'genre_romance': 'Romance genre indicator',
            'genre_science_fiction': 'Science Fiction genre indicator',
            'genre_thriller': 'Thriller genre indicator',
            'genre_war': 'War genre indicator',
            'genre_western': 'Western genre indicator',
            'genre_other': 'Other genre indicator',
            
            # Temporal features
            'decade_pre_1950': 'Pre-1950s movies indicator',
            'decade_mid_century': 'Mid-century (1950s-1970s) movies indicator', 
            'decade_modern': 'Modern (1980s-1990s) movies indicator',
            'decade_contemporary': 'Contemporary (2000s+) movies indicator',
            'decade_normalized': 'Release decade (normalized)',
            
            # Runtime features
            'runtime_short': 'Short runtime (<90 min) indicator',
            'runtime_medium': 'Medium runtime (90-120 min) indicator',
            'runtime_long': 'Long runtime (120-180 min) indicator',
            'runtime_epic': 'Epic runtime (>180 min) indicator',
            'runtime_normalized': 'Runtime in minutes (normalized)',
            'has_runtime': 'Runtime information available indicator',
            
            # Rating features
            'rating_average': 'Average user rating (normalized)',
            'rating_count_log': 'Number of ratings (log-scaled)',
            'rating_interaction': 'Rating × log(count) interaction',
            'rating_high': 'High rating (≥8.0) indicator',
            'rating_medium': 'Medium rating (6.0-8.0) indicator',
            'rating_low': 'Low rating (<6.0) indicator',
            
            # Cast/crew features
            'cast_size': 'Number of cast members (normalized)',
            'crew_size': 'Number of crew members (normalized)',
            'cast_avg_age': 'Average cast age (normalized)',
            'cast_gender_balance': 'Cast gender balance (0=male, 1=female)',
            'cast_known_ratio': 'Ratio of cast with known info',
            'crew_avg_age': 'Average crew age (normalized)',
            'crew_gender_balance': 'Crew gender balance (0=male, 1=female)',
            'crew_known_ratio': 'Ratio of crew with known info'
        }
        
    def get_description(self, feature_name: str) -> str:
        """Get human-readable description for a feature."""
        # Check exact match first
        if feature_name in self.name_mappings:
            return self.name_mappings[feature_name]
            
        # Check prefix matches
        for prefix, description in self.name_mappings.items():
            if feature_name.startswith(prefix):
                return description
                
        return f"Feature: {feature_name}"
        
    def get_feature_categories(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Group features by category."""
        categories = {
            'text': [],
            'genre': [],
            'temporal': [],
            'runtime': [],
            'rating': [],
            'cast_crew': [],
            'other': []
        }
        
        for name in feature_names:
            if name.startswith('text_'):
                categories['text'].append(name)
            elif name.startswith('genre_'):
                categories['genre'].append(name)
            elif name.startswith('decade_'):
                categories['temporal'].append(name)
            elif name.startswith('runtime_'):
                categories['runtime'].append(name)
            elif name.startswith('rating_'):
                categories['rating'].append(name)
            elif name.startswith(('cast_', 'crew_')):
                categories['cast_crew'].append(name)
            else:
                categories['other'].append(name)
                
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
        
    def create_feature_documentation(self, 
                                   feature_names: List[str], 
                                   output_path: str = "data/processed/feature_documentation.md") -> None:
        """Generate feature documentation file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        categories = self.get_feature_categories(feature_names)
        
        with open(output_path, 'w') as f:
            f.write("# Movie Features Documentation\n\n")
            f.write(f"Total features: {len(feature_names)}\n\n")
            
            for category, features in categories.items():
                f.write(f"## {category.replace('_', ' ').title()} Features\n\n")
                f.write(f"Count: {len(features)}\n\n")
                
                for feature in features:
                    description = self.get_description(feature)
                    f.write(f"- **{feature}**: {description}\n")
                    
                f.write("\n")
                
        logger.info(f"Feature documentation written to: {output_path}")


class ScalingUtilities:
    """Utilities for feature scaling and normalization."""
    
    @staticmethod
    def get_scaler(scaler_type: str = 'standard') -> Any:
        """Get sklearn scaler instance by type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if scaler_type not in scalers:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Available: {list(scalers.keys())}")
            
        return scalers[scaler_type]
        
    @staticmethod
    def scale_features(features: np.ndarray, 
                      scaler_type: str = 'standard',
                      fit: bool = True,
                      scaler: Optional[Any] = None) -> Tuple[np.ndarray, Any]:
        """
        Scale feature matrix.
        
        Args:
            features: Feature matrix to scale
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            fit: Whether to fit the scaler
            scaler: Pre-fitted scaler to use (if provided, scaler_type ignored)
            
        Returns:
            Tuple of (scaled_features, fitted_scaler)
        """
        if scaler is None:
            scaler = ScalingUtilities.get_scaler(scaler_type)
            
        if fit:
            scaled_features = scaler.fit_transform(features)
        else:
            scaled_features = scaler.transform(features)
            
        return scaled_features, scaler
        
    @staticmethod
    def scale_feature_groups(features: np.ndarray,
                           feature_names: List[str],
                           group_scalers: Dict[str, str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply different scaling to different feature groups.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            group_scalers: Dict mapping feature prefixes to scaler types
            
        Returns:
            Tuple of (scaled_features, fitted_scalers_dict)
        """
        if group_scalers is None:
            group_scalers = {
                'text_': 'standard',
                'genre_': 'minmax', 
                'runtime_': 'standard',
                'rating_': 'standard',
                'cast_': 'robust',
                'crew_': 'robust'
            }
            
        scaled_features = features.copy()
        fitted_scalers = {}
        
        for prefix, scaler_type in group_scalers.items():
            # Find feature indices matching prefix
            indices = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
            
            if indices:
                group_features = features[:, indices]
                scaler = ScalingUtilities.get_scaler(scaler_type)
                
                scaled_group = scaler.fit_transform(group_features)
                scaled_features[:, indices] = scaled_group
                
                fitted_scalers[prefix] = scaler
                logger.info(f"Scaled {len(indices)} features with prefix '{prefix}' using {scaler_type}")
                
        return scaled_features, fitted_scalers


class CategoryEncoder:
    """Utilities for encoding categorical features."""
    
    @staticmethod
    def create_one_hot_encoder(categories: List[str]) -> Tuple[Dict[str, int], int]:
        """
        Create one-hot encoding mapping.
        
        Args:
            categories: List of category names
            
        Returns:
            Tuple of (category_to_index_mapping, total_features)
        """
        category_map = {cat: i for i, cat in enumerate(sorted(set(categories)))}
        return category_map, len(category_map)
        
    @staticmethod
    def encode_categorical(values: List[str], 
                         category_map: Optional[Dict[str, int]] = None,
                         handle_unknown: str = 'ignore') -> Tuple[np.ndarray, Dict[str, int]]:
        """
        One-hot encode categorical values.
        
        Args:
            values: List of categorical values
            category_map: Pre-defined category mapping
            handle_unknown: How to handle unknown categories ('ignore', 'error')
            
        Returns:
            Tuple of (one_hot_matrix, category_mapping)
        """
        if category_map is None:
            category_map, _ = CategoryEncoder.create_one_hot_encoder(values)
            
        n_categories = len(category_map)
        encoded = np.zeros((len(values), n_categories))
        
        for i, value in enumerate(values):
            if value in category_map:
                encoded[i, category_map[value]] = 1
            elif handle_unknown == 'error':
                raise ValueError(f"Unknown category: {value}")
            # 'ignore' case: leave as zeros
            
        return encoded, category_map
        
    @staticmethod
    def encode_multi_label(multi_values: List[List[str]],
                          category_map: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Encode multi-label categorical data (e.g., multiple genres per movie).
        
        Args:
            multi_values: List of lists of category values
            category_map: Pre-defined category mapping
            
        Returns:
            Tuple of (multi_hot_matrix, category_mapping)
        """
        # Flatten all values to create vocabulary
        all_values = [val for sublist in multi_values for val in sublist]
        
        if category_map is None:
            category_map, _ = CategoryEncoder.create_one_hot_encoder(all_values)
            
        n_categories = len(category_map)
        encoded = np.zeros((len(multi_values), n_categories))
        
        for i, values in enumerate(multi_values):
            for value in values:
                if value in category_map:
                    encoded[i, category_map[value]] = 1
                    
        return encoded, category_map


class FeatureSelector:
    """Feature selection utilities."""
    
    @staticmethod
    def select_k_best(features: np.ndarray,
                     targets: np.ndarray,
                     k: int = 1000,
                     score_func: str = 'f_classif') -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Select k best features using univariate statistics.
        
        Args:
            features: Feature matrix
            targets: Target values
            k: Number of features to select
            score_func: Scoring function ('f_classif', 'mutual_info_classif')
            
        Returns:
            Tuple of (selected_features, selected_indices, fitted_selector)
        """
        score_functions = {
            'f_classif': f_classif,
            'mutual_info_classif': mutual_info_classif
        }
        
        if score_func not in score_functions:
            raise ValueError(f"Unknown score function: {score_func}")
            
        selector = SelectKBest(score_functions[score_func], k=k)
        selected_features = selector.fit_transform(features, targets)
        selected_indices = selector.get_support(indices=True)
        
        logger.info(f"Selected {len(selected_indices)} features out of {features.shape[1]}")
        
        return selected_features, selected_indices, selector
        
    @staticmethod
    def select_by_variance(features: np.ndarray,
                          threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select features by variance threshold.
        
        Args:
            features: Feature matrix
            threshold: Minimum variance threshold
            
        Returns:
            Tuple of (selected_features, selected_indices)
        """
        variances = np.var(features, axis=0)
        selected_indices = np.where(variances > threshold)[0]
        selected_features = features[:, selected_indices]
        
        logger.info(f"Selected {len(selected_indices)} features with variance > {threshold}")
        
        return selected_features, selected_indices
        
    @staticmethod
    def reduce_dimensionality(features: np.ndarray,
                            n_components: int = 100,
                            method: str = 'pca') -> Tuple[np.ndarray, Any]:
        """
        Reduce feature dimensionality.
        
        Args:
            features: Feature matrix
            n_components: Number of components to keep
            method: Dimensionality reduction method ('pca')
            
        Returns:
            Tuple of (reduced_features, fitted_transformer)
        """
        if method == 'pca':
            transformer = PCA(n_components=n_components)
            reduced_features = transformer.fit_transform(features)
            
            logger.info(f"Reduced dimensionality from {features.shape[1]} to {n_components}")
            logger.info(f"Explained variance ratio: {transformer.explained_variance_ratio_.sum():.3f}")
            
            return reduced_features, transformer
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")


class FeatureAnalyzer:
    """Analyze feature importance and correlations."""
    
    @staticmethod
    def analyze_feature_correlations(features: np.ndarray,
                                   feature_names: List[str],
                                   threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze feature correlations and identify highly correlated pairs.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            threshold: Correlation threshold for flagging
            
        Returns:
            Dictionary with correlation analysis results
        """
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available, correlation analysis limited")
            return {}
            
        df = pd.DataFrame(features, columns=feature_names)
        corr_matrix = df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        n_features = len(feature_names)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corr_pairs.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': corr_val
                    })
                    
        # Sort by absolute correlation
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        analysis = {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'n_high_corr_pairs': len(high_corr_pairs),
            'threshold': threshold,
            'max_correlation': corr_matrix.abs().values.max() if not corr_matrix.empty else 0
        }
        
        logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > {threshold})")
        
        return analysis
        
    @staticmethod
    def feature_importance_summary(feature_names: List[str],
                                 importances: np.ndarray,
                                 top_k: int = 20) -> Dict[str, Any]:
        """
        Create feature importance summary.
        
        Args:
            feature_names: List of feature names
            importances: Feature importance scores
            top_k: Number of top features to include
            
        Returns:
            Dictionary with importance analysis
        """
        # Sort by importance
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = importance_pairs[:top_k]
        
        # Group by feature category
        name_mapper = FeatureNameMapper()
        categories = name_mapper.get_feature_categories([name for name, _ in top_features])
        
        summary = {
            'top_features': [
                {'name': name, 'importance': float(importance)} 
                for name, importance in top_features
            ],
            'categories_represented': list(categories.keys()),
            'total_features': len(feature_names),
            'importance_stats': {
                'mean': float(np.mean(importances)),
                'std': float(np.std(importances)),
                'max': float(np.max(importances)),
                'min': float(np.min(importances))
            }
        }
        
        return summary


def save_feature_metadata(feature_names: List[str],
                         feature_descriptions: Optional[Dict[str, str]] = None,
                         output_path: str = "data/processed/feature_metadata.json") -> None:
    """
    Save feature metadata to JSON file.
    
    Args:
        feature_names: List of feature names
        feature_descriptions: Optional descriptions for features
        output_path: Path to save metadata file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    name_mapper = FeatureNameMapper()
    
    metadata = {
        'feature_count': len(feature_names),
        'feature_categories': name_mapper.get_feature_categories(feature_names),
        'features': [
            {
                'name': name,
                'description': feature_descriptions.get(name, name_mapper.get_description(name)) if feature_descriptions else name_mapper.get_description(name),
                'category': _get_feature_category(name)
            }
            for name in feature_names
        ],
        'created_at': pd.Timestamp.now().isoformat() if PANDAS_AVAILABLE else "unknown"
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Feature metadata saved to: {output_path}")


def _get_feature_category(feature_name: str) -> str:
    """Get category for a single feature name."""
    if feature_name.startswith('text_'):
        return 'text'
    elif feature_name.startswith('genre_'):
        return 'genre'
    elif feature_name.startswith('decade_'):
        return 'temporal'
    elif feature_name.startswith('runtime_'):
        return 'runtime'
    elif feature_name.startswith('rating_'):
        return 'rating'
    elif feature_name.startswith(('cast_', 'crew_')):
        return 'cast_crew'
    else:
        return 'other'


if __name__ == "__main__":
    # Example usage
    sample_feature_names = [
        'text_action', 'text_movie', 'text_great',
        'genre_action', 'genre_drama', 'genre_comedy',
        'decade_contemporary', 'decade_modern',
        'runtime_medium', 'runtime_normalized',
        'rating_average', 'rating_high',
        'cast_size', 'crew_avg_age'
    ]
    
    # Test feature name mapping
    mapper = FeatureNameMapper()
    
    print("Feature Descriptions:")
    for name in sample_feature_names[:5]:
        print(f"  {name}: {mapper.get_description(name)}")
        
    print(f"\nFeature Categories:")
    categories = mapper.get_feature_categories(sample_feature_names)
    for cat, features in categories.items():
        print(f"  {cat}: {len(features)} features")
        
    # Test scaling utilities
    sample_features = np.random.randn(100, len(sample_feature_names))
    scaled_features, scalers = ScalingUtilities.scale_feature_groups(
        sample_features, sample_feature_names
    )
    print(f"\nScaled features shape: {scaled_features.shape}")
    print(f"Fitted scalers: {list(scalers.keys())}")
    
    # Save metadata
    save_feature_metadata(sample_feature_names, output_path="temp_metadata.json")
    print("Feature metadata saved!")