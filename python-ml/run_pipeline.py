#!/usr/bin/env python3
"""
MovieRecs Data Pipeline Runner

Simple script to execute the data pipeline with common configurations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_prep import main

if __name__ == "__main__":
    main()