# src/__init__.py
"""
Fake News Detection Package
Professional machine learning system for detecting fake news
"""

__version__ = "1.0.0"
__author__ = "AI Developer"
__email__ = "developer@example.com"

# Package imports
from . import utils
from . import data_preprocessing
from . import feature_engineering
from . import model_training
from . import model_evaluation

__all__ = [
    "utils",
    "data_preprocessing", 
    "feature_engineering",
    "model_training",
    "model_evaluation"
]