"""
Land Cover Segmentation Package

A comprehensive deep learning system for semantic segmentation of satellite imagery
to classify land cover into 8 distinct classes using transfer learning.
"""

__version__ = "1.0.0"
__author__ = "CALIDADSOFTWARE Team"
__email__ = "contact@calidadsoftware.com"

from .utils import *
from .models import *
from .data import *
from .training import *
from .evaluation import *