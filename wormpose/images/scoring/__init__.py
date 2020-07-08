"""
Module containing everything related to the image similarity,
including the systems to distribute the image score calculation to several processes when analyzing a video
"""

from wormpose.images.scoring.scoring_data_manager import BaseScoringDataManager, ScoringDataManager
from wormpose.images.scoring.results_scoring import ResultsScoring
