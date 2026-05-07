from caps.pipeline import CAPSPipeline
from caps.acquisition.signal_matrix import SignalMatrix
from caps.compression.entropy_weighting import EntropyWeighter
from caps.compression.dimensionality import BehavioralCompressor
from caps.compression.clustering import TwoStageClustering
from caps.refinement.qualitative import QualitativeRefinement
from caps.refinement.alignment import AlignmentValidator
from caps.segmentation.profiles import SegmentProfiler
from caps.segmentation.maps import SegmentMap

__version__ = "1.0.0"
__author__ = "Ahmed Elsayed"
__license__ = "MIT"

__all__ = [
    "CAPSPipeline",
    "SignalMatrix",
    "EntropyWeighter",
    "BehavioralCompressor",
    "TwoStageClustering",
    "QualitativeRefinement",
    "AlignmentValidator",
    "SegmentProfiler",
    "SegmentMap",
]