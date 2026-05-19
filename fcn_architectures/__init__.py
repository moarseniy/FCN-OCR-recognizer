from .legacy_fcn import FullyConvTextRecognizer, LegacyFCN
from .registry import available_architectures, create_model, normalize_architecture_name
from .vertical_segmentator_fcn import VerticalSegmentatorFCN

__all__ = [
    "FullyConvTextRecognizer",
    "LegacyFCN",
    "VerticalSegmentatorFCN",
    "available_architectures",
    "create_model",
    "normalize_architecture_name",
]
