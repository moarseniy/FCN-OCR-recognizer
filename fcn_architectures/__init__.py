from .legacy_fcn import FullyConvTextRecognizer, LegacyFCN
from .legacy_fcn_wide import LegacyFCNWide
from .registry import available_architectures, create_model, normalize_architecture_name
from .residual_temporal_fcn import ResidualTemporalFCN
from .vertical_segmentator_fcn import VerticalSegmentatorFCN

__all__ = [
    "FullyConvTextRecognizer",
    "LegacyFCN",
    "LegacyFCNWide",
    "ResidualTemporalFCN",
    "VerticalSegmentatorFCN",
    "available_architectures",
    "create_model",
    "normalize_architecture_name",
]
