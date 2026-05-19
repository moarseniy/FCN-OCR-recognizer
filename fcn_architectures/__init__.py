from .legacy_fcn import FullyConvTextRecognizer, LegacyFCN
from .registry import available_architectures, create_model, normalize_architecture_name

__all__ = [
    "FullyConvTextRecognizer",
    "LegacyFCN",
    "available_architectures",
    "create_model",
    "normalize_architecture_name",
]
