"""
Bandexa: contextual bandits with Neural Thompson Sampling (NeuralLinearTS) in PyTorch.
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("bandexa")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
