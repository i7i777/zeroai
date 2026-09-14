"""Reproduction of Nanda et al. 2023 "Progress measures for grokking via
mechanistic interpretability" (arXiv:2301.05217).

Faithful PyTorch re-implementation of the mainline one-layer transformer
trained on modular addition, plus the Fourier/mechanistic analysis and the
progress measures (restricted / excluded loss, Gini coefficient, weight norm).
"""
from .config import GrokConfig, mainline, load_config
from .data import make_modular_addition_data, ModularAdditionData
from .model import OneLayerTransformer
from . import measures, fourier

__version__ = "0.1.0"
