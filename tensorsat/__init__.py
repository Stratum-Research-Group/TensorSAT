"""
tensorsat: Implementation of the rank-bounded singlet-swap tensor decomposition
for quantum, SAT, AI, and infinite-context acceleration.
"""

from .uncrossing import s_factor, eval_slice_A, generate_uncrossing_tensor
from .attention import SingletAttention, KVCache
from .model_wrapper import TensorSATBlock
from .cnf_preprocessor import CNFFormula
from .scientific import IEEETensorContractor, GridPoissonSolver
from .stabilizer import DecayStabilizer, StateRMSNorm
from .hybrid import HybridSingletAttention
from .adapter import SoftmaxToSingletAdapter

__version__ = "0.2.0"

__all__ = [
    "s_factor",
    "eval_slice_A",
    "generate_uncrossing_tensor",
    "SingletAttention",
    "KVCache",
    "TensorSATBlock",
    "CNFFormula",
    "IEEETensorContractor",
    "GridPoissonSolver",
    "DecayStabilizer",
    "StateRMSNorm",
    "HybridSingletAttention",
    "SoftmaxToSingletAdapter",
]
