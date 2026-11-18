"""
tensorsat: Implementation of the rank-bounded singlet-swap tensor decomposition
for quantum, SAT, and AI acceleration.
"""

from .uncrossing import s_factor, eval_slice_A, generate_uncrossing_tensor
from .attention import SingletAttention, KVCache
from .model_wrapper import TensorSATBlock
from .cnf_preprocessor import CNFFormula
from .scientific import IEEETensorContractor, GridPoissonSolver

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
]
