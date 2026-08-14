"""Internal mathematical types for the NumPy backend."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


# Real scalar values. Complex numbers are intentionally excluded.
type Scalar = int | float | np.integer[Any] | np.floating[Any]

# Internal representation of a finite-dimensional real vector.
type Vector = npt.NDArray[np.float64]

# Internal representation of a finite-dimensional real matrix.
type Matrix = npt.NDArray[np.float64]


# R -> R
type ScalarToScalarFunc = Callable[[Scalar], Scalar]

# R^n -> R
type VectorToScalarFunc = Callable[[Vector], Scalar]

# R -> R^n
type ScalarToVectorFunc = Callable[[Scalar], Vector]

# R^n -> R^m
type VectorToVectorFunc = Callable[[Vector], Vector]


__all__ = [
    "Scalar",
    "Vector",
    "Matrix",
    "ScalarToScalarFunc",
    "VectorToScalarFunc",
    "ScalarToVectorFunc",
    "VectorToVectorFunc",
]
