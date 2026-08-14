"""Internal mathematical types for the JAX backend."""

from collections.abc import Callable

from jax import Array


# JAX represents mathematical scalars, vectors, and matrices with the same
# array class. Their ranks are validated at runtime.
type Scalar = Array
type Vector = Array
type Matrix = Array


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
