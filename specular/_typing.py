from collections.abc import Sequence
from typing import Callable, TypeAlias
import numpy as np

# Scalar inputs and outputs used by one-dimensional routines.
Scalar: TypeAlias = int | float | np.number

# Vector inputs and outputs accepted by multidimensional routines.
Vector: TypeAlias = list | np.ndarray

# A scalar-valued function of one scalar variable.
ScalarToScalarFunc: TypeAlias = Callable[[Scalar], Scalar]

# A scalar-valued function of one vector variable.
VectorToScalarFunc: TypeAlias = Callable[[Vector], Scalar]

# A vector-valued function of one scalar variable.
ScalarToVectorFunc: TypeAlias = Callable[[Scalar], Vector]

# A vector-valued function of one vector variable.
VectorToVectorFunc: TypeAlias = Callable[[Vector], Vector]

# A single component function f_i(x), used when components are passed as
# a sequence [f_1, ..., f_m].
ComponentFunc: TypeAlias = ScalarToScalarFunc | VectorToScalarFunc

# An indexed component provider f_j(x, j), used when one callable computes
# the selected component from its index.
IndexedComponentFunc: TypeAlias = Callable[[Scalar, int], Scalar] | Callable[[Vector, int], Scalar]

# Component functions for stochastic methods. Users may pass either a sequence
# of component functions [f_1, ..., f_m] or one indexed provider f_j(x, j).
ComponentFuncs: TypeAlias = Sequence[ComponentFunc] | IndexedComponentFunc