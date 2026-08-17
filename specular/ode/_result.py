"""Result model shared by the scalar ODE schemes."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ODEResult:
    """Values produced by a scalar specular ODE scheme.

    ``t`` and ``u`` contain the initial value and every accepted step, while
    ``sigma`` contains the scale associated with each represented interval.
    ``number_of_field_evaluations`` counts calls to ``F(t, u)`` made by the
    solver.
    """

    t: FloatArray
    u: FloatArray
    sigma: FloatArray
    number_of_field_evaluations: int


__all__ = ["ODEResult"]
