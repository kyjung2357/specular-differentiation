"""Result model shared by the scalar ODE methods."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ODEResult:
    """Values produced by a scalar specular ODE method.

    ``t`` and ``u`` contain the initial value and every accepted step, while
    ``sigma`` contains the scale associated with each represented interval.
    In automatic ``minimize_defect`` mode, ``0.0`` and ``inf`` denote the
    zero- and infinite-scale limiting methods, respectively.
    ``number_of_field_evaluations`` counts calls to ``F(t, u)`` made by the
    solver.
    """

    t: FloatArray
    u: FloatArray
    sigma: FloatArray
    number_of_field_evaluations: int


__all__ = ["ODEResult"]
