"""Type definitions of the MIO-based logistic regression (LogReg-AIC / LogReg-BIC).

MIO (mixed-integer optimization) for feature selection in logistic regression
(Sato et al., 2016, Comput. Optim. Appl.).
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from modules.algo.common.types import MIOParams, MIOSolution


class LogRegParams(MIOParams):
    """Parameters of the LogReg MILO.

    The Gurobi log is written to ``log_dir/logreg_milo_<timestamp>.log`` when
    ``log_dir`` is set.
    """

    # Penalty coefficient F (penalty on the number of features)
    # F=2 → AIC, F=log(n) → BIC
    penalty_F: float = Field(default=2.0, ge=0.0)
    # Tangent point set: "V1" (5 points), "V2" (9 points), "V3" (17 points)
    tangent_set: str = Field(default="V3")


class LogRegSolution(MIOSolution):
    """Optimization result of the LogReg MILO: weight vector w (p,) and bias b."""

    w: NDArray[np.float64]
    b: float
