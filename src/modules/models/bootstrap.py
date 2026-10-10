"""Model registration bootstrap.

Models are registered by calling register_all() explicitly,
not implicitly through import side effects.
"""

import math

from modules.models.base import register_model
from modules.models.logreg import LogRegSpec
from modules.models.proposed import ProposedSpec
from modules.models.svm_cv import SVMCVSpec
from modules.models.svm_l1 import SVML1Spec
from modules.models.svm_rfe import SVMRFESpec


def register_all() -> None:
    """Register all models in the registry.

    Call once at application startup.
    """
    register_model(ProposedSpec("proposed_hl", variant="hl"))  # type: ignore[arg-type]
    register_model(ProposedSpec("proposed_ls", variant="ls"))  # type: ignore[arg-type]
    register_model(SVMCVSpec())  # type: ignore[arg-type]
    register_model(SVML1Spec())  # type: ignore[arg-type]
    register_model(SVMRFESpec())  # type: ignore[arg-type]
    register_model(LogRegSpec("logreg_aic", lambda n: 2.0))  # type: ignore[arg-type]
    register_model(LogRegSpec("logreg_bic", lambda n: math.log(n)))  # type: ignore[arg-type]
