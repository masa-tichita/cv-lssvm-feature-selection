"""SVM-CV solver: the cross-validation-based MIO formulation of Purcaru et al. (2026).

Paper:
Mixed-Integer Optimization for Cross-Validated Feature Selection in Linear
Classification, Journal of Optimization Theory and Applications, 2026.

Implements the formulation of Sect. 3.3 of that paper:
- Upper-level objective: mean validation hinge loss over the folds.
- Lower-level SVM optimality: primal feasibility, dual feasibility, stationarity,
  and reversed weak duality, written as a single-level MIQO.

SVM-CV is evaluated by the features selected by this MIO solution and the
per-fold models (w^(k), b^(k)) of the same solution.
"""

import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from modules.algo.common.solver import (
    STOPPED_WITH_SOLUTION,
    configure_mio_model,
    fold_models_from_vars,
    gurobi_log_file,
    log_solved,
    mean_fold_decision,
    mip_gap,
    optimize_mio_model,
    selected_from_z,
    var_values,
)
from modules.algo.svm_cv.types import SVMCVParams, SVMCVSolution
from modules.data.splits import split_k_fold_indices
from utils.ml import normalize_binary_labels, predict_labels

# SVM-CV also accepts a solution found before SUBOPTIMAL / NUMERIC stops
_ACCEPTED_STATUSES = STOPPED_WITH_SOLUTION | {GRB.SUBOPTIMAL, GRB.NUMERIC}


def _paper_stationarity_big_m(X: NDArray[np.float64], C: float) -> float:
    """Compute the common Big-M of Purcaru et al. (2026), Eq. (29)."""
    if X.shape[1] == 0:
        return 0.0
    return float(C * np.max(np.sum(np.abs(X), axis=0)))


class SVMCVSolver:
    """SVM-CV solver."""

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        params: SVMCVParams,
    ) -> SVMCVSolution:
        """Solve the SVM-CV MIO for a given C."""
        start_time = time.perf_counter()
        y_normalized = normalize_binary_labels(y)

        train_folds, val_folds = split_k_fold_indices(
            n_samples=X.shape[0],
            labels=y_normalized,
            n_folds=params.n_folds,
            shuffle=params.shuffle,
            random_seed=params.seed,
        )

        model, z_vars, w_vars, b_vars = self._build_model(
            X=X,
            y=y_normalized,
            train_folds=train_folds,
            val_folds=val_folds,
            params=params,
        )
        log_name = f"svm_cv_C={params.C:g}_{int(time.time() * 1000)}.log"
        configure_mio_model(model, params, gurobi_log_file(params.log_dir, log_name))
        if params.optimality_tol is not None:
            model.Params.OptimalityTol = params.optimality_tol
        optimize_mio_model(model, "SVM-CV", _ACCEPTED_STATUSES)

        _, p = X.shape
        z = var_values(z_vars, p)
        fold_models = fold_models_from_vars(w_vars, b_vars, len(train_folds), p)
        selected_features = selected_from_z(z)

        obj_value = float(model.ObjVal)
        solver_status = int(model.Status)
        opt_gap = mip_gap(model)
        model.dispose()
        solve_time = time.perf_counter() - start_time

        solution = SVMCVSolution(
            z=z,
            selected_features=selected_features,
            n_selected_features=len(selected_features),
            C=params.C,
            obj_value=obj_value,
            opt_gap=opt_gap,
            solve_time=solve_time,
            solver_status=solver_status,
            fold_models=fold_models,
        )
        log_solved(
            f"SVM-CV (C={params.C})",
            solution.n_selected_features,
            p,
            solution.obj_value,
            solution.solve_time,
        )
        return solution

    def predict(
        self,
        X: NDArray[np.float64],
        solution: SVMCVSolution,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict with the mean decision value of the per-fold w, b of the MIQO."""
        if not solution.fold_models:
            raise ValueError("fold_models are not available")
        decision = mean_fold_decision(X, solution.fold_models)
        return predict_labels(decision), decision

    def _build_model(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        train_folds: list[NDArray[np.intp]],
        val_folds: list[NDArray[np.intp]],
        params: SVMCVParams,
    ) -> tuple[gp.Model, gp.tupledict, gp.tupledict, gp.tupledict]:
        """Build the MIQO of SVM-CV (Sect. 3.3 of Purcaru et al., 2026)."""
        _, p = X.shape
        k_count = len(train_folds)
        C = params.C
        max_features = params.max_features if params.max_features is not None else p
        if max_features > p:
            raise ValueError(f"max_features ({max_features}) > p ({p})")

        # Proposition 3.3: 1/2||w||^2 <= 2C gives U = 2*sqrt(C).
        U = 2.0 * np.sqrt(C)
        # Proposition 3.4, Eq. (29): Big-M shared by all folds, as in the paper's experiments.
        # M = C * max_j sum_{i in [n]} |x_ij|
        stationarity_M = _paper_stationarity_big_m(X, C)

        mdl = gp.Model("svm_cv")
        t = mdl.addVars(p, vtype=GRB.BINARY, name="t")
        w = mdl.addVars(k_count, p, lb=-U, ub=U, name="w")
        b = mdl.addVars(k_count, lb=-GRB.INFINITY, name="b")

        eta: dict[tuple[int, int], gp.Var] = {}
        xi: dict[tuple[int, int], gp.Var] = {}
        alpha: dict[tuple[int, int], gp.Var] = {}

        for k, val_idx in enumerate(val_folds):
            for i in val_idx:
                eta[k, int(i)] = mdl.addVar(lb=0.0, name=f"eta[{k},{int(i)}]")

        for k, train_idx in enumerate(train_folds):
            alpha_ub = C / len(train_idx)
            for i in train_idx:
                i_int = int(i)
                xi[k, i_int] = mdl.addVar(lb=0.0, name=f"xi[{k},{i_int}]")
                alpha[k, i_int] = mdl.addVar(
                    lb=0.0,
                    ub=alpha_ub,
                    name=f"alpha[{k},{i_int}]",
                )

        mdl.setObjective(
            gp.quicksum(
                eta[k, int(i)] / (k_count * len(val_folds[k]))
                for k in range(k_count)
                for i in val_folds[k]
            ),
            GRB.MINIMIZE,
        )

        mdl.addConstr(
            gp.quicksum(t[j] for j in range(p)) <= max_features,
            name="cardinality",
        )

        for k in range(k_count):
            for i in val_folds[k]:
                i_int = int(i)
                margin = (
                    gp.quicksum(w[k, j] * float(X[i_int, j]) for j in range(p)) + b[k]
                )
                mdl.addConstr(
                    eta[k, i_int] >= 1.0 - float(y[i_int]) * margin,
                    name=f"val_hinge[{k},{i_int}]",
                )

            for i in train_folds[k]:
                i_int = int(i)
                margin = (
                    gp.quicksum(w[k, j] * float(X[i_int, j]) for j in range(p)) + b[k]
                )
                mdl.addConstr(
                    xi[k, i_int] >= 1.0 - float(y[i_int]) * margin,
                    name=f"train_hinge[{k},{i_int}]",
                )

            mdl.addConstr(
                gp.quicksum(alpha[k, int(i)] * float(y[int(i)]) for i in train_folds[k])
                == 0.0,
                name=f"dual_balance[{k}]",
            )

            mdl.addQConstr(
                gp.quicksum(w[k, j] * w[k, j] for j in range(p))
                + (C / len(train_folds[k]))
                * gp.quicksum(xi[k, int(i)] for i in train_folds[k])
                - gp.quicksum(alpha[k, int(i)] for i in train_folds[k])
                <= 0.0,
                name=f"strong_duality[{k}]",
            )

            M = stationarity_M
            for j in range(p):
                mdl.addConstr(w[k, j] <= U * t[j], name=f"w_on_upper[{k},{j}]")
                mdl.addConstr(w[k, j] >= -U * t[j], name=f"w_on_lower[{k},{j}]")

                stationarity = (
                    gp.quicksum(
                        alpha[k, int(i)] * float(y[int(i)]) * float(X[int(i), j])
                        for i in train_folds[k]
                    )
                    - w[k, j]
                )
                mdl.addConstr(
                    stationarity <= M * (1.0 - t[j]),
                    name=f"stationarity_upper[{k},{j}]",
                )
                mdl.addConstr(
                    stationarity >= -M * (1.0 - t[j]),
                    name=f"stationarity_lower[{k},{j}]",
                )

        mdl.update()
        return mdl, t, w, b
