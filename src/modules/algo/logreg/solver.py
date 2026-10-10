"""LogReg MILO: MILO formulation for logistic regression.

Implementation of Sato et al. (2016), "Feature subset selection for logistic
regression via mixed integer optimization", Comput. Optim. Appl.
- Piecewise-linear approximation of the logistic loss.
- Solved as a MILO with Gurobi.

Formulation in the paper (Eqs. 12-15):
    minimize  2 * sum_i(t_i) + F * (sum_j(z_j) + 1)
    subject to:
        t_i >= f'(v_k) * (y_i * (w^T x_i + b) - v_k) + f(v_k)  (∀i, ∀k)
        z_j = 0 => w_j = 0  (∀j)
        z_j ∈ {0, 1}  (∀j)

where:
    f(v) = log(1 + exp(-v))  (logistic loss)
    f'(v) = -exp(-v) / (1 + exp(-v)) = -1 / (1 + exp(v))  (derivative)
"""

import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from modules.algo.common.solver import (
    configure_mio_model,
    gurobi_log_file,
    log_solved,
    mip_gap,
    optimize_mio_model,
    selected_from_z,
    var_values,
)
from modules.algo.logreg.types import LogRegParams, LogRegSolution
from utils.ml import linear_decision, normalize_binary_labels, predict_labels

# Sets of tangent points recommended in the paper
# V1: 5 points, V2: 9 points, V3: 17 points
TANGENT_SETS: dict[str, list[float]] = {
    "V1": [0.0, -1.90, 1.90],  # ±∞ handled separately
    "V2": [0.0, -0.89, 0.89, -1.90, 1.90, -3.55, 3.55],
    "V3": [
        0.0,
        -0.44,
        0.44,
        -0.89,
        0.89,
        -1.37,
        1.37,
        -1.90,
        1.90,
        -2.63,
        2.63,
        -3.55,
        3.55,
        -5.16,
        5.16,
    ],
}


def _logistic_loss(v: float) -> float:
    """Logistic loss f(v) = log(1 + exp(-v))."""
    # For numerical stability
    if v > 20:
        return np.exp(-v)
    elif v < -20:
        return -v
    return np.log(1 + np.exp(-v))


def _logistic_loss_derivative(v: float) -> float:
    """Derivative of the logistic loss f'(v) = -1 / (1 + exp(v))."""
    # For numerical stability
    if v > 20:
        return -np.exp(-v)
    elif v < -20:
        return -1.0
    return -1.0 / (1 + np.exp(v))


class LogRegMILO:
    """LogReg MILO (feature selection for logistic regression).

    Paper: Feature subset selection for logistic regression via mixed integer optimization
    (Sato et al., 2016)

    Usage:
        milo = LogRegMILO()
        solution = milo.train(X, y, params)
        predictions, decisions = milo.predict(X_test, solution)
    """

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        params: LogRegParams,
    ) -> LogRegSolution:
        """Build and optimize the LogReg MILO model.

        1. Normalize labels to ±1.
        2. Build the MILO model (piecewise-linear approximation).
        3. Solve with Gurobi.
        4. Extract the solution.
        """
        start_time = time.time()

        # 1. Normalize labels
        y_normalized = normalize_binary_labels(y)

        # 2. Build the MILO model
        model, z_vars, w_vars, b_var, t_vars = self._build_model(
            X=X,
            y=y_normalized,
            params=params,
        )
        log_name = f"logreg_milo_{int(time.time() * 1000)}.log"
        configure_mio_model(model, params, gurobi_log_file(params.log_dir, log_name))

        # 3. Optimize
        optimize_mio_model(model, "LogReg MILO")

        # 4. Extract the solution
        p = X.shape[1]
        z = var_values(z_vars, p)
        selected_features = selected_from_z(z)
        solution = LogRegSolution(
            z=z,
            w=var_values(w_vars, p),
            b=float(b_var.X),
            selected_features=selected_features,
            n_selected_features=len(selected_features),
            obj_value=float(model.ObjVal),
            opt_gap=mip_gap(model),
            solve_time=time.time() - start_time,
            solver_status=int(model.Status),
        )
        log_solved(
            "LogReg MILO",
            solution.n_selected_features,
            p,
            solution.obj_value,
            solution.solve_time,
        )

        # Dispose of the Gurobi model to avoid memory leaks
        model.dispose()

        return solution

    def predict(
        self,
        X: NDArray[np.float64],
        solution: LogRegSolution,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict with decision = X @ w + b and y_pred = sign(decision) (0 → +1).

        Returns:
            (predictions, decision_values)
        """
        decision = linear_decision(X, solution.w, solution.b)
        return predict_labels(decision), decision

    def _build_model(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        params: LogRegParams,
    ) -> tuple[gp.Model, gp.tupledict, gp.tupledict, gp.Var, gp.tupledict]:
        """Build the MILO model.

        Returns:
            (model, z, w, b, t): Gurobi model and variables.
        """
        n, p = X.shape

        # Get the tangent points
        tangent_points = TANGENT_SETS.get(params.tangent_set, TANGENT_SETS["V3"])

        # Build the Gurobi model
        mdl = gp.Model("logreg_milo")

        # Variables
        z = mdl.addVars(p, vtype=GRB.BINARY, name="z")
        w = mdl.addVars(p, lb=-GRB.INFINITY, name="w")
        b = mdl.addVar(lb=-GRB.INFINITY, name="b")
        t = mdl.addVars(n, lb=0.0, name="t")  # Upper bound on the logistic loss

        # Objective: 2 * sum(t_i) + F * (sum(z_j) + 1)
        # Eq. (12) in the paper: logistic loss + penalty on the number of features
        penalty_term = params.penalty_F * (gp.quicksum(z[j] for j in range(p)) + 1)
        mdl.setObjective(
            2 * gp.quicksum(t[i] for i in range(n)) + penalty_term, GRB.MINIMIZE
        )

        # Cardinality constraint (optional)
        if params.max_features is not None:
            mdl.addConstr(
                gp.quicksum(z[j] for j in range(p)) <= params.max_features,
                name="budget",
            )

        # Constraints
        # 1) Piecewise-linear approximation constraints
        # t_i >= f'(v_k) * (y_i * (w^T x_i + b) - v_k) + f(v_k)
        for i in range(n):
            yi = float(y[i])
            xi = X[i, :]

            # Finite tangent points
            for v_k in tangent_points:
                f_vk = _logistic_loss(v_k)
                f_prime_vk = _logistic_loss_derivative(v_k)

                # y_i * (w^T x_i + b)
                margin = yi * (gp.quicksum(w[j] * float(xi[j]) for j in range(p)) + b)

                # f'(v_k) * (margin - v_k) + f(v_k)
                rhs = f_prime_vk * (margin - v_k) + f_vk
                mdl.addConstr(t[i] >= rhs, name=f"pwl[{i},{v_k:.2f}]")

            # Tangent as v → -∞: t_i >= -v (i.e., t_i >= -(y_i * (w^T x_i + b)))
            margin_neg = yi * (gp.quicksum(w[j] * float(xi[j]) for j in range(p)) + b)
            mdl.addConstr(t[i] >= -margin_neg, name=f"pwl_neg_inf[{i}]")

            # Tangent as v → +∞: t_i >= 0 (always satisfied, so not added)

        # 2) z_j = 0 => w_j = 0 (indicator constraint)
        for j in range(p):
            mdl.addConstr(
                (z[j] == 0) >> (w[j] == 0),
                name=f"ind_zero_w[{j}]",
            )

        mdl.update()
        return mdl, z, w, b, t
