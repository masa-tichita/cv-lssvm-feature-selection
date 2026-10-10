"""Solver of the proposed method (Proposed-HL / Proposed-LS)."""

import time
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray

from modules.algo.common.solver import (
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
from modules.algo.proposed.types import ProposedParams, ProposedSolution, Variant
from modules.data.splits import split_k_fold_indices
from utils.gurobi_io import save_gurobi_model
from utils.ml import normalize_binary_labels, predict_labels


class ProposedSolver:
    """Cross-validated feature selection with LS-SVMs (proposed method).

    See the accompanying paper:
    - Lower-level problem: train an LS-SVM on each fold k → (w^(k), b^(k)),
      written through its optimality conditions.
    - Upper-level problem: minimize the validation loss on the held-out samples.
    - Validation model selected by ``variant``: "hl" (Proposed-HL,
      hinge-loss-based validation model; a MILO problem) or "ls" (Proposed-LS,
      least-squares-based validation model; a MIQO problem).

    Usage:
        solver = ProposedSolver()
        solution = solver.train(X, y, params)
        predictions, decisions = solver.predict(X_test, solution)
    """

    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        params: ProposedParams,
        save_model: bool = False,
        output_dir: str | None = None,
    ) -> ProposedSolution:
        """Build and optimize the MIO model (MILO for HL, MIQO for LS).

        1. Normalize labels to ±1.
        2. Split into K folds (StratifiedKFold preferred).
        3. Build the MIO model.
        4. Solve with Gurobi.
        5. Extract the solution and return a ProposedSolution.
        """
        start_time = time.time()
        label = f"Proposed-{params.variant.upper()}"

        # 1. Normalize labels
        y_normalized = normalize_binary_labels(y)

        # 2. K-fold split (indices only)
        T_folds, V_folds = split_k_fold_indices(
            n_samples=X.shape[0],
            labels=y_normalized,
            n_folds=params.n_folds,
            shuffle=params.shuffle,
            random_seed=params.seed,
        )

        # 3. Build the MIO model
        model, z_vars, w_vars, b_vars = self._build_model(
            X=X,
            y=y_normalized,
            T_folds=T_folds,
            V_folds=V_folds,
            gamma=params.gamma,
            max_feature_num=params.max_features,
            variant=params.variant,
            var_bound_w=params.var_bound_w,
            var_bound_b=params.var_bound_b,
            bigm_link=params.bigm_link,
        )
        configure_mio_model(
            model, params, gurobi_log_file(params.log_dir, f"gamma={params.gamma}.log")
        )

        # Save LP/MPS (optional)
        if save_model and output_dir:
            save_gurobi_model(
                model,
                base_dir=Path(output_dir),
                model_name="proposed",
                variant=params.variant,
                n_folds=params.n_folds,
                gamma=params.gamma,
            )

        # 4. Optimize
        optimize_mio_model(model, label)

        # 5. Extract the solution
        p = X.shape[1]
        z = var_values(z_vars, p)
        selected_features = selected_from_z(z)
        solution = ProposedSolution(
            z=z,
            fold_models=fold_models_from_vars(w_vars, b_vars, len(T_folds), p),
            selected_features=selected_features,
            n_selected_features=len(selected_features),
            obj_value=float(model.ObjVal),
            opt_gap=mip_gap(model),
            solve_time=time.time() - start_time,
            solver_status=int(model.Status),
        )
        log_solved(
            label, solution.n_selected_features, p, solution.obj_value, solution.solve_time
        )

        # Dispose of the Gurobi model to avoid memory leaks
        model.dispose()

        return solution

    def predict(
        self,
        X: NDArray[np.float64],
        solution: ProposedSolution,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict with the mean decision value of all folds.

        decision = mean([X @ w^(k) + b^(k) for k in K]),
        y_pred = sign(decision) with 0 mapped to +1.

        Returns:
            (predictions, decision_values)
        """
        decision = mean_fold_decision(X, solution.fold_models)
        return predict_labels(decision), decision

    def _build_model(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        T_folds: list[NDArray[np.intp]],
        V_folds: list[NDArray[np.intp]],
        gamma: float,
        max_feature_num: int | None,
        variant: Variant,
        var_bound_w: float | None = None,
        var_bound_b: float | None = None,
        bigm_link: bool = False,
    ) -> tuple[gp.Model, gp.tupledict, gp.tupledict, gp.tupledict]:
        """Build the MIO model (internal; MILO for HL, MIQO for LS).

        Returns:
            (model, z, w, b): Gurobi model and variable dicts.
        """
        n, p = X.shape
        K = len(T_folds)
        is_ls = variant == "ls"

        # Precompute per-fold statistics
        Q: list[NDArray[np.float64]] = []
        s: list[NDArray[np.float64]] = []
        r: list[NDArray[np.float64]] = []
        Sy: list[float] = []
        m: list[int] = []

        for k in range(K):
            Tk = np.asarray(T_folds[k], dtype=np.intp)
            Xk = X[Tk, :]
            yk = y[Tk]

            Q.append(Xk.T @ Xk)
            s.append(Xk.sum(axis=0))
            r.append(Xk.T @ yk)
            Sy.append(float(yk.sum()))
            m.append(len(Tk))

        # Build the Gurobi model
        mdl = gp.Model(f"proposed_{variant}")

        # Variables
        z = mdl.addVars(p, vtype=GRB.BINARY, name="z")
        w_lb = -var_bound_w if var_bound_w is not None else -GRB.INFINITY
        w_ub = var_bound_w if var_bound_w is not None else GRB.INFINITY
        b_lb = -var_bound_b if var_bound_b is not None else -GRB.INFINITY
        b_ub = var_bound_b if var_bound_b is not None else GRB.INFINITY
        w = mdl.addVars(K, p, lb=w_lb, ub=w_ub, name="w")
        b = mdl.addVars(K, lb=b_lb, ub=b_ub, name="b")

        # Slack variables xi[k, i] (validation samples only)
        xi: dict[tuple[int, int], gp.Var] = {}
        for k in range(K):
            for i in V_folds[k]:
                lb = -GRB.INFINITY if is_ls else 0.0
                xi[k, int(i)] = mdl.addVar(lb=lb, name=f"xi[{k},{int(i)}]")

        # Objective
        if is_ls:
            mdl.setObjective(
                gp.quicksum(xi[k, int(i)] * xi[k, int(i)] for k, i in xi.keys()),
                GRB.MINIMIZE,
            )
        else:
            mdl.setObjective(gp.quicksum(xi.values()), GRB.MINIMIZE)

        # Cardinality constraint (optional)
        if max_feature_num is not None:
            mdl.addConstr(
                gp.quicksum(z[j] for j in range(p)) == max_feature_num, name="budget"
            )

        # Constraints

        # 1) Margin constraints (validation samples)
        for k in range(K):
            Vk = np.asarray(V_folds[k], dtype=np.intp)
            for i in Vk:
                expr = gp.quicksum(w[k, j] * float(X[i, j]) for j in range(p)) + b[k]
                if is_ls:
                    mdl.addConstr(
                        float(y[i]) * expr == 1.0 - xi[k, int(i)],
                        name=f"margin[{k},{int(i)}]",
                    )
                else:
                    mdl.addConstr(
                        float(y[i]) * expr + xi[k, int(i)] >= 1.0,
                        name=f"margin[{k},{int(i)}]",
                    )

        # 2) z_j = 0 => w_j^{(k)} = 0
        for k in range(K):
            for j in range(p):
                mdl.addConstr(
                    (z[j] == 0) >> (w[k, j] == 0),
                    name=f"ind_zero_w[{k},{j}]",
                )

        # 3) z_j = 1 => LS-SVM KKT conditions
        for k in range(K):
            Qk = Q[k]
            sk = s[k]
            rk = r[k]
            for j in range(p):
                expr = (
                    w[k, j]
                    + gamma
                    * gp.quicksum(float(Qk[j, ell]) * w[k, ell] for ell in range(p))
                    + gamma * float(sk[j]) * b[k]
                )
                rhs = gamma * float(rk[j])
                mdl.addConstr(
                    (z[j] == 1) >> (expr == rhs),
                    name=f"ind_lssvm_w[{k},{j}]",
                )

        # 4) Big-M linking constraints (tighten the LP relaxation)
        if bigm_link and var_bound_w is not None:
            W = var_bound_w
            for k in range(K):
                for j in range(p):
                    mdl.addConstr(
                        w[k, j] <= W * z[j],
                        name=f"bigm_upper[{k},{j}]",
                    )
                    mdl.addConstr(
                        w[k, j] >= -W * z[j],
                        name=f"bigm_lower[{k},{j}]",
                    )

        # 5) KKT condition for the bias (always active)
        for k in range(K):
            expr = m[k] * b[k] + gp.quicksum(
                float(s[k][ell]) * w[k, ell] for ell in range(p)
            )
            mdl.addConstr(expr == float(Sy[k]), name=f"b_def[{k}]")

        mdl.update()
        return mdl, z, w, b
