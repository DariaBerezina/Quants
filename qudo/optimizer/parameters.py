"""
Классическая часть QAOA: подбор параметров (γ, β).

По умолчанию используется COBYLA из scipy — тот же метод, что и в исходной
версии проекта. Если scipy в образе контейнера отсутствует, автоматически
включается реализация Нелдера — Мида на numpy, поэтому лончер остаётся
работоспособным при минимальном наборе зависимостей.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

try:  # pragma: no cover - зависит от образа
    from scipy.optimize import minimize as _scipy_minimize

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _scipy_minimize = None
    SCIPY_AVAILABLE = False


@dataclass(slots=True)
class OptimizationOutcome:
    x: np.ndarray
    value: float
    evaluations: int
    method: str


def minimize_parameters(
    function: Callable[[np.ndarray], float],
    x0: Sequence[float],
    method: str = "cobyla",
    max_evaluations: int = 200,
) -> OptimizationOutcome:
    counter = {"n": 0}

    def wrapped(vector: np.ndarray) -> float:
        counter["n"] += 1
        return float(function(np.asarray(vector, dtype=float)))

    x0 = np.asarray(x0, dtype=float)

    if method == "cobyla" and SCIPY_AVAILABLE:
        result = _scipy_minimize(
            wrapped,
            x0=x0,
            method="COBYLA",
            options={"maxiter": max_evaluations, "rhobeg": 0.35},
        )
        return OptimizationOutcome(
            x=np.asarray(result.x, dtype=float),
            value=float(result.fun),
            evaluations=counter["n"],
            method="COBYLA (scipy)",
        )

    if method == "nelder-mead" and SCIPY_AVAILABLE:
        result = _scipy_minimize(
            wrapped,
            x0=x0,
            method="Nelder-Mead",
            options={"maxfev": max_evaluations, "xatol": 1e-4, "fatol": 1e-6},
        )
        return OptimizationOutcome(
            x=np.asarray(result.x, dtype=float),
            value=float(result.fun),
            evaluations=counter["n"],
            method="Nelder-Mead (scipy)",
        )

    x, value = _nelder_mead(wrapped, x0, max_evaluations)
    return OptimizationOutcome(
        x=x,
        value=value,
        evaluations=counter["n"],
        method="Nelder-Mead (numpy)",
    )


def _nelder_mead(
    function: Callable[[np.ndarray], float],
    x0: np.ndarray,
    max_evaluations: int,
    step: float = 0.35,
) -> tuple[np.ndarray, float]:
    """Компактная реализация метода Нелдера — Мида без внешних зависимостей."""

    dimension = len(x0)
    simplex = [np.asarray(x0, dtype=float)]
    for index in range(dimension):
        point = np.asarray(x0, dtype=float).copy()
        point[index] += step
        simplex.append(point)

    values = [function(point) for point in simplex]
    evaluations = len(values)

    alpha, gamma_expand, rho, sigma = 1.0, 2.0, 0.5, 0.5

    while evaluations < max_evaluations:
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        best, worst = simplex[0], simplex[-1]
        centroid = np.mean(np.array(simplex[:-1]), axis=0)

        if np.max(np.abs(np.array(simplex[1:]) - best)) < 1e-6:
            break

        reflected = centroid + alpha * (centroid - worst)
        reflected_value = function(reflected)
        evaluations += 1

        if reflected_value < values[0]:
            expanded = centroid + gamma_expand * (reflected - centroid)
            expanded_value = function(expanded)
            evaluations += 1
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
            continue

        if reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
            continue

        contracted = centroid + rho * (worst - centroid)
        contracted_value = function(contracted)
        evaluations += 1

        if contracted_value < values[-1]:
            simplex[-1], values[-1] = contracted, contracted_value
            continue

        for index in range(1, len(simplex)):
            simplex[index] = simplex[0] + sigma * (simplex[index] - simplex[0])
            values[index] = function(simplex[index])
            evaluations += 1

    order = int(np.argmin(values))
    return simplex[order], float(values[order])
