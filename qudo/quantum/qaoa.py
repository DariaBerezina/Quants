"""
Кудитный вариант алгоритма квантовой приближённой оптимизации (QAOA).

Схема алгоритма:

    |ψ(γ, β)> = prod_{l=1..p} [ U_M(β_l) U_C(γ_l) ] |+>^{⊗ n}

    U_C(γ) = exp(-i γ C / s)   — диагональный фазовый оператор (s — масштаб
                                 коэффициентов, приводящий углы к [0, 2π]);
    U_M(β) = prod_i exp(-i β M_i), M_i = X_{d_i} + X_{d_i}^† — кудитный
                                 смеситель (оператор сдвига уровней);
    |+>     — равномерная суперпозиция по всем уровням кудитов.

Параметры (γ_l, β_l) подбираются классическим оптимизатором по среднему
значению энергии; поддерживается также критерий CVaR, который на задачах
с сильными штрафами обычно даёт более качественные выборки.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import numpy as np

from ..formulation.qudo_model import QUDOModel
from ..optimizer.parameters import minimize_parameters
from .backends import BaseBackend


@dataclass(slots=True)
class QAOAResult:
    gammas: list[float]
    betas: list[float]
    expectation: float
    evaluations: int
    method: str
    restarts: int
    counts: dict[str, int] = field(default_factory=dict)


class QuditQAOA:
    def __init__(
        self,
        model: QUDOModel,
        backend: BaseBackend,
        layers: int = 2,
        optimizer: str = "cobyla",
        restarts: int = 3,
        max_evaluations: int = 200,
        objective: str = "expectation",
        cvar_alpha: float = 0.2,
        seed: int | None = None,
    ) -> None:
        if layers < 1:
            raise ValueError("QAOA должен содержать хотя бы один слой.")

        self.model = model
        self.backend = backend
        self.layers = int(layers)
        self.optimizer = optimizer
        self.restarts = max(1, int(restarts))
        self.max_evaluations = int(max_evaluations)
        self.objective = objective
        self.cvar_alpha = float(cvar_alpha)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------

    def _split(self, vector: np.ndarray) -> tuple[list[float], list[float]]:
        gammas = [float(v) for v in vector[: self.layers]]
        betas = [float(v) for v in vector[self.layers :]]
        return gammas, betas

    def _cost(self, vector: np.ndarray) -> float:
        gammas, betas = self._split(vector)

        if self.objective == "cvar" and hasattr(self.backend, "simulator"):
            return self.backend.simulator.cvar(gammas, betas, self.cvar_alpha)

        return self.backend.expectation(gammas, betas)

    def initial_parameters(self, attempt: int) -> np.ndarray:
        """Линейный «отжиговый» профиль параметров + случайное возмущение."""

        steps = np.arange(1, self.layers + 1) / self.layers
        gammas = 0.9 * steps
        betas = 0.9 * (pi / 4.0) * (1.0 - steps + 1.0 / self.layers)

        vector = np.concatenate([gammas, betas])

        if attempt > 0:
            vector = vector + self.rng.normal(scale=0.35, size=vector.shape)

        return vector

    # ------------------------------------------------------------------

    def optimize(self, logger=None) -> QAOAResult:
        best_vector: np.ndarray | None = None
        best_value = float("inf")
        best_method = ""
        evaluations = 0

        for attempt in range(self.restarts):
            outcome = minimize_parameters(
                self._cost,
                self.initial_parameters(attempt),
                method=self.optimizer,
                max_evaluations=self.max_evaluations,
            )
            evaluations += outcome.evaluations

            if logger is not None:
                logger.log(
                    f"Старт {attempt + 1}/{self.restarts}: критерий = {outcome.value:.4f}, "
                    f"вычислений целевой функции = {outcome.evaluations}",
                    level=2,
                )

            if outcome.value < best_value:
                best_value = outcome.value
                best_vector = outcome.x
                best_method = outcome.method

        assert best_vector is not None
        gammas, betas = self._split(best_vector)

        return QAOAResult(
            gammas=gammas,
            betas=betas,
            expectation=float(self.backend.expectation(gammas, betas)),
            evaluations=evaluations,
            method=best_method,
            restarts=self.restarts,
        )

    def sample(self, result: QAOAResult, shots: int) -> dict[str, int]:
        counts = self.backend.sample(result.gammas, result.betas, shots)
        result.counts = counts
        return counts
