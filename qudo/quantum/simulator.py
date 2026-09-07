"""
Кудитный эмулятор QAOA на numpy.

Эмулирует эволюцию в пространстве уровней кудитов (размерности d_1 x ... x d_n),
а не в кубитном гильбертовом пространстве. Это позволяет:

    * подбирать параметры (γ, β) без обращения к квантовым сервисам платформы,
      то есть без расхода квоты на запуски;
    * получать в точности тот же результат, что даёт сгенерированная схема
      OpenQASM на идеальном эмуляторе платформы (ideem_ideal_30 / Qiskit Aer),
      поскольку кудитные операции и вентили схемы соответствуют друг другу
      один в один.

Эмулятор используется как самостоятельный вычислитель (backend "local"),
как средство подбора параметров для аппаратных сервисов и как эталон при
проверке корректности сгенерированной схемы.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mixer import mixer_edges


@dataclass(slots=True)
class SimulationLimits:
    max_state_space: int = 1 << 22   # ~4.2e6 амплитуд (64 МБ complex128)


class QuditQAOASimulator:
    """QAOA над набором кудитов произвольных размерностей."""

    def __init__(
        self,
        dims: list[int],
        linear_terms: list[np.ndarray],
        pair_terms: dict[tuple[int, int], np.ndarray],
        constant: float = 0.0,
        scale: float = 1.0,
        limits: SimulationLimits | None = None,
    ) -> None:
        self.dims = list(dims)
        self.constant = float(constant)
        self.scale = float(scale) if scale else 1.0
        self.limits = limits or SimulationLimits()

        size = int(np.prod(self.dims)) if self.dims else 1
        if size > self.limits.max_state_space:
            raise MemoryError(
                "Размер пространства состояний "
                f"{size} превышает лимит локальной эмуляции "
                f"{self.limits.max_state_space}. Используйте выгрузку QUBO "
                "для сервиса отжига (SimBif) или запуск схемы на эмуляторе платформы."
            )

        self.cost = self._build_cost(linear_terms, pair_terms)
        self.flat_cost = self.cost.reshape(-1)
        self.edges = [mixer_edges(dim) for dim in self.dims]

    # ------------------------------------------------------------------

    def _build_cost(
        self,
        linear_terms: list[np.ndarray],
        pair_terms: dict[tuple[int, int], np.ndarray],
    ) -> np.ndarray:
        cost = np.zeros(self.dims, dtype=float)

        for index, vector in enumerate(linear_terms):
            shape = [1] * len(self.dims)
            shape[index] = self.dims[index]
            cost = cost + np.asarray(vector, dtype=float).reshape(shape)

        for (left, right), matrix in pair_terms.items():
            shape = [1] * len(self.dims)
            shape[left] = self.dims[left]
            shape[right] = self.dims[right]
            cost = cost + np.asarray(matrix, dtype=float).reshape(shape)

        return cost

    # ------------------------------------------------------------------

    def initial_state(self) -> np.ndarray:
        size = int(np.prod(self.dims))
        amplitude = 1.0 / np.sqrt(size)
        return np.full(self.dims, amplitude, dtype=complex)

    def _apply_cost(self, state: np.ndarray, gamma: float) -> np.ndarray:
        return state * np.exp(-1j * gamma * (self.cost / self.scale))

    def _apply_mixer(self, state: np.ndarray, beta: float) -> np.ndarray:
        cos = np.cos(beta)
        sin = np.sin(beta)

        for axis, edges in enumerate(self.edges):
            for left, right in edges:
                index_left = self._slice(axis, left)
                index_right = self._slice(axis, right)

                amplitude_left = state[index_left].copy()
                amplitude_right = state[index_right].copy()

                state[index_left] = cos * amplitude_left - 1j * sin * amplitude_right
                state[index_right] = -1j * sin * amplitude_left + cos * amplitude_right

        return state

    def _slice(self, axis: int, level: int) -> tuple:
        selector: list = [slice(None)] * len(self.dims)
        selector[axis] = level
        return tuple(selector)

    # ------------------------------------------------------------------

    def run(self, gammas, betas) -> np.ndarray:
        if len(gammas) != len(betas):
            raise ValueError("Число параметров γ и β должно совпадать.")

        state = self.initial_state()
        for gamma, beta in zip(gammas, betas):
            state = self._apply_cost(state, float(gamma))
            state = self._apply_mixer(state, float(beta))
        return state

    def probabilities(self, gammas, betas) -> np.ndarray:
        state = self.run(gammas, betas)
        probabilities = np.abs(state.reshape(-1)) ** 2
        total = probabilities.sum()
        return probabilities / total if total else probabilities

    def expectation(self, gammas, betas) -> float:
        probabilities = self.probabilities(gammas, betas)
        return float(probabilities @ self.flat_cost + self.constant)

    def sample(self, gammas, betas, shots: int, rng: np.random.Generator) -> dict[int, int]:
        """Возвращает счётчики по плоским индексам состояний."""

        probabilities = self.probabilities(gammas, betas)
        draws = rng.multinomial(shots, probabilities)
        return {int(index): int(count) for index, count in enumerate(draws) if count}

    def levels_from_flat_index(self, index: int) -> list[int]:
        return [int(value) for value in np.unravel_index(index, self.dims)]

    def cvar(self, gammas, betas, alpha: float = 0.2) -> float:
        """CVaR-оценка (нижний хвост распределения энергий)."""

        probabilities = self.probabilities(gammas, betas)
        order = np.argsort(self.flat_cost)
        sorted_probabilities = probabilities[order]
        sorted_costs = self.flat_cost[order]

        cumulative = np.cumsum(sorted_probabilities)
        cutoff = np.searchsorted(cumulative, alpha) + 1
        weights = sorted_probabilities[:cutoff].copy()
        total = weights.sum()
        if total <= 0:
            return float(sorted_costs[0] + self.constant)
        weights /= total
        return float(weights @ sorted_costs[:cutoff] + self.constant)
