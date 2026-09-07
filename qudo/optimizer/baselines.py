"""
Классические эталоны для сравнения с кудитным QAOA.

    * GreedySearch     — жадное размещение (промышленный эталон «здравого смысла»);
    * ExhaustiveSearch — полный перебор, даёт глобальный оптимум и позволяет
                         вычислить optimality gap на малых экземплярах.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..model.infrastructure import RESOURCES
from .objective import ObjectiveFunction


@dataclass(slots=True)
class SolutionRecord:
    placement: list[int]
    cost: float
    feasible: bool
    explored_states: int = 0


class GreedySearch:
    """Последовательно выбирает хост с минимальным приростом сетевой стоимости."""

    def __init__(self, objective: ObjectiveFunction) -> None:
        self.objective = objective
        self.problem = objective.problem

    def solve(self) -> SolutionRecord:
        problem = self.problem
        placement = [-1] * problem.num_virtual_machines

        order = sorted(
            range(problem.num_virtual_machines),
            key=self._weight,
            reverse=True,
        )

        for vm_index in order:
            best_host = None
            best_delta = float("inf")

            for host in problem.allowed_hosts(vm_index):
                if not self._fits(vm_index, host, placement):
                    continue

                delta = self._network_delta(vm_index, host, placement)
                if delta < best_delta:
                    best_delta = delta
                    best_host = host

            if best_host is None:
                return SolutionRecord(placement, float("inf"), False)

            placement[vm_index] = best_host

        return SolutionRecord(
            placement=placement,
            cost=self.objective.evaluate(placement),
            feasible=self.objective.is_feasible(placement),
        )

    def _weight(self, vm_index: int) -> float:
        vm = self.problem.virtual_machines[vm_index]
        return sum(vm.requirement(resource) for resource in RESOURCES)

    def _fits(self, vm_index: int, host_index: int, placement: list[int]) -> bool:
        host = self.problem.hosts[host_index]

        for resource in RESOURCES:
            load = self.problem.virtual_machines[vm_index].requirement(resource)
            for other, assigned in enumerate(placement):
                if assigned == host_index:
                    load += self.problem.virtual_machines[other].requirement(resource)
            if load > host.capacity(resource) + 1e-9:
                return False

        return True

    def _network_delta(self, vm_index: int, host_index: int, placement: list[int]) -> float:
        traffic = self.problem.traffic_matrix
        latency = self.problem.latency_matrix

        total = 0.0
        for other, assigned in enumerate(placement):
            if assigned == -1:
                continue
            total += float(traffic[vm_index, other]) * float(latency[host_index, assigned])
        return total


class ExhaustiveSearch:
    """Полный перебор допустимых размещений (только для малых экземпляров)."""

    def __init__(self, objective: ObjectiveFunction, max_states: int = 2_000_000) -> None:
        self.objective = objective
        self.problem = objective.problem
        self.max_states = max_states

    def state_space(self) -> int:
        size = 1
        for vm_index in range(self.problem.num_virtual_machines):
            size *= len(self.problem.allowed_hosts(vm_index))
        return size

    def solve(self) -> SolutionRecord | None:
        if self.state_space() > self.max_states:
            return None

        best: list[int] | None = None
        best_cost = float("inf")
        explored = 0

        domains = [
            self.problem.allowed_hosts(vm_index)
            for vm_index in range(self.problem.num_virtual_machines)
        ]

        for candidate in product(*domains):
            explored += 1
            placement = list(candidate)

            if not self.objective.is_feasible(placement):
                continue

            cost = self.objective.evaluate(placement)
            if cost < best_cost:
                best_cost = cost
                best = placement

        if best is None:
            return SolutionRecord([], float("inf"), False, explored)

        return SolutionRecord(best, best_cost, True, explored)
