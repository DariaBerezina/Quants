"""
Классическая целевая функция задачи размещения ВМ.

Используется как «арбитр»: именно она определяет итоговую стоимость решения
и его допустимость, независимо от того, каким способом решение получено —
кудитным QAOA, отжигом, жадным алгоритмом или полным перебором. Такой
порядок важен потому, что штрафные слагаемые квантовой модели могут быть
приближёнными (режим unbalanced), а проверка аппаратных ограничений должна
оставаться строгой.

    C(x) = sum_{i<j} T_ij * L[x_i, x_j]
         + sum_{h} [ w_cpu * over(cpu)^2 + w_ram * over(ram)^2 + w_bw * over(bw)^2 ]

    over(r) = max(0, load_{h,r} - cap_{h,r})
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.infrastructure import RESOURCES
from ..model.instance import AllocationProblem


@dataclass(slots=True)
class PenaltyWeights:
    cpu: float = 1000.0
    ram: float = 1000.0
    bandwidth: float = 1000.0

    def weight(self, resource: str) -> float:
        return float(getattr(self, resource))


class ObjectiveFunction:
    def __init__(
        self,
        problem: AllocationProblem,
        penalties: PenaltyWeights | None = None,
    ) -> None:
        self.problem = problem
        self.penalties = penalties or PenaltyWeights()

    # ------------------------------------------------------------------

    def evaluate(self, placement) -> float:
        self._check(placement)
        return self.network_cost(placement) + self.penalty(placement)

    def network_cost(self, placement) -> float:
        traffic = np.asarray(self.problem.traffic_matrix, dtype=float)
        latency = np.asarray(self.problem.latency_matrix, dtype=float)

        total = 0.0
        count = self.problem.num_virtual_machines

        for i in range(count):
            host_i = int(placement[i])
            for j in range(i + 1, count):
                total += float(traffic[i, j]) * float(latency[host_i, int(placement[j])])

        return total

    def loads(self, placement) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for resource in RESOURCES:
            usage = np.zeros(self.problem.num_hosts, dtype=float)
            for vm_index, host in enumerate(placement):
                usage[int(host)] += self.problem.virtual_machines[vm_index].requirement(resource)
            result[resource] = usage
        return result

    def overloads(self, placement) -> dict[str, np.ndarray]:
        loads = self.loads(placement)
        result: dict[str, np.ndarray] = {}
        for resource in RESOURCES:
            capacities = np.array(
                [host.capacity(resource) for host in self.problem.hosts], dtype=float
            )
            result[resource] = np.maximum(0.0, loads[resource] - capacities)
        return result

    def penalty(self, placement) -> float:
        overloads = self.overloads(placement)
        total = 0.0
        for resource in RESOURCES:
            total += self.penalties.weight(resource) * float(np.sum(overloads[resource] ** 2))
        return total

    def is_feasible(self, placement) -> bool:
        if placement is None or len(placement) != self.problem.num_virtual_machines:
            return False

        for vm_index, host in enumerate(placement):
            if int(host) not in self.problem.allowed_hosts(vm_index):
                return False

        overloads = self.overloads(placement)
        return all(float(np.max(overloads[resource], initial=0.0)) <= 1e-9 for resource in RESOURCES)

    def report(self, placement) -> dict:
        loads = self.loads(placement)
        return {
            "placement": [int(host) for host in placement],
            "network_cost": self.network_cost(placement),
            "penalty": self.penalty(placement),
            "cost": self.evaluate(placement),
            "feasible": self.is_feasible(placement),
            "load": {
                resource: [float(value) for value in loads[resource]] for resource in RESOURCES
            },
            "capacity": {
                resource: [float(host.capacity(resource)) for host in self.problem.hosts]
                for resource in RESOURCES
            },
        }

    # ------------------------------------------------------------------

    def _check(self, placement) -> None:
        if len(placement) != self.problem.num_virtual_machines:
            raise ValueError("Некорректная длина вектора размещения.")
