"""
Экземпляр прикладной задачи размещения виртуальных машин.

Входные данные (соответствуют постановке 2.2.4.1):

    * описание инфраструктуры ЦОД — множество хостов, их аппаратные ёмкости
      (CPU, RAM, пропускная способность) и матрица сетевых задержек L;
    * описание нагрузки — множество ВМ, их требования к ресурсам и матрица
      интенсивности сетевого взаимодействия T.

Для загрузки в ОПКВ экземпляр сериализуется в JSON: лончер получает его
через параметр ``--input /data/<файл>.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .infrastructure import RESOURCES, Host, VirtualMachine


@dataclass(slots=True)
class AllocationProblem:
    """Полное описание задачи распределения ресурсов."""

    hosts: List[Host]
    virtual_machines: List[VirtualMachine]

    latency_matrix: np.ndarray
    traffic_matrix: np.ndarray

    name: str = "qudo-instance"

    # ------------------------------------------------------------------
    # Базовые свойства
    # ------------------------------------------------------------------

    @property
    def num_hosts(self) -> int:
        return len(self.hosts)

    @property
    def num_virtual_machines(self) -> int:
        return len(self.virtual_machines)

    def allowed_hosts(self, vm_index: int) -> tuple[int, ...]:
        """Список допустимых хостов для ВМ (в порядке возрастания индекса)."""

        allowed = self.virtual_machines[vm_index].allowed_hosts
        if allowed is None:
            return tuple(range(self.num_hosts))

        ordered = tuple(sorted(set(int(h) for h in allowed)))
        if not ordered:
            raise ValueError(f"У ВМ {vm_index} пустое множество допустимых хостов.")
        if ordered[0] < 0 or ordered[-1] >= self.num_hosts:
            raise ValueError(f"Недопустимый индекс хоста в allowed_hosts ВМ {vm_index}.")
        return ordered

    def qudit_dimensions(self) -> list[int]:
        """Размерности кудитов: d_i = |допустимые хосты ВМ i|."""

        return [len(self.allowed_hosts(i)) for i in range(self.num_virtual_machines)]

    # ------------------------------------------------------------------
    # Валидация и агрегаты
    # ------------------------------------------------------------------

    def validate(self) -> None:
        n_hosts = self.num_hosts
        n_vm = self.num_virtual_machines

        if n_hosts < 1:
            raise ValueError("Требуется хотя бы один физический хост.")
        if n_vm < 1:
            raise ValueError("Требуется хотя бы одна виртуальная машина.")

        self.latency_matrix = np.asarray(self.latency_matrix, dtype=float)
        self.traffic_matrix = np.asarray(self.traffic_matrix, dtype=float)

        if self.latency_matrix.shape != (n_hosts, n_hosts):
            raise ValueError(f"Матрица задержек должна иметь размер ({n_hosts}, {n_hosts}).")

        if self.traffic_matrix.shape != (n_vm, n_vm):
            raise ValueError(f"Матрица трафика должна иметь размер ({n_vm}, {n_vm}).")

        if not np.allclose(self.traffic_matrix, self.traffic_matrix.T):
            raise ValueError("Матрица интенсивности трафика должна быть симметричной.")

        for i in range(n_vm):
            self.allowed_hosts(i)

    def total_capacity(self, resource: str) -> float:
        return sum(host.capacity(resource) for host in self.hosts)

    def total_requirement(self, resource: str) -> float:
        return sum(vm.requirement(resource) for vm in self.virtual_machines)

    def is_potentially_feasible(self) -> bool:
        """Необходимое условие существования допустимого размещения."""

        return all(
            self.total_capacity(resource) >= self.total_requirement(resource)
            for resource in RESOURCES
        )

    def summary(self) -> dict:
        return {
            "name": self.name,
            "hosts": self.num_hosts,
            "virtual_machines": self.num_virtual_machines,
            "qudit_dimensions": self.qudit_dimensions(),
            "capacity": {r: self.total_capacity(r) for r in RESOURCES},
            "requirement": {r: self.total_requirement(r) for r in RESOURCES},
            "potentially_feasible": self.is_potentially_feasible(),
        }

    # ------------------------------------------------------------------
    # Сериализация (формат входного файла ОПКВ)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hosts": [host.to_dict() for host in self.hosts],
            "virtual_machines": [vm.to_dict() for vm in self.virtual_machines],
            "latency_matrix": np.asarray(self.latency_matrix, dtype=float).tolist(),
            "traffic_matrix": np.asarray(self.traffic_matrix, dtype=float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AllocationProblem":
        problem = cls(
            hosts=[Host.from_dict(item) for item in payload["hosts"]],
            virtual_machines=[
                VirtualMachine.from_dict(item) for item in payload["virtual_machines"]
            ],
            latency_matrix=np.asarray(payload["latency_matrix"], dtype=float),
            traffic_matrix=np.asarray(payload["traffic_matrix"], dtype=float),
            name=str(payload.get("name", "qudo-instance")),
        )
        problem.validate()
        return problem
