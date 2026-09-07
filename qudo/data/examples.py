"""
Тестовые экземпляры задачи распределения ресурсов.

Экземпляры доступны лончеру по имени (параметр --template) и могут быть
выгружены в JSON (--save-problem) для загрузки в ОПКВ как входной файл.
"""

from __future__ import annotations

import numpy as np

from ..model.infrastructure import Host, VirtualMachine
from ..model.instance import AllocationProblem


def create_small_problem() -> AllocationProblem:
    """4 хоста, 4 ВМ. Ресурсных ограничений хватает: работает сетевая часть."""

    hosts = [
        Host(id=i, cpu_capacity=8, ram_capacity=16, bandwidth_capacity=10, name=f"host{i}")
        for i in range(4)
    ]

    virtual_machines = [
        VirtualMachine(id=0, cpu_requirement=2, ram_requirement=4, bandwidth_requirement=2),
        VirtualMachine(id=1, cpu_requirement=3, ram_requirement=4, bandwidth_requirement=1),
        VirtualMachine(id=2, cpu_requirement=1, ram_requirement=2, bandwidth_requirement=2),
        VirtualMachine(id=3, cpu_requirement=2, ram_requirement=3, bandwidth_requirement=1),
    ]

    latency_matrix = np.array(
        [
            [2, 2, 4, 5],
            [2, 3, 3, 4],
            [4, 3, 1, 2],
            [5, 4, 2, 2],
        ],
        dtype=float,
    )

    traffic_matrix = np.array(
        [
            [0, 8, 2, 1],
            [8, 0, 4, 3],
            [2, 4, 0, 5],
            [1, 3, 5, 0],
        ],
        dtype=float,
    )

    problem = AllocationProblem(
        hosts=hosts,
        virtual_machines=virtual_machines,
        latency_matrix=latency_matrix,
        traffic_matrix=traffic_matrix,
        name="dc-small-4x4",
    )
    problem.validate()
    return problem


def create_resource_constrained_problem() -> AllocationProblem:
    """
    2 хоста, 3 ВМ с дефицитом CPU/RAM/пропускной способности.

    Размещение всех ВМ на одном хосте нарушает сразу три ограничения,
    поэтому экземпляр проверяет корректность штрафных слагаемых.
    """

    hosts = [
        Host(id=0, cpu_capacity=6, ram_capacity=10, bandwidth_capacity=6, name="host0"),
        Host(id=1, cpu_capacity=6, ram_capacity=10, bandwidth_capacity=6, name="host1"),
    ]

    virtual_machines = [
        VirtualMachine(id=0, cpu_requirement=4, ram_requirement=5, bandwidth_requirement=3),
        VirtualMachine(id=1, cpu_requirement=3, ram_requirement=4, bandwidth_requirement=2),
        VirtualMachine(id=2, cpu_requirement=2, ram_requirement=3, bandwidth_requirement=2),
    ]

    latency_matrix = np.array([[1, 5], [5, 1]], dtype=float)

    traffic_matrix = np.array(
        [
            [0, 8, 6],
            [8, 0, 7],
            [6, 7, 0],
        ],
        dtype=float,
    )

    problem = AllocationProblem(
        hosts=hosts,
        virtual_machines=virtual_machines,
        latency_matrix=latency_matrix,
        traffic_matrix=traffic_matrix,
        name="dc-constrained-2x3",
    )
    problem.validate()
    return problem


def create_medium_problem(seed: int = 42) -> AllocationProblem:
    """
    6 хостов, 6 ВМ; часть ВМ закреплена за подмножеством хостов (зоны доступности).

    Разные размерности кудитов (d_i) проверяют общий случай постановки.
    """

    rng = np.random.default_rng(seed)

    hosts = [
        Host(id=i, cpu_capacity=12, ram_capacity=24, bandwidth_capacity=16, name=f"host{i}")
        for i in range(6)
    ]

    zones = [None, (0, 1, 2), None, (3, 4, 5), (0, 1, 2, 3), None]

    virtual_machines = [
        VirtualMachine(
            id=i,
            cpu_requirement=int(rng.integers(2, 6)),
            ram_requirement=int(rng.integers(3, 9)),
            bandwidth_requirement=float(rng.integers(2, 6)),
            allowed_hosts=zones[i],
        )
        for i in range(6)
    ]

    latency = rng.integers(1, 6, size=(6, 6)).astype(float)
    latency = (latency + latency.T) / 2.0
    np.fill_diagonal(latency, 1.0)

    traffic = rng.integers(0, 9, size=(6, 6)).astype(float)
    traffic = (traffic + traffic.T) / 2.0
    np.fill_diagonal(traffic, 0.0)

    problem = AllocationProblem(
        hosts=hosts,
        virtual_machines=virtual_machines,
        latency_matrix=latency,
        traffic_matrix=traffic,
        name="dc-medium-6x6",
    )
    problem.validate()
    return problem


TEMPLATES = {
    "small": create_small_problem,
    "constrained": create_resource_constrained_problem,
    "medium": create_medium_problem,
}


def load_template(name: str) -> AllocationProblem:
    factory = TEMPLATES.get(name)
    if factory is None:
        raise ValueError(
            f"Неизвестный шаблон '{name}'. Доступны: {', '.join(sorted(TEMPLATES))}"
        )
    return factory()
