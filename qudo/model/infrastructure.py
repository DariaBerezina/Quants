"""
Описание инфраструктуры ЦОД: физические хосты и виртуальные машины.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# Учитываемые аппаратные ресурсы. Порядок фиксирован: используется
# при построении штрафных слагаемых и при выводе отчётов.
RESOURCES: tuple[str, ...] = ("cpu", "ram", "bandwidth")


@dataclass(slots=True)
class Host:
    """Физический сервер (хост) центра обработки данных."""

    id: int

    cpu_capacity: float
    ram_capacity: float
    bandwidth_capacity: float

    name: str = ""

    def capacity(self, resource: str) -> float:
        return float(getattr(self, f"{resource}_capacity"))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "cpu_capacity": self.cpu_capacity,
            "ram_capacity": self.ram_capacity,
            "bandwidth_capacity": self.bandwidth_capacity,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Host":
        return cls(
            id=int(payload["id"]),
            cpu_capacity=float(payload["cpu_capacity"]),
            ram_capacity=float(payload["ram_capacity"]),
            bandwidth_capacity=float(payload["bandwidth_capacity"]),
            name=str(payload.get("name", "")),
        )

    def __str__(self) -> str:
        return (
            f"Host(id={self.id}, CPU={self.cpu_capacity}, "
            f"RAM={self.ram_capacity}, BW={self.bandwidth_capacity})"
        )


@dataclass(slots=True)
class VirtualMachine:
    """
    Виртуальная машина (размещаемая нагрузка).

    ``allowed_hosts`` — множество допустимых хостов для данной ВМ.
    Именно его мощность задаёт размерность кудита, кодирующего ВМ
    (см. постановку задачи 2.2.4.1: «каждая виртуальная машина кодируется
    одним кудитом размерности, равной числу допустимых хостов»).
    ``None`` означает, что допустимы все хосты.
    """

    id: int

    cpu_requirement: float
    ram_requirement: float
    bandwidth_requirement: float

    allowed_hosts: tuple[int, ...] | None = None

    name: str = ""

    def requirement(self, resource: str) -> float:
        return float(getattr(self, f"{resource}_requirement"))

    def to_dict(self) -> dict:
        payload = {
            "id": self.id,
            "name": self.name,
            "cpu_requirement": self.cpu_requirement,
            "ram_requirement": self.ram_requirement,
            "bandwidth_requirement": self.bandwidth_requirement,
        }
        if self.allowed_hosts is not None:
            payload["allowed_hosts"] = list(self.allowed_hosts)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "VirtualMachine":
        allowed: Sequence[int] | None = payload.get("allowed_hosts")
        return cls(
            id=int(payload["id"]),
            cpu_requirement=float(payload["cpu_requirement"]),
            ram_requirement=float(payload["ram_requirement"]),
            bandwidth_requirement=float(payload["bandwidth_requirement"]),
            allowed_hosts=tuple(int(h) for h in allowed) if allowed is not None else None,
            name=str(payload.get("name", "")),
        )

    def __str__(self) -> str:
        return (
            f"VM(id={self.id}, CPU={self.cpu_requirement}, "
            f"RAM={self.ram_requirement}, BW={self.bandwidth_requirement})"
        )
