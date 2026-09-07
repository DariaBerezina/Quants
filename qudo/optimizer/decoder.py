"""
Декодирование результатов измерений в размещение виртуальных машин.

Работает одинаково и с распределением, полученным локальной эмуляцией,
и со словарём счётчиков, который возвращает платформа после запуска схемы
на выбранном сервисе (лог вида ``{"0100": 125, "1010": 109, ...}``).

Битовая строка разбивается на блоки кудитов; номер уровня — позиция единицы
в блоке; хост определяется по списку допустимых хостов ВМ:

        01 00 10 ...  ->  уровни (0, ...)  ->  хосты (allowed_hosts[i][level])

Строки, не принадлежащие подпространству one-hot (возможны только при
исполнении на шумном вычислителе), помечаются как некорректные и в выбор
решения не попадают; их доля выводится как показатель качества запуска.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..formulation.qudo_model import QUDOModel
from .objective import ObjectiveFunction


@dataclass(slots=True)
class DecodedSample:
    bitstring: str
    counts: int
    placement: list[int]
    cost: float
    feasible: bool
    model_energy: float


@dataclass(slots=True)
class DecodeReport:
    total_shots: int = 0
    valid_shots: int = 0
    feasible_shots: int = 0
    unique_states: int = 0
    samples: list[DecodedSample] = field(default_factory=list)

    best_feasible: DecodedSample | None = None
    best_overall: DecodedSample | None = None

    @property
    def valid_fraction(self) -> float:
        return self.valid_shots / self.total_shots if self.total_shots else 0.0

    @property
    def feasible_fraction(self) -> float:
        return self.feasible_shots / self.total_shots if self.total_shots else 0.0

    def probability_of(self, placement) -> float:
        target = [int(host) for host in placement]
        shots = sum(
            sample.counts for sample in self.samples if sample.placement == target
        )
        return shots / self.total_shots if self.total_shots else 0.0


class Decoder:
    def __init__(
        self,
        model: QUDOModel,
        objective: ObjectiveFunction,
        bit_order: str = "little",
    ) -> None:
        self.model = model
        self.objective = objective
        self.bit_order = bit_order

    def decode(self, counts: Mapping[str, int]) -> DecodeReport:
        report = DecodeReport()

        for bitstring, shots in counts.items():
            shots = int(shots)
            report.total_shots += shots
            report.unique_states += 1

            try:
                bits = self.model.layout.bits_from_bitstring(bitstring, self.bit_order)
            except ValueError:
                continue

            levels, valid = self.model.layout.levels_from_bits(bits)
            if not valid or levels is None:
                continue

            report.valid_shots += shots

            placement = self.model.placement_from_levels(levels)
            cost = self.objective.evaluate(placement)
            feasible = self.objective.is_feasible(placement)

            if feasible:
                report.feasible_shots += shots

            sample = DecodedSample(
                bitstring=bitstring,
                counts=shots,
                placement=placement,
                cost=cost,
                feasible=feasible,
                model_energy=self.model.energy_from_bits(bits),
            )
            report.samples.append(sample)

            if report.best_overall is None or cost < report.best_overall.cost:
                report.best_overall = sample

            if feasible and (
                report.best_feasible is None or cost < report.best_feasible.cost
            ):
                report.best_feasible = sample

        report.samples.sort(key=lambda item: (-item.counts, item.cost))
        return report
