"""
Проверка согласованности кудитной модели с исходной прикладной задачей.

Штрафные слагаемые могут быть заданы неудачно (слишком слабый штраф —
оптимум модели окажется недопустимым; режим unbalanced вообще приближённый).
Поэтому перед запуском квантовой части лончер на небольших экземплярах
перебирает все размещения и проверяет:

    argmin по модели QUDO  ==  argmin по точной целевой функции
                               среди допустимых размещений

Результат проверки выводится в лог и попадает в отчёт: он показывает,
можно ли доверять минимуму модели без дополнительной классической
постобработки.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..formulation.qudo_model import QUDOModel
from .objective import ObjectiveFunction


@dataclass(slots=True)
class ModelValidation:
    checked: bool
    consistent: bool | None = None
    states: int = 0
    model_optimum_placement: list[int] | None = None
    model_optimum_energy: float | None = None
    exact_optimum_placement: list[int] | None = None
    exact_optimum_cost: float | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "consistent": self.consistent,
            "states": self.states,
            "model_optimum_placement": self.model_optimum_placement,
            "model_optimum_energy": self.model_optimum_energy,
            "exact_optimum_placement": self.exact_optimum_placement,
            "exact_optimum_cost": self.exact_optimum_cost,
            "message": self.message,
        }


def validate_model(
    model: QUDOModel,
    objective: ObjectiveFunction,
    max_states: int = 200_000,
) -> ModelValidation:
    problem = model.problem

    domains = [
        problem.allowed_hosts(vm_index)
        for vm_index in range(problem.num_virtual_machines)
    ]

    states = 1
    for domain in domains:
        states *= len(domain)

    if states > max_states:
        return ModelValidation(
            checked=False,
            states=states,
            message=(
                "Пространство размещений слишком велико для полной проверки модели; "
                "допустимость решения проверяется классически при декодировании."
            ),
        )

    best_model_energy = float("inf")
    best_model_placement: list[int] | None = None

    best_exact_cost = float("inf")
    best_exact_placement: list[int] | None = None

    for candidate in product(*domains):
        placement = list(candidate)

        energy = model.energy_from_levels(model.levels_from_placement(placement))
        if energy < best_model_energy:
            best_model_energy = energy
            best_model_placement = placement

        if objective.is_feasible(placement):
            cost = objective.evaluate(placement)
            if cost < best_exact_cost:
                best_exact_cost = cost
                best_exact_placement = placement

    consistent = (
        best_model_placement is not None
        and best_exact_placement is not None
        and objective.is_feasible(best_model_placement)
        and abs(objective.evaluate(best_model_placement) - best_exact_cost) < 1e-6
    )

    message = (
        "Минимум модели QUDO совпадает с точным оптимумом задачи."
        if consistent
        else (
            "Минимум модели QUDO не совпадает с точным оптимумом: "
            "увеличьте вес штрафов (--penalty-weight) или используйте режим --penalty slack."
        )
    )

    return ModelValidation(
        checked=True,
        consistent=bool(consistent),
        states=states,
        model_optimum_placement=best_model_placement,
        model_optimum_energy=best_model_energy,
        exact_optimum_placement=best_exact_placement,
        exact_optimum_cost=best_exact_cost if best_exact_placement else None,
        message=message,
    )
