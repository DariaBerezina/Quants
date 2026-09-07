"""
Кудитная формулировка задачи размещения ВМ (модель QUDO).

Модель:

    min  C(x) = sum_{i<j} T_ij * L[x_i, x_j] + sum_{h,r} P_{h,r}(x)

    x_i  — кудитная переменная: номер хоста для ВМ i, x_i ∈ {0, ..., d_i-1};
    T    — матрица интенсивности сетевого взаимодействия между ВМ;
    L    — матрица сетевых задержек между хостами;
    P    — штрафные слагаемые за нарушение аппаратных ограничений
           по CPU, RAM и пропускной способности.

Через индикаторы y_{i,a} = [x_i = a] модель записывается как квадратичная
безусловная задача (QUBO/QUDO):

    C(y) = sum_{i<j} sum_{a,b} T_ij L_ab y_{i,a} y_{j,b}
         + sum_{h,r} P_{h,r}(y)

Ограничение sum_a y_{i,a} = 1 выполняется автоматически, поскольку кудит
всегда находится ровно в одном уровне; при выгрузке QUBO-матрицы для
эмулятора отжига оно добавляется отдельным штрафом (см. qubo.py).

Способы учёта ресурсных ограничений (load_{h,r} <= cap_{h,r}):

    * "slack"      — точное представление неравенства через дополнительные
                     бинарные переменные: P*(load + s - cap)^2. Даёт
                     математически эквивалентную QUBO-модель, но добавляет
                     O(log cap) переменных на каждое связывающее ограничение.
    * "unbalanced" — представление без дополнительных переменных:
                     λ1*g + λ2*g^2, g = load - cap («unbalanced penalization»).
                     Число кубитов минимально, но модель приближённая:
                     допустимость итогового решения проверяется классически
                     на этапе декодирования результатов измерений.
    * "auto"       — "slack", если связывающих ограничений нет или их
                     представление умещается в лимит кубитов; иначе
                     "unbalanced" (с предупреждением в лог).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable

import numpy as np

from ..model.infrastructure import RESOURCES
from ..model.instance import AllocationProblem
from ..model.qudit import BIT, QUDIT, Layout, build_layout

PENALTY_MODES = ("auto", "slack", "unbalanced")


@dataclass(slots=True)
class PenaltyConfig:
    """Настройки штрафных слагаемых."""

    mode: str = "auto"
    weight: float | None = None          # P для режима slack
    lambda1: float | None = None         # λ1 для режима unbalanced
    lambda2: float | None = None         # λ2 для режима unbalanced
    max_slack_vars: int = 24             # лимит числа slack-переменных для режима auto
    max_state_space: int = 1 << 20       # лимит пространства состояний для режима auto


@dataclass(slots=True)
class BindingConstraint:
    """Связывающее ограничение (host, resource), которое может нарушаться."""

    host: int
    resource: str
    capacity: float
    participants: tuple[tuple[int, float], ...]   # (индекс ВМ, требование)
    slack_coefficients: tuple[float, ...] = ()


class QUDOModel:
    """Квадратичная безусловная модель задачи в переменных y_{i,a} (+ slack)."""

    def __init__(
        self,
        problem: AllocationProblem,
        penalty: PenaltyConfig | None = None,
    ) -> None:
        self.problem = problem
        self.penalty = penalty or PenaltyConfig()

        self.network_scale = self._network_scale()
        self.constraints: list[BindingConstraint] = self._binding_constraints()
        self.mode = self._resolve_mode()

        if self.mode == "slack":
            slack_specs = [
                (f"slack:h{c.host}:{c.resource}:b{b}", coefficient)
                for c in self.constraints
                for b, coefficient in enumerate(c.slack_coefficients)
            ]
        else:
            slack_specs = []

        self.layout: Layout = build_layout(problem, slack_specs)

        self.linear: dict[int, float] = {}
        self.quadratic: dict[tuple[int, int], float] = {}
        self.constant: float = 0.0

        self._build_network_terms()
        self._build_resource_terms()

        self.scale = self._coefficient_scale()

    # ------------------------------------------------------------------
    # Подготовка
    # ------------------------------------------------------------------

    def _network_scale(self) -> float:
        """Верхняя оценка размаха сетевой части целевой функции."""

        traffic = np.asarray(self.problem.traffic_matrix, dtype=float)
        latency = np.asarray(self.problem.latency_matrix, dtype=float)

        spread = float(np.max(latency) - np.min(latency))
        total_traffic = float(np.sum(np.triu(traffic, k=1)))

        return max(1.0, total_traffic * max(spread, 1.0) + 1.0)

    def _binding_constraints(self) -> list[BindingConstraint]:
        """
        Отбирает ограничения, которые в принципе могут быть нарушены.

        Если суммарное требование ВМ, допущенных на хост, не превышает его
        ёмкость, ограничение избыточно и в модель не включается — это
        сокращает и число слагаемых, и число slack-переменных.
        """

        problem = self.problem
        constraints: list[BindingConstraint] = []

        for host_index, host in enumerate(problem.hosts):
            for resource in RESOURCES:
                capacity = host.capacity(resource)

                participants = tuple(
                    (vm_index, problem.virtual_machines[vm_index].requirement(resource))
                    for vm_index in range(problem.num_virtual_machines)
                    if host_index in problem.allowed_hosts(vm_index)
                    and problem.virtual_machines[vm_index].requirement(resource) > 0.0
                )

                max_load = sum(requirement for _, requirement in participants)
                if max_load <= capacity + 1e-12:
                    continue

                forced_load = sum(
                    requirement
                    for vm_index, requirement in participants
                    if problem.allowed_hosts(vm_index) == (host_index,)
                )
                slack_range = max(0.0, capacity - forced_load)

                constraints.append(
                    BindingConstraint(
                        host=host_index,
                        resource=resource,
                        capacity=capacity,
                        participants=participants,
                        slack_coefficients=_slack_coefficients(slack_range),
                    )
                )

        return constraints

    def _resolve_mode(self) -> str:
        mode = self.penalty.mode
        if mode not in PENALTY_MODES:
            raise ValueError(f"Неизвестный режим штрафов: {mode}")

        if mode != "auto":
            return mode

        slack_vars = sum(len(c.slack_coefficients) for c in self.constraints)
        if slack_vars == 0:
            return "slack"

        # Точное представление неравенств увеличивает пространство состояний
        # в 2^slack_vars раз. Режим auto выбирает его только тогда, когда
        # задача остаётся в пределах локальной эмуляции и разумного числа
        # кубитов для сервисов платформы.
        assignment_space = 1
        for dimension in self.problem.qudit_dimensions():
            assignment_space *= dimension

        if (
            slack_vars <= self.penalty.max_slack_vars
            and assignment_space * (1 << slack_vars) <= self.penalty.max_state_space
        ):
            return "slack"

        return "unbalanced"

    @property
    def slack_variable_count(self) -> int:
        return len(self.layout.bit_registers)

    # ------------------------------------------------------------------
    # Слагаемые модели
    # ------------------------------------------------------------------

    def _add_linear(self, var: int, weight: float) -> None:
        if weight:
            self.linear[var] = self.linear.get(var, 0.0) + weight

    def _add_quadratic(self, u: int, v: int, weight: float) -> None:
        if not weight or u == v:
            if weight and u == v:
                self._add_linear(u, weight)
            return
        key = (u, v) if u < v else (v, u)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + weight

    def _build_network_terms(self) -> None:
        """C_network = sum_{i<j} sum_{a,b} T_ij * L[host_a, host_b] * y_ia * y_jb."""

        problem = self.problem
        traffic = np.asarray(problem.traffic_matrix, dtype=float)
        latency = np.asarray(problem.latency_matrix, dtype=float)

        for i in range(problem.num_virtual_machines):
            hosts_i = problem.allowed_hosts(i)
            for j in range(i + 1, problem.num_virtual_machines):
                intensity = float(traffic[i, j])
                if intensity == 0.0:
                    continue

                hosts_j = problem.allowed_hosts(j)

                for a, host_a in enumerate(hosts_i):
                    var_a = self.layout.var(i, a)
                    for b, host_b in enumerate(hosts_j):
                        weight = intensity * float(latency[host_a, host_b])
                        if weight:
                            self._add_quadratic(var_a, self.layout.var(j, b), weight)

    def _build_resource_terms(self) -> None:
        """Штрафы за нарушение CPU / RAM / пропускной способности."""

        if self.mode == "slack":
            weight = self.penalty.weight
            if weight is None:
                weight = 2.0 * self.network_scale

            for constraint in self.constraints:
                terms: list[tuple[int, float]] = [
                    (self._assignment_var(vm_index, constraint.host), requirement)
                    for vm_index, requirement in constraint.participants
                ]
                terms += [
                    (var, coefficient)
                    for var, coefficient in self._slack_vars(constraint)
                ]
                self._add_square_penalty(terms, constraint.capacity, weight)
            return

        lambda2 = self.penalty.lambda2
        if lambda2 is None:
            lambda2 = self.network_scale
        lambda1 = self.penalty.lambda1
        if lambda1 is None:
            lambda1 = lambda2

        for constraint in self.constraints:
            terms = [
                (self._assignment_var(vm_index, constraint.host), requirement)
                for vm_index, requirement in constraint.participants
            ]

            # λ2 * (load - cap)^2
            self._add_square_penalty(terms, constraint.capacity, lambda2)

            # λ1 * (load - cap)
            for var, coefficient in terms:
                self._add_linear(var, lambda1 * coefficient)
            self.constant -= lambda1 * constraint.capacity

    def _add_square_penalty(
        self,
        terms: Iterable[tuple[int, float]],
        capacity: float,
        weight: float,
    ) -> None:
        """Раскрывает weight * (sum_v w_v * v - capacity)^2 при v^2 = v."""

        items = [(var, float(w)) for var, w in terms if w]

        for var, w in items:
            self._add_linear(var, weight * (w * w - 2.0 * capacity * w))

        for index, (var_u, w_u) in enumerate(items):
            for var_v, w_v in items[index + 1 :]:
                self._add_quadratic(var_u, var_v, weight * 2.0 * w_u * w_v)

        self.constant += weight * capacity * capacity

    def _assignment_var(self, vm_index: int, host_index: int) -> int:
        hosts = self.problem.allowed_hosts(vm_index)
        return self.layout.var(vm_index, hosts.index(host_index))

    def _slack_vars(self, constraint: BindingConstraint) -> list[tuple[int, float]]:
        prefix = f"slack:h{constraint.host}:{constraint.resource}:b"
        result: list[tuple[int, float]] = []
        for register in self.layout.bit_registers:
            if register.label.startswith(prefix):
                result.append((register.first_var, float(register.meta[0])))
        return result

    def _coefficient_scale(self) -> float:
        values = [abs(v) for v in self.linear.values()]
        values += [abs(v) for v in self.quadratic.values()]
        return max(values) if values else 1.0

    # ------------------------------------------------------------------
    # Вычисление энергии
    # ------------------------------------------------------------------

    def energy_from_bits(self, bits) -> float:
        value = self.constant
        for var, weight in self.linear.items():
            if bits[var]:
                value += weight
        for (u, v), weight in self.quadratic.items():
            if bits[u] and bits[v]:
                value += weight
        return float(value)

    def energy_from_levels(self, levels) -> float:
        return self.energy_from_bits(self.layout.bits_from_levels(levels))

    def placement_from_levels(self, levels) -> list[int]:
        """Уровни кудитов -> размещение ВМ по физическим хостам."""

        placement: list[int] = []
        for vm_index in range(self.problem.num_virtual_machines):
            hosts = self.problem.allowed_hosts(vm_index)
            placement.append(int(hosts[int(levels[vm_index])]))
        return placement

    def levels_from_placement(self, placement) -> list[int]:
        levels = []
        for vm_index, host in enumerate(placement):
            hosts = self.problem.allowed_hosts(vm_index)
            levels.append(hosts.index(int(host)))
        # Slack-переменные выбираются оптимально (минимизируют штраф).
        for constraint in self.constraints:
            if self.mode != "slack":
                break
            load = sum(
                requirement
                for vm_index, requirement in constraint.participants
                if int(placement[vm_index]) == constraint.host
            )
            target = max(0.0, constraint.capacity - load)
            coefficients = [c for _, c in self._slack_vars(constraint)]
            for value in _greedy_bits(coefficients, target):
                levels.append(value)
        return levels

    # ------------------------------------------------------------------
    # Представление в пространстве уровней (для кудитного симулятора)
    # ------------------------------------------------------------------

    def level_terms(self) -> tuple[list[np.ndarray], dict[tuple[int, int], np.ndarray]]:
        """
        Перегруппировывает коэффициенты в кудитную форму:

            C(x) = const + sum_i h_i(x_i) + sum_{i<j} J_ij(x_i, x_j)

        Возвращает список векторов h_i и словарь матриц J_ij.
        """

        registers = self.layout.registers
        linear_terms = [np.zeros(register.dim, dtype=float) for register in registers]
        pair_terms: dict[tuple[int, int], np.ndarray] = {}

        for var, weight in self.linear.items():
            register_index, slot = self.layout.owner(var)
            register = registers[register_index]
            for level in range(register.dim):
                linear_terms[register_index][level] += weight * register.indicator(slot, level)

        for (u, v), weight in self.quadratic.items():
            ru, su = self.layout.owner(u)
            rv, sv = self.layout.owner(v)

            if ru == rv:
                register = registers[ru]
                for level in range(register.dim):
                    linear_terms[ru][level] += (
                        weight
                        * register.indicator(su, level)
                        * register.indicator(sv, level)
                    )
                continue

            left, right = (ru, rv) if ru < rv else (rv, ru)
            left_slot, right_slot = (su, sv) if ru < rv else (sv, su)

            matrix = pair_terms.get((left, right))
            if matrix is None:
                matrix = np.zeros((registers[left].dim, registers[right].dim), dtype=float)
                pair_terms[(left, right)] = matrix

            for level_left in range(registers[left].dim):
                indicator_left = registers[left].indicator(left_slot, level_left)
                if indicator_left == 0.0:
                    continue
                for level_right in range(registers[right].dim):
                    matrix[level_left, level_right] += (
                        weight * indicator_left * registers[right].indicator(right_slot, level_right)
                    )

        return linear_terms, pair_terms

    # ------------------------------------------------------------------

    def describe(self) -> dict:
        return {
            "encoding": "one-hot (кудит -> d кубитов)",
            "penalty_mode": self.mode,
            "qudit_dimensions": [r.dim for r in self.layout.qudit_registers],
            "qubits_total": self.layout.n_vars,
            "qubits_assignment": sum(r.n_vars for r in self.layout.qudit_registers),
            "qubits_slack": self.slack_variable_count,
            "binding_constraints": [
                {"host": c.host, "resource": c.resource, "capacity": c.capacity}
                for c in self.constraints
            ],
            "linear_terms": len(self.linear),
            "quadratic_terms": len(self.quadratic),
            "coefficient_scale": self.scale,
            "state_space": self.layout.state_space_size,
        }


def _slack_coefficients(bound: float) -> tuple[float, ...]:
    """
    Ограниченное («bounded coefficient») двоичное разложение slack-переменной.

    Коэффициенты 1, 2, 4, ..., остаток. Позволяет представить любое целое
    значение из [0, bound] и использует меньше переменных, чем разложение
    по степеням двойки до 2^ceil(log2(bound)).
    """

    total = int(ceil(bound - 1e-9))
    if total <= 0:
        return ()

    coefficients: list[float] = []
    remaining = total
    power = 1
    while remaining > 0:
        value = min(power, remaining)
        coefficients.append(float(value))
        remaining -= value
        power *= 2

    return tuple(coefficients)


def _greedy_bits(coefficients: list[float], target: float) -> list[int]:
    """Подбор значений slack-битов, ближайших снизу к target."""

    order = sorted(range(len(coefficients)), key=lambda i: -coefficients[i])
    values = [0] * len(coefficients)
    remaining = target
    for index in order:
        if coefficients[index] <= remaining + 1e-9:
            values[index] = 1
            remaining -= coefficients[index]
    return values
