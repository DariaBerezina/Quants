"""
Кудитная кодировка задачи и её понижение до кубитов.

Кудитная переменная x_i — номер хоста, выбранного для ВМ i. Размерность
кудита d_i равна числу допустимых хостов для этой ВМ (постановка 2.2.4.1).

Платформа ОПКВ принимает от прикладного модуля схему в формате OpenQASM 2.0,
то есть кубитную схему. Поэтому каждый логический кудит понижается до
кубитов унитарным вложением «один кудит → d кубитов» (позиционное, one-hot,
кодирование):

        |a>_d  ->  |0...0 1 0...0>,   единица на позиции a.

Такое вложение выбрано вместо двоичного (ceil(log2 d) кубитов) по трём
причинам:

    1. Оно сохраняет число уровней ровно равным числу допустимых хостов,
       то есть буквально реализует «один кудит размерности d на одну ВМ»;
       при двоичном кодировании и d, не равном степени двойки, появляются
       «лишние» состояния и требуются дополнительные штрафы.
    2. Все слагаемые целевой функции становятся не более чем двухлокальными:
       достаточно однокубитных фазовых гейтов и управляемых фазовых гейтов,
       без многоуправляемых MCP-гейтов, недоступных в базисе qelib1.inc.
    3. Кудитный смеситель (оператор сдвига X_d + X_d^†) точно совпадает
       с XY-смесителем на подпространстве однократного возбуждения, поэтому
       эволюция не выводит систему из допустимого подпространства и
       некорректных результатов измерения не возникает.

Дополнительные (slack) переменные точной QUBO-модели кодируются как
регистры размерности 2 — один кубит на переменную.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

QUDIT = "qudit"
BIT = "bit"


@dataclass(frozen=True, slots=True)
class Register:
    """Логический регистр: кудит ВМ либо однобитовая slack-переменная."""

    index: int
    kind: str
    dim: int
    label: str
    first_var: int
    n_vars: int
    meta: tuple = ()

    def var_indices(self) -> range:
        return range(self.first_var, self.first_var + self.n_vars)

    def indicator(self, slot: int, level: int) -> float:
        """Значение бинарной переменной ``slot`` при уровне ``level``."""

        if self.kind == QUDIT:
            return 1.0 if slot == level else 0.0
        return float(level)


class Layout:
    """
    Раскладка «кудиты + slack» по бинарным переменным и кубитам.

    Индекс бинарной переменной совпадает с индексом кубита в схеме
    OpenQASM (``qreg q[n]``) и с индексом переменной в QUBO-матрице,
    выгружаемой для страницы «Квантовая оптимизация».
    """

    def __init__(self, registers: Sequence[Register]) -> None:
        self.registers = list(registers)
        self.n_vars = sum(register.n_vars for register in self.registers)
        self.dims = [register.dim for register in self.registers]

        self.qudit_registers = [r for r in self.registers if r.kind == QUDIT]
        self.bit_registers = [r for r in self.registers if r.kind == BIT]

        self._var_owner: list[tuple[int, int]] = [(-1, -1)] * self.n_vars
        for register in self.registers:
            for slot, var in enumerate(register.var_indices()):
                self._var_owner[var] = (register.index, slot)

    # ------------------------------------------------------------------

    @property
    def state_space_size(self) -> int:
        size = 1
        for dim in self.dims:
            size *= dim
        return size

    def var(self, register_index: int, slot: int) -> int:
        return self.registers[register_index].first_var + slot

    def owner(self, var: int) -> tuple[int, int]:
        return self._var_owner[var]

    def indicator(self, var: int, level: int) -> float:
        register_index, slot = self._var_owner[var]
        return self.registers[register_index].indicator(slot, level)

    def variable_names(self) -> list[str]:
        names: list[str] = []
        for register in self.registers:
            for slot in range(register.n_vars):
                if register.kind == QUDIT:
                    names.append(f"{register.label}:host{register.meta[slot]}")
                else:
                    names.append(register.label)
        return names

    # ------------------------------------------------------------------
    # Кодирование / декодирование
    # ------------------------------------------------------------------

    def bits_from_levels(self, levels: Sequence[int]) -> np.ndarray:
        bits = np.zeros(self.n_vars, dtype=np.int8)
        for register, level in zip(self.registers, levels):
            if register.kind == QUDIT:
                bits[register.first_var + int(level)] = 1
            else:
                bits[register.first_var] = int(level)
        return bits

    def levels_from_bits(self, bits: Sequence[int]) -> tuple[list[int] | None, bool]:
        """
        Обратное преобразование. Возвращает (уровни, признак корректности).

        Некорректный результат возможен только при исполнении на шумном
        вычислителе: идеальная схема не выводит систему из подпространства
        one-hot.
        """

        levels: list[int] = []
        valid = True

        for register in self.registers:
            chunk = [int(bits[v]) for v in register.var_indices()]
            if register.kind == QUDIT:
                ones = [slot for slot, value in enumerate(chunk) if value == 1]
                if len(ones) != 1:
                    valid = False
                    levels.append(ones[0] if ones else 0)
                else:
                    levels.append(ones[0])
            else:
                levels.append(chunk[0])

        return levels, valid

    # ------------------------------------------------------------------
    # Формат битовых строк ОПКВ / Qiskit
    # ------------------------------------------------------------------

    def bits_from_bitstring(self, bitstring: str, bit_order: str = "little") -> np.ndarray:
        """
        Разбирает ключ словаря счётчиков.

        В Qiskit и в логах ОПКВ старший (левый) символ строки соответствует
        классическому биту с наибольшим номером, поэтому по умолчанию строка
        разворачивается.
        """

        clean = bitstring.replace(" ", "")
        ordered = clean[::-1] if bit_order == "little" else clean

        if len(ordered) < self.n_vars:
            raise ValueError(
                f"Длина битовой строки {len(ordered)} меньше числа кубитов {self.n_vars}."
            )

        return np.array([int(ordered[i]) for i in range(self.n_vars)], dtype=np.int8)

    def bitstring_from_bits(self, bits: Sequence[int], bit_order: str = "little") -> str:
        text = "".join(str(int(b)) for b in bits)
        return text[::-1] if bit_order == "little" else text


def build_layout(
    problem,
    slack_specs: Iterable[tuple[str, float]] = (),
) -> Layout:
    """
    Строит раскладку: сначала кудиты ВМ, затем slack-переменные.

    slack_specs — последовательность пар (метка, вес) для дополнительных
    бинарных переменных точной QUBO-модели.
    """

    registers: list[Register] = []
    cursor = 0
    index = 0

    for vm_index in range(problem.num_virtual_machines):
        hosts = problem.allowed_hosts(vm_index)
        dim = len(hosts)
        registers.append(
            Register(
                index=index,
                kind=QUDIT,
                dim=dim,
                label=f"vm{vm_index}",
                first_var=cursor,
                n_vars=dim,
                meta=hosts,
            )
        )
        cursor += dim
        index += 1

    for label, weight in slack_specs:
        registers.append(
            Register(
                index=index,
                kind=BIT,
                dim=2,
                label=label,
                first_var=cursor,
                n_vars=1,
                meta=(float(weight),),
            )
        )
        cursor += 1
        index += 1

    return Layout(registers)
