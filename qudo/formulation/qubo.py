"""
Выгрузка QUDO-модели в QUBO-матрицу для страницы «Квантовая оптимизация».

Руководство пользователя ОПКВ требует от входных данных страницы
«Квантовая оптимизация» (сервисы SimBif, ICIM RQC) матрицу QUBO в формате
*.npy, симметричную относительно главной диагонали. Целевая функция —

    E(y) = y^T Q y,      y ∈ {0,1}^n

Диагональ Q содержит линейные коэффициенты (y_v^2 = y_v), внедиагональные
элементы — половины квадратичных коэффициентов, что и обеспечивает
симметричность.

Так как классический решатель отжига не «знает» о кудитной структуре,
условие «ровно один хост на ВМ» добавляется штрафом

    P_onehot * sum_i ( sum_a y_{i,a} - 1 )^2

Значение P_onehot по умолчанию заведомо превышает размах остальных
слагаемых, поэтому нарушающие структуру решения отбрасываются решателем.
"""

from __future__ import annotations

import numpy as np

from ..model.qudit import QUDIT
from .qudo_model import QUDOModel


class QUBOExport:
    """QUBO-представление модели + карта переменных для декодирования."""

    def __init__(self, model: QUDOModel, onehot_penalty: float | None = None) -> None:
        self.model = model
        self.layout = model.layout

        if onehot_penalty is None:
            onehot_penalty = 2.0 * (model.scale * max(1, model.layout.n_vars))
        self.onehot_penalty = float(onehot_penalty)

        self.matrix = self._build_matrix()
        self.offset = model.constant + self.onehot_penalty * len(self.layout.qudit_registers)

    def _build_matrix(self) -> np.ndarray:
        size = self.layout.n_vars
        matrix = np.zeros((size, size), dtype=float)

        for var, weight in self.model.linear.items():
            matrix[var, var] += weight

        for (u, v), weight in self.model.quadratic.items():
            matrix[u, v] += weight / 2.0
            matrix[v, u] += weight / 2.0

        # Штраф "ровно один уровень кудита".
        for register in self.layout.registers:
            if register.kind != QUDIT:
                continue
            variables = list(register.var_indices())
            for var in variables:
                matrix[var, var] -= self.onehot_penalty
            for index, u in enumerate(variables):
                for v in variables[index + 1 :]:
                    matrix[u, v] += self.onehot_penalty
                    matrix[v, u] += self.onehot_penalty

        # Гарантия точной симметрии после накопления погрешностей.
        return (matrix + matrix.T) / 2.0

    # ------------------------------------------------------------------

    def energy(self, bits) -> float:
        vector = np.asarray(bits, dtype=float)
        return float(vector @ self.matrix @ vector + self.offset)

    def save(self, path: str) -> str:
        np.save(path, self.matrix)
        return path

    def variables(self) -> list[dict]:
        names = self.layout.variable_names()
        description: list[dict] = []
        for var, name in enumerate(names):
            register_index, slot = self.layout.owner(var)
            register = self.layout.registers[register_index]
            entry = {"index": var, "name": name, "register": register.label, "kind": register.kind}
            if register.kind == QUDIT:
                entry["vm"] = register_index
                entry["host"] = int(register.meta[slot])
            else:
                entry["weight"] = float(register.meta[0])
            description.append(entry)
        return description

    def decode(self, bits) -> tuple[list[int] | None, bool]:
        """Битовый вектор решателя отжига -> размещение ВМ."""

        levels, valid = self.layout.levels_from_bits(bits)
        if levels is None:
            return None, False
        return self.model.placement_from_levels(levels), valid

    def describe(self) -> dict:
        return {
            "format": "npy",
            "shape": list(self.matrix.shape),
            "symmetric": bool(np.allclose(self.matrix, self.matrix.T)),
            "onehot_penalty": self.onehot_penalty,
            "offset": self.offset,
            "variables": self.variables(),
        }
