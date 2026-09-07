"""
Вычислители (backends) для кудитного QAOA.

Платформа предоставляет несколько типов сервисов (Таблица 4 руководства
пользователя ОПКВ). С точки зрения прикладного модуля они делятся на три
группы:

    * локальная кудитная эмуляция внутри контейнера лончера ("local") —
      не требует ни qiskit, ни обращений к сервисам платформы;
    * исполнение схемы прямо из кода через открытый эмулятор Qiskit Aer
      ("qiskit") — соответствует сервису Qiskit Aer Simulator IBM;
    * сервисы платформы, принимающие QASM-схему (ideem_ideal_30,
      ideem_noisy_30, superconducting, atom_msu, iqc_yb, iqc_ca, triangulum).
      Для них лончер подбирает параметры локально и выгружает готовую схему;
      запуск выполняется средствами платформы, а результат возвращается
      в модуль в режиме --mode decode.

Отдельно стоят сервисы квантового отжига (SimBif, ICIM RQC): они принимают
не схему, а QUBO-матрицу в формате npy (см. formulation/qubo.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..formulation.qudo_model import QUDOModel
from .qasm import build_qaoa_qasm
from .simulator import QuditQAOASimulator


@dataclass(frozen=True, slots=True)
class ServiceInfo:
    key: str
    title: str
    kind: str                 # "local" | "aer" | "qasm" | "qubo"
    max_qubits: int | None = None
    note: str = ""


PLATFORM_SERVICES: dict[str, ServiceInfo] = {
    "local": ServiceInfo("local", "Локальная кудитная эмуляция (контейнер лончера)", "local"),
    "qiskit": ServiceInfo("qiskit", "Qiskit Aer Simulator IBM", "aer"),
    "ideem_ideal_30": ServiceInfo(
        "ideem_ideal_30", "Идеальный эмулятор на 30 кубитов", "qasm", 30
    ),
    "ideem_noisy_30": ServiceInfo(
        "ideem_noisy_30", "Шумный эмулятор на 30 кубитов", "qasm", 30
    ),
    "triangulum": ServiceInfo("triangulum", "SpinQ Triangulum", "qasm", 3),
    "superconducting": ServiceInfo(
        "superconducting", "Superconducting Qubits MISIS", "qasm", None,
        "Экспериментальный образец: возможны шумовые искажения результата.",
    ),
    "atom_msu": ServiceInfo(
        "atom_msu", "Atom MSU", "qasm", None,
        "Экспериментальный образец на холодных нейтральных атомах.",
    ),
    "iqc_yb": ServiceInfo("iqc_yb", "IQC Yb+ RQC", "qasm", None),
    "iqc_ca": ServiceInfo("iqc_ca", "IQC Ca+ RQC", "qasm", None),
    "iqc_qudit_sim": ServiceInfo(
        "iqc_qudit_sim", "IQC Yb+ Simulator RQC (кудитный эмулятор)", "qasm", None,
        "Кудитный вычислитель: схема выгружается в QASM, кудитная модель — в JSON.",
    ),
    "simbif": ServiceInfo("simbif", "SimBif (эмулятор бифуркационной машины)", "qubo"),
    "icim": ServiceInfo("icim", "ICIM RQC (когерентная машина Изинга)", "qubo"),
}


class BaseBackend:
    """Базовый интерфейс вычислителя."""

    def __init__(self, model: QUDOModel, service: ServiceInfo, seed: int | None = None) -> None:
        self.model = model
        self.service = service
        self.rng = np.random.default_rng(seed)

        linear_terms, pair_terms = model.level_terms()
        self.simulator = QuditQAOASimulator(
            dims=model.layout.dims,
            linear_terms=linear_terms,
            pair_terms=pair_terms,
            constant=model.constant,
            scale=model.scale,
        )

    # -- API -----------------------------------------------------------

    @property
    def name(self) -> str:
        return self.service.key

    @property
    def executes_locally(self) -> bool:
        return self.service.kind in ("local", "aer")

    def expectation(self, gammas: Sequence[float], betas: Sequence[float]) -> float:
        return self.simulator.expectation(gammas, betas)

    def sample(self, gammas, betas, shots: int) -> dict[str, int]:
        raise NotImplementedError

    def qasm(self, gammas, betas, measure: bool = True) -> str:
        return build_qaoa_qasm(
            self.model,
            gammas,
            betas,
            measure=measure,
            header=[f"target service: {self.service.key} ({self.service.title})"],
        )

    # -- вспомогательное -----------------------------------------------

    def _counts_from_flat(self, flat_counts: dict[int, int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for index, shots in flat_counts.items():
            levels = self.simulator.levels_from_flat_index(index)
            bits = self.model.layout.bits_from_levels(levels)
            key = self.model.layout.bitstring_from_bits(bits)
            counts[key] = counts.get(key, 0) + int(shots)
        return counts


class LocalQuditBackend(BaseBackend):
    """Точная кудитная эмуляция внутри контейнера лончера."""

    def sample(self, gammas, betas, shots: int) -> dict[str, int]:
        flat_counts = self.simulator.sample(gammas, betas, shots, self.rng)
        return self._counts_from_flat(flat_counts)


class QasmExportBackend(BaseBackend):
    """
    Сервис платформы, принимающий QASM-схему.

    Подбор параметров выполняется локально (кудитная эмуляция полностью
    соответствует генерируемой схеме), а сам запуск делает платформа.
    Метод sample возвращает эмулированное распределение — оно служит
    прогнозом и эталоном для сравнения с результатом реального запуска.
    """

    def sample(self, gammas, betas, shots: int) -> dict[str, int]:
        flat_counts = self.simulator.sample(gammas, betas, shots, self.rng)
        return self._counts_from_flat(flat_counts)


class QiskitAerBackend(BaseBackend):
    """Исполнение схемы через открытый эмулятор Qiskit Aer (лицензия MIT)."""

    def __init__(self, model: QUDOModel, service: ServiceInfo, seed: int | None = None) -> None:
        super().__init__(model, service, seed)

        try:
            from qiskit import qasm2, transpile          # noqa: F401
            from qiskit_aer import AerSimulator
        except ImportError as error:  # pragma: no cover - зависит от образа
            raise RuntimeError(
                "Для backend=qiskit требуются пакеты qiskit и qiskit-aer. "
                "Используйте backend=local либо соберите образ с этими пакетами."
            ) from error

        self._qasm2 = qasm2
        self._transpile = transpile
        self._simulator = AerSimulator(seed_simulator=seed)

    def sample(self, gammas, betas, shots: int) -> dict[str, int]:
        text = self.qasm(gammas, betas, measure=True)
        circuit = self._qasm2.loads(text, custom_instructions=self._qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        compiled = self._transpile(circuit, self._simulator)
        job = self._simulator.run(compiled, shots=shots)
        return {str(key): int(value) for key, value in job.result().get_counts().items()}

    def expectation(self, gammas, betas) -> float:
        """Оценка среднего по выборке (как на реальном сервисе)."""

        shots = getattr(self, "expectation_shots", 0)
        if not shots:
            return super().expectation(gammas, betas)

        counts = self.sample(gammas, betas, shots)
        total = 0
        energy = 0.0
        for bitstring, repeats in counts.items():
            bits = self.model.layout.bits_from_bitstring(bitstring)
            energy += repeats * self.model.energy_from_bits(bits)
            total += repeats
        return energy / total if total else float("inf")


def create_backend(name: str, model: QUDOModel, seed: int | None = None) -> BaseBackend:
    service = PLATFORM_SERVICES.get(name)
    if service is None:
        raise ValueError(
            f"Неизвестный сервис '{name}'. Доступны: {', '.join(sorted(PLATFORM_SERVICES))}"
        )

    if service.kind == "local":
        return LocalQuditBackend(model, service, seed)
    if service.kind == "aer":
        return QiskitAerBackend(model, service, seed)
    if service.kind == "qasm":
        return QasmExportBackend(model, service, seed)

    raise ValueError(
        f"Сервис '{name}' работает с QUBO-матрицей, а не со схемой. "
        "Выгрузите матрицу параметром --qubo-file и запустите её на странице "
        "«Квантовая оптимизация»."
    )
