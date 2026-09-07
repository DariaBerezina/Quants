#!/usr/bin/env python3
"""
Автономный набор проверок проекта QUDO.

Запуск:  python tests/run_tests.py

Проверяется главное свойство адаптации: локальная кудитная эмуляция и
сгенерированная схема OpenQASM 2.0 описывают одно и то же квантовое
состояние. Если это так, параметры (γ, β), подобранные внутри контейнера
лончера, можно без изменений использовать при запуске схемы на сервисах
платформы, а результат локального прогона служит эталоном для сравнения
с результатом реального вычислителя.

Проверки, требующие qiskit, пропускаются, если пакет не установлен.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qudo.data.examples import create_resource_constrained_problem, create_small_problem
from qudo.formulation.qubo import QUBOExport
from qudo.formulation.qudo_model import PenaltyConfig, QUDOModel
from qudo.optimizer.decoder import Decoder
from qudo.optimizer.objective import ObjectiveFunction
from qudo.optimizer.validation import validate_model
from qudo.quantum.backends import LocalQuditBackend, PLATFORM_SERVICES
from qudo.quantum.qasm import build_qaoa_qasm

PASSED: list[str] = []
SKIPPED: list[str] = []


def check(name: str, condition: bool, details: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: ПРОВАЛ. {details}")
    PASSED.append(name)
    print(f"  [OK] {name}")


def skip(name: str, reason: str) -> None:
    SKIPPED.append(name)
    print(f"  [--] {name}: пропущено ({reason})")


# ----------------------------------------------------------------------


def test_model_consistency() -> None:
    print("Модель QUDO против точной целевой функции")

    for factory in (create_small_problem, create_resource_constrained_problem):
        problem = factory()
        model = QUDOModel(problem, PenaltyConfig(mode="auto"))
        objective = ObjectiveFunction(problem)
        result = validate_model(model, objective)

        check(
            f"argmin модели == точный оптимум ({problem.name}, режим {model.mode})",
            result.consistent is True,
            result.message,
        )


def test_qubo_export() -> None:
    print("Выгрузка QUBO для страницы «Квантовая оптимизация»")

    problem = create_small_problem()
    model = QUDOModel(problem, PenaltyConfig(mode="auto"))
    export = QUBOExport(model)

    check(
        "матрица симметрична относительно главной диагонали",
        bool(np.allclose(export.matrix, export.matrix.T)),
    )

    size = export.matrix.shape[0]
    check("размер матрицы равен числу переменных", size == model.layout.n_vars)

    # Полный перебор бинарных векторов: минимум QUBO должен давать
    # то же размещение, что и точный оптимум задачи.
    objective = ObjectiveFunction(problem)
    best_energy = float("inf")
    best_bits = None

    for bits in itertools.product((0, 1), repeat=size):
        vector = np.array(bits, dtype=float)
        energy = float(vector @ export.matrix @ vector)
        if energy < best_energy:
            best_energy = energy
            best_bits = bits

    placement, valid = export.decode(np.array(best_bits, dtype=int))
    check("минимум QUBO лежит в подпространстве one-hot", valid)
    check(
        "минимум QUBO соответствует оптимальному размещению",
        placement == [2, 2, 2, 2],
        f"получено {placement}",
    )
    check("решение допустимо", objective.is_feasible(placement))


def test_decoder_roundtrip() -> None:
    print("Кодирование и декодирование битовых строк")

    problem = create_small_problem()
    model = QUDOModel(problem, PenaltyConfig(mode="auto"))
    objective = ObjectiveFunction(problem)
    decoder = Decoder(model, objective)

    placement = [0, 1, 2, 3]
    levels = model.levels_from_placement(placement)
    bits = model.layout.bits_from_levels(levels)
    bitstring = model.layout.bitstring_from_bits(bits)

    report = decoder.decode({bitstring: 100})
    check(
        "размещение восстанавливается из битовой строки",
        report.best_overall is not None and report.best_overall.placement == placement,
    )
    check("все выборки признаны корректными", report.valid_fraction == 1.0)

    broken = "1" * model.layout.n_vars
    report = decoder.decode({broken: 10})
    check("нарушение one-hot распознаётся", report.valid_fraction == 0.0)


def test_qasm_matches_emulation() -> None:
    print("Эквивалентность схемы OpenQASM и кудитной эмуляции")

    try:
        from qiskit import qasm2
        from qiskit.quantum_info import Statevector
    except ImportError:
        skip("схема OpenQASM == кудитная эмуляция", "qiskit не установлен")
        return

    problem = create_small_problem()
    model = QUDOModel(problem, PenaltyConfig(mode="auto"))
    backend = LocalQuditBackend(model, PLATFORM_SERVICES["local"], seed=1)

    gammas = [0.63, 1.17]
    betas = [0.41, 0.92]

    text = build_qaoa_qasm(model, gammas, betas, measure=False)
    circuit = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    amplitudes = np.asarray(Statevector.from_instruction(circuit))
    probabilities_circuit = np.abs(amplitudes) ** 2

    probabilities_emulator = backend.simulator.probabilities(gammas, betas)

    layout = model.layout
    mapped = np.zeros_like(probabilities_emulator)
    onehot_mass = 0.0

    for index, probability in enumerate(probabilities_circuit):
        if probability < 1e-15:
            continue
        bits = [(index >> qubit) & 1 for qubit in range(layout.n_vars)]
        levels, valid = layout.levels_from_bits(bits)
        if not valid:
            continue
        onehot_mass += probability
        flat = int(np.ravel_multi_index(tuple(levels), layout.dims))
        mapped[flat] += probability

    check(
        "схема не покидает подпространство one-hot",
        abs(onehot_mass - 1.0) < 1e-9,
        f"вес допустимого подпространства {onehot_mass:.12f}",
    )
    check(
        "распределения схемы и эмулятора совпадают",
        bool(np.allclose(mapped, probabilities_emulator, atol=1e-9)),
        f"максимальное расхождение {np.max(np.abs(mapped - probabilities_emulator)):.3e}",
    )

    # Ожидаемая энергия, посчитанная по схеме, совпадает с эмуляцией.
    energy_circuit = float(mapped @ backend.simulator.flat_cost + model.constant)
    energy_emulator = backend.simulator.expectation(gammas, betas)
    check(
        "средняя энергия совпадает",
        abs(energy_circuit - energy_emulator) < 1e-9,
        f"{energy_circuit} vs {energy_emulator}",
    )


def test_qasm_gate_set() -> None:
    print("Совместимость схемы с базисом платформы")

    problem = create_small_problem()
    model = QUDOModel(problem, PenaltyConfig(mode="auto"))
    text = build_qaoa_qasm(model, [0.5], [0.5])

    allowed = {"x", "h", "rx", "ry", "rz", "u1", "u3", "cx", "barrier", "measure"}
    used = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM"):
            continue
        if line.startswith("include") or line.startswith("qreg") or line.startswith("creg"):
            continue
        used.add(line.split("(")[0].split(" ")[0].split(";")[0])

    check(
        "используются только вентили qelib1.inc из перечня редактора платформы",
        used <= allowed,
        f"лишние вентили: {sorted(used - allowed)}",
    )
    check("схема объявляет формат OpenQASM 2.0", text.lstrip().splitlines()[0].startswith("//"))
    check('подключается qelib1.inc', 'include "qelib1.inc";' in text)


def test_constrained_instance_quantum_run() -> None:
    print("Сквозной прогон экземпляра с дефицитом ресурсов")

    from qudo.optimizer.experiment import Experiment, ExperimentConfig

    problem = create_resource_constrained_problem()
    config = ExperimentConfig(
        backend="local",
        layers=3,
        shots=2048,
        restarts=2,
        objective="cvar",
        seed=11,
        penalty=PenaltyConfig(mode="auto"),
    )
    outcome = Experiment(problem, config).run()

    solution = outcome.report["solution"]
    check("итоговое размещение допустимо", bool(solution["feasible"]))
    check(
        "достигнут глобальный оптимум",
        abs(outcome.report["gaps"]["qaoa_percent"]) < 1e-9,
        f"gap = {outcome.report['gaps']['qaoa_percent']}",
    )


def main() -> int:
    tests = [
        test_model_consistency,
        test_qubo_export,
        test_decoder_roundtrip,
        test_qasm_matches_emulation,
        test_qasm_gate_set,
        test_constrained_instance_quantum_run,
    ]

    for test in tests:
        test()
        print()

    print(f"Пройдено проверок: {len(PASSED)}, пропущено: {len(SKIPPED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
