#!/usr/bin/env python3
"""
Лончер прикладного модуля QUDO для «Облачной платформы квантовых вычислений».

Модуль оформлен по правилам платформы для квантовых приложений
(раздел «Работа с доступными квантовыми приложениями» руководства
пользователя): параметры задаются флагами командной строки, входные и
выходные файлы располагаются в каталоге /data, ход выполнения печатается
в лог запуска, а распределение результатов измерений выводится последней
строкой лога в виде словаря счётчиков — из него платформа строит
гистограмму «Измерение вероятностей».

Режимы работы (--mode):

    solve   — построить кудитную модель, подобрать параметры QAOA, выполнить
              запуск на выбранном вычислителе, декодировать результат;
    export  — только подготовить артефакты для платформы: схему OpenQASM 2.0
              для страницы «Квантовые вычисления» и QUBO-матрицу *.npy
              для страницы «Квантовая оптимизация»;
    decode  — разобрать словарь счётчиков, полученный после запуска схемы
              на сервисе платформы, и выдать итоговое размещение ВМ.

Примеры запуска (в формате параметра cmd лончера ОПКВ):

    qudo --template small --backend qiskit --shots 1024 --layers 2 \
         --output-file /data/qudo-output.json --qasm-file /data/qudo-circuit.qasm

    qudo --input /data/dc.json --mode export --backend ideem_ideal_30 \
         --qasm-file /data/qudo-circuit.qasm --qubo-file /data/qudo-qubo.npy

    qudo --input /data/dc.json --mode decode --counts-file /data/counts.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Пакет лежит рядом с лончером: это позволяет запускать модуль как
# `python qudo_launcher.py ...` внутри контейнера без установки.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qudo.data.examples import TEMPLATES, load_template
from qudo.formulation.qubo import QUBOExport
from qudo.formulation.qudo_model import PENALTY_MODES, PenaltyConfig, QUDOModel
from qudo.model.instance import AllocationProblem
from qudo.optimizer.decoder import Decoder
from qudo.optimizer.experiment import Experiment, ExperimentConfig
from qudo.optimizer.objective import ObjectiveFunction
from qudo.optimizer.baselines import ExhaustiveSearch, GreedySearch
from qudo.optimizer.validation import validate_model
from qudo.platform import (
    PlatformLogger,
    read_json,
    resolve_path,
    write_json,
    write_text,
)
from qudo.quantum.backends import PLATFORM_SERVICES, create_backend
from qudo.quantum.qasm import build_qaoa_qasm, circuit_statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qudo",
        description=(
            "Кудитный алгоритм дискретной оптимизации распределения ресурсов: "
            "размещение виртуальных машин по физическим хостам с учётом сетевой связности."
        ),
    )

    source = parser.add_argument_group("входные данные")
    source.add_argument("--input", default=None, help="JSON с описанием ЦОД и нагрузки (/data/...)")
    source.add_argument(
        "--template",
        default=None,
        choices=sorted(TEMPLATES),
        help="встроенный тестовый экземпляр вместо файла",
    )
    source.add_argument(
        "--counts-file",
        default=None,
        help="JSON со счётчиками измерений платформы (режим decode)",
    )

    outputs = parser.add_argument_group("выходные данные")
    outputs.add_argument("--output-file", default="qudo-output.json", help="отчёт о решении")
    outputs.add_argument("--qasm-file", default=None, help="схема OpenQASM 2.0")
    outputs.add_argument("--qubo-file", default=None, help="QUBO-матрица в формате npy")
    outputs.add_argument("--save-problem", default=None, help="выгрузить экземпляр задачи в JSON")
    outputs.add_argument("--print-qasm", action="store_true", help="вывести схему в лог")

    run = parser.add_argument_group("параметры запуска")
    run.add_argument("--mode", default="solve", choices=("solve", "export", "decode"))
    run.add_argument(
        "--backend",
        default="local",
        choices=sorted(PLATFORM_SERVICES),
        help="вычислитель платформы",
    )
    run.add_argument("--shots", type=int, default=1024, help="количество запусков схемы")
    run.add_argument("--layers", type=int, default=2, help="число слоёв QAOA (p)")
    run.add_argument("--optimizer", default="cobyla", choices=("cobyla", "nelder-mead"))
    run.add_argument("--restarts", type=int, default=3, help="число стартов классической оптимизации")
    run.add_argument("--max-evaluations", type=int, default=200)
    run.add_argument("--objective", default="expectation", choices=("expectation", "cvar"))
    run.add_argument("--cvar-alpha", type=float, default=0.2)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--bit-order", default="little", choices=("little", "big"))

    model = parser.add_argument_group("модель")
    model.add_argument("--penalty", default="auto", choices=PENALTY_MODES)
    model.add_argument("--penalty-weight", type=float, default=None, help="P для режима slack")
    model.add_argument("--lambda1", type=float, default=None)
    model.add_argument("--lambda2", type=float, default=None)
    model.add_argument("--max-slack-vars", type=int, default=24)
    model.add_argument("--no-validate", action="store_true", help="не проверять модель перебором")
    model.add_argument("--no-baselines", action="store_true", help="не считать эталоны")

    parser.add_argument("--verbose", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--json", action="store_true", help="вывести отчёт в лог в формате JSON")

    return parser


# ----------------------------------------------------------------------


def load_problem(args, logger: PlatformLogger) -> AllocationProblem:
    if args.input:
        path = resolve_path(args.input)
        logger.log(f"Загрузка входных данных: {path}")
        return AllocationProblem.from_dict(read_json(path))

    template = args.template or "small"
    logger.log(f"Загрузка входных данных: встроенный экземпляр '{template}'")
    return load_template(template)


def build_penalty_config(args) -> PenaltyConfig:
    return PenaltyConfig(
        mode=args.penalty,
        weight=args.penalty_weight,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        max_slack_vars=args.max_slack_vars,
    )


def write_artifacts(args, model: QUDOModel, gammas, betas, logger: PlatformLogger) -> dict:
    artifacts: dict[str, str] = {}

    if args.qasm_file:
        path = resolve_path(args.qasm_file)
        text = build_qaoa_qasm(
            model,
            gammas,
            betas,
            measure=True,
            header=[
                f"instance: {model.problem.name}",
                f"penalty mode: {model.mode}",
                f"target service: {args.backend}",
            ],
        )
        write_text(path, text)
        artifacts["qasm"] = path
        stats = circuit_statistics(text)
        artifacts["circuit"] = stats
        logger.log(
            f"Схема OpenQASM 2.0 сохранена: {path} "
            f"(кубитов {stats['qubits']}, вентилей {stats['gates_total']}, "
            f"двухкубитных {stats['two_qubit_gates']}, глубина ~{stats['depth_estimate']})"
        )
        if args.print_qasm:
            for line in text.splitlines():
                logger.log(line, level=2)

    if args.qubo_file:
        path = resolve_path(args.qubo_file)
        export = QUBOExport(model)
        export.save(path)
        artifacts["qubo"] = path
        artifacts["qubo_variables"] = write_json(
            path.replace(".npy", "") + "-variables.json", export.describe()
        )
        logger.log(
            f"QUBO-матрица сохранена: {path} "
            f"(размер {export.matrix.shape[0]}x{export.matrix.shape[1]}, симметрична)"
        )

    return artifacts


def platform_hint(args, model: QUDOModel, logger: PlatformLogger) -> None:
    service = PLATFORM_SERVICES[args.backend]

    if service.kind == "qasm":
        logger.log(
            "Для запуска на сервисе платформы: страница «Квантовые вычисления» -> "
            "вкладка QASM -> вставить схему -> «Сохранить» -> выбрать сервис "
            f"{service.title}, шоты {args.shots} -> «Выполнить». "
            "Полученный лог со счётчиками сохранить и передать модулю "
            "в режиме --mode decode --counts-file."
        )
    elif service.kind == "qubo":
        logger.log(
            "Для запуска отжига: страница «Квантовая оптимизация» -> «Загрузить .npy файл» -> "
            f"выбрать сервис {service.title} -> «Выполнить»."
        )

    if model.mode == "unbalanced":
        logger.log(
            "Внимание: ресурсные ограничения учтены приближённо (режим unbalanced, "
            "без slack-переменных). Допустимость итогового размещения проверяется "
            "классически при декодировании измерений."
        )


# ----------------------------------------------------------------------


def run_solve(args, problem: AllocationProblem, logger: PlatformLogger) -> dict:
    config = ExperimentConfig(
        backend=args.backend,
        layers=args.layers,
        shots=args.shots,
        optimizer=args.optimizer,
        restarts=args.restarts,
        max_evaluations=args.max_evaluations,
        objective=args.objective,
        cvar_alpha=args.cvar_alpha,
        seed=args.seed,
        penalty=build_penalty_config(args),
        validate=not args.no_validate,
        baselines=not args.no_baselines,
    )

    experiment = Experiment(problem, config, logger=logger)
    outcome = experiment.run()

    artifacts = write_artifacts(args, outcome.model, outcome.qaoa.gammas, outcome.qaoa.betas, logger)
    platform_hint(args, outcome.model, logger)

    report = outcome.report
    report["mode"] = "solve"
    report["artifacts"] = artifacts
    report["counts"] = outcome.counts

    _log_solution(report, logger)
    logger.counts(outcome.counts)

    return report


def run_export(args, problem: AllocationProblem, logger: PlatformLogger) -> dict:
    model = QUDOModel(problem, build_penalty_config(args))
    objective = ObjectiveFunction(problem)

    logger.log(
        "Кудитная модель построена: "
        f"кудитов {len(model.layout.qudit_registers)}, "
        f"кубитов {model.layout.n_vars}, режим штрафов {model.mode}"
    )

    validation = validate_model(model, objective) if not args.no_validate else None
    if validation is not None and validation.checked:
        logger.log(f"Проверка модели: {validation.message}")

    if PLATFORM_SERVICES[args.backend].kind == "qubo":
        # Сервисам квантового отжига схема не нужна: достаточно QUBO-матрицы,
        # поэтому параметры QAOA не подбираются.
        if not args.qubo_file:
            args.qubo_file = "qudo-qubo.npy"
        artifacts = write_artifacts(args, model, [], [], logger)
        platform_hint(args, model, logger)
        return {
            "mode": "export",
            "problem": problem.summary(),
            "model": model.describe(),
            "validation": validation.to_dict() if validation is not None else None,
            "target_service": {
                "key": args.backend,
                "title": PLATFORM_SERVICES[args.backend].title,
                "kind": "qubo",
            },
            "artifacts": artifacts,
        }

    backend = create_backend("local", model, args.seed)
    from qudo.quantum.qaoa import QuditQAOA

    qaoa = QuditQAOA(
        model,
        backend,
        layers=args.layers,
        optimizer=args.optimizer,
        restarts=args.restarts,
        max_evaluations=args.max_evaluations,
        objective=args.objective,
        cvar_alpha=args.cvar_alpha,
        seed=args.seed,
    )

    logger.log("Подбор параметров QAOA локальной кудитной эмуляцией")
    result = qaoa.optimize(logger=logger)
    logger.log(
        f"Параметры: gamma={[round(v, 4) for v in result.gammas]}, "
        f"beta={[round(v, 4) for v in result.betas]}, <C>={result.expectation:.4f}"
    )

    if not args.qasm_file and not args.qubo_file:
        args.qasm_file = "qudo-circuit.qasm"
        args.qubo_file = "qudo-qubo.npy"

    artifacts = write_artifacts(args, model, result.gammas, result.betas, logger)
    platform_hint(args, model, logger)

    return {
        "mode": "export",
        "problem": problem.summary(),
        "model": model.describe(),
        "validation": validation.to_dict() if validation is not None else None,
        "qaoa": {
            "layers": args.layers,
            "gamma": result.gammas,
            "beta": result.betas,
            "expectation": result.expectation,
            "evaluations": result.evaluations,
            "optimizer": result.method,
        },
        "target_service": {
            "key": args.backend,
            "title": PLATFORM_SERVICES[args.backend].title,
            "kind": PLATFORM_SERVICES[args.backend].kind,
            "shots": args.shots,
        },
        "artifacts": artifacts,
    }


def run_decode(args, problem: AllocationProblem, logger: PlatformLogger) -> dict:
    if not args.counts_file:
        raise ValueError("Для режима decode требуется параметр --counts-file.")

    path = resolve_path(args.counts_file)
    logger.log(f"Загрузка результатов запуска: {path}")

    payload = read_json(path)
    counts = payload.get("counts", payload) if isinstance(payload, dict) else payload
    counts = {str(key): int(value) for key, value in counts.items()}

    model = QUDOModel(problem, build_penalty_config(args))
    objective = ObjectiveFunction(problem)

    decode = Decoder(model, objective, bit_order=args.bit_order).decode(counts)
    logger.log(
        f"Разобрано {decode.total_shots} измерений, "
        f"корректных {decode.valid_fraction * 100:.1f}%, "
        f"допустимых {decode.feasible_fraction * 100:.1f}%"
    )

    best = decode.best_feasible or decode.best_overall
    if best is None:
        raise ValueError("В переданных счётчиках нет ни одного корректного состояния.")

    report: dict = {
        "mode": "decode",
        "problem": problem.summary(),
        "model": model.describe(),
        "solution": objective.report(best.placement),
        "statistics": {
            "total_shots": decode.total_shots,
            "valid_fraction": decode.valid_fraction,
            "feasible_fraction": decode.feasible_fraction,
            "probability_of_solution": decode.probability_of(best.placement),
        },
        "baselines": {},
        "gaps": {},
        "counts": counts,
    }

    if not args.no_baselines:
        greedy = GreedySearch(objective).solve()
        exhaustive = ExhaustiveSearch(objective).solve()
        report["baselines"]["greedy"] = {
            "placement": greedy.placement,
            "cost": greedy.cost,
            "feasible": greedy.feasible,
        }
        if exhaustive is not None and exhaustive.placement:
            report["baselines"]["exhaustive"] = {
                "placement": exhaustive.placement,
                "cost": exhaustive.cost,
                "explored_states": exhaustive.explored_states,
            }
            if exhaustive.cost:
                report["gaps"]["qaoa_percent"] = (
                    (best.cost - exhaustive.cost) / exhaustive.cost * 100.0
                )
                report["gaps"]["greedy_percent"] = (
                    (greedy.cost - exhaustive.cost) / exhaustive.cost * 100.0
                )

    _log_solution(report, logger)
    logger.counts(counts)

    return report


# ----------------------------------------------------------------------


def _log_solution(report: dict, logger: PlatformLogger) -> None:
    solution = report.get("solution")
    if not solution:
        return

    logger.log(
        "Найдено размещение ВМ по хостам: "
        f"{solution['placement']}, суммарная сетевая задержка {solution['network_cost']:.2f}, "
        f"ограничения {'соблюдены' if solution['feasible'] else 'НАРУШЕНЫ'}",
        level=0,
    )

    baselines = report.get("baselines", {})
    if "exhaustive" in baselines:
        logger.log(
            f"Глобальный оптимум (полный перебор): {baselines['exhaustive']['placement']}, "
            f"стоимость {baselines['exhaustive']['cost']:.2f}",
            level=0,
        )
    if "greedy" in baselines:
        logger.log(
            f"Жадный алгоритм: {baselines['greedy']['placement']}, "
            f"стоимость {baselines['greedy']['cost']:.2f}",
            level=1,
        )

    gaps = report.get("gaps", {})
    if gaps.get("qaoa_percent") is not None:
        logger.log(f"Отклонение QAOA от оптимума: {gaps['qaoa_percent']:.2f}%", level=0)
    if gaps.get("greedy_percent") is not None:
        logger.log(f"Отклонение жадного алгоритма: {gaps['greedy_percent']:.2f}%", level=1)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logger = PlatformLogger(verbose=args.verbose)

    logger.log(
        f"Запущена задача QUDO: размещение виртуальных машин, режим {args.mode}, "
        f"сервис {args.backend}",
        level=0,
    )

    try:
        problem = load_problem(args, logger)
        problem.validate()

        logger.log(
            f"Инфраструктура: хостов {problem.num_hosts}, ВМ {problem.num_virtual_machines}, "
            f"размерности кудитов {problem.qudit_dimensions()}"
        )

        if not problem.is_potentially_feasible():
            logger.log(
                "Предупреждение: суммарных ресурсов ЦОД недостаточно для всей нагрузки — "
                "допустимого размещения не существует."
            )

        if args.save_problem:
            path = write_json(resolve_path(args.save_problem), problem.to_dict())
            logger.log(f"Экземпляр задачи сохранён: {path}")

        if args.mode == "solve":
            report = run_solve(args, problem, logger)
        elif args.mode == "export":
            report = run_export(args, problem, logger)
        else:
            report = run_decode(args, problem, logger)

        if args.output_file:
            path = write_json(resolve_path(args.output_file), report)
            logger.log(f"Отчёт сохранён: {path}", level=0)

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))

    except Exception as error:  # noqa: BLE001 - лончер обязан вернуть код ошибки
        logger.error(str(error))
        if args.verbose >= 2:
            raise
        return 1

    logger.report_elapsed()
    logger.log("Счёт задачи завершён успешно", level=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
