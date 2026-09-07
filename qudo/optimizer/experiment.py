"""
Сквозной сценарий решения задачи QUDO на платформе.

Порядок действий соответствует постановке 2.2.4.1:

    1. по описанию инфраструктуры ЦОД и характеристикам нагрузки строится
       кудитная формулировка в виде модели квадратичной безусловной
       оптимизации (модуль formulation);
    2. модель проверяется на согласованность с исходной задачей;
    3. задача решается кудитным вариантом QAOA на выбранном вычислителе;
    4. результаты измерений декодируются в размещение ВМ по хостам;
    5. решение сравнивается с жадным алгоритмом и (на малых экземплярах)
       с глобальным оптимумом.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..formulation.qudo_model import PenaltyConfig, QUDOModel
from ..model.instance import AllocationProblem
from ..quantum.backends import create_backend
from ..quantum.qaoa import QAOAResult, QuditQAOA
from .baselines import ExhaustiveSearch, GreedySearch, SolutionRecord
from .decoder import DecodeReport, Decoder
from .objective import ObjectiveFunction, PenaltyWeights
from .validation import ModelValidation, validate_model


@dataclass(slots=True)
class ExperimentConfig:
    backend: str = "local"
    layers: int = 2
    shots: int = 1024
    optimizer: str = "cobyla"
    restarts: int = 3
    max_evaluations: int = 200
    objective: str = "expectation"
    cvar_alpha: float = 0.2
    seed: int | None = 7
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    validate: bool = True
    baselines: bool = True


@dataclass(slots=True)
class ExperimentOutcome:
    model: QUDOModel
    qaoa: QAOAResult
    decode: DecodeReport
    validation: ModelValidation
    greedy: SolutionRecord | None
    exhaustive: SolutionRecord | None
    counts: dict[str, int]
    backend_name: str
    report: dict[str, Any]


class Experiment:
    def __init__(
        self,
        problem: AllocationProblem,
        config: ExperimentConfig | None = None,
        logger=None,
    ) -> None:
        self.problem = problem
        self.config = config or ExperimentConfig()
        self.logger = logger

        self.objective = ObjectiveFunction(problem, PenaltyWeights())
        self.model = QUDOModel(problem, self.config.penalty)

    # ------------------------------------------------------------------

    def _log(self, message: str, level: int = 1) -> None:
        if self.logger is not None:
            self.logger.log(message, level=level)

    def run(self) -> ExperimentOutcome:
        config = self.config
        description = self.model.describe()

        self._log(
            "Кудитная модель построена: "
            f"кудитов {len(description['qudit_dimensions'])}, "
            f"размерности {description['qudit_dimensions']}, "
            f"кубитов {description['qubits_total']} "
            f"(размещение {description['qubits_assignment']}, slack {description['qubits_slack']})"
        )
        self._log(
            f"Штрафы: режим {description['penalty_mode']}, "
            f"связывающих ограничений {len(description['binding_constraints'])}; "
            f"слагаемых: линейных {description['linear_terms']}, "
            f"квадратичных {description['quadratic_terms']}"
        )

        validation = ModelValidation(checked=False, message="Проверка модели отключена.")
        if config.validate:
            validation = self._validate_and_tune()
            if validation.checked:
                self._log(f"Проверка модели: {validation.message}", level=1)
            else:
                self._log(validation.message, level=2)

        backend = create_backend(config.backend, self.model, config.seed)
        self._log(f"Вычислитель: {backend.service.title}")

        if backend.service.max_qubits is not None and (
            self.model.layout.n_vars > backend.service.max_qubits
        ):
            raise ValueError(
                f"Схеме требуется {self.model.layout.n_vars} кубитов, "
                f"сервис {backend.service.title} поддерживает {backend.service.max_qubits}."
            )

        qaoa = QuditQAOA(
            self.model,
            backend,
            layers=config.layers,
            optimizer=config.optimizer,
            restarts=config.restarts,
            max_evaluations=config.max_evaluations,
            objective=config.objective,
            cvar_alpha=config.cvar_alpha,
            seed=config.seed,
        )

        self._log(
            f"Запуск кудитного QAOA: слоёв p={config.layers}, "
            f"шотов {config.shots}, оптимизатор {config.optimizer}, "
            f"стартов {config.restarts}"
        )

        result = qaoa.optimize(logger=self.logger)
        self._log(
            "Параметры подобраны: "
            f"gamma={[round(v, 4) for v in result.gammas]}, "
            f"beta={[round(v, 4) for v in result.betas]}, "
            f"<C>={result.expectation:.4f}, вычислений {result.evaluations} ({result.method})"
        )

        counts = qaoa.sample(result, config.shots)
        self._log(f"Измерения выполнены: различных состояний {len(counts)}")

        decode = Decoder(self.model, self.objective).decode(counts)
        self._log(
            f"Декодирование: корректных выборок {decode.valid_fraction * 100:.1f}%, "
            f"допустимых {decode.feasible_fraction * 100:.1f}%"
        )

        greedy = None
        exhaustive = None
        if config.baselines:
            greedy = GreedySearch(self.objective).solve()
            exhaustive = ExhaustiveSearch(self.objective).solve()

        report = self._build_report(result, decode, validation, greedy, exhaustive, backend)

        return ExperimentOutcome(
            model=self.model,
            qaoa=result,
            decode=decode,
            validation=validation,
            greedy=greedy,
            exhaustive=exhaustive,
            counts=counts,
            backend_name=backend.service.key,
            report=report,
        )

    # ------------------------------------------------------------------

    def _validate_and_tune(self, max_attempts: int = 3) -> ModelValidation:
        """
        Проверяет модель и при необходимости усиливает штрафы.

        Если минимум приближённой модели не совпал с точным оптимумом задачи,
        веса штрафов увеличиваются в 4 раза и модель строится заново. Это
        избавляет пользователя от ручного подбора --lambda1/--lambda2.
        """

        validation = validate_model(self.model, self.objective)

        attempt = 0
        while validation.checked and not validation.consistent and attempt < max_attempts:
            attempt += 1

            base = self.model.penalty
            scale = 4.0 ** attempt
            reference = self.model.network_scale

            tuned = PenaltyConfig(
                mode=self.model.mode,
                weight=(base.weight or 2.0 * reference) * scale,
                lambda1=(base.lambda1 or reference) * scale,
                lambda2=(base.lambda2 or reference) * scale,
                max_slack_vars=base.max_slack_vars,
                max_state_space=base.max_state_space,
            )

            self._log(
                f"Минимум модели не совпал с оптимумом: повышаю веса штрафов "
                f"в {scale:.0f} раз (попытка {attempt}/{max_attempts})"
            )

            self.model = QUDOModel(self.problem, tuned)
            validation = validate_model(self.model, self.objective)

        return validation

    def _build_report(
        self,
        qaoa: QAOAResult,
        decode: DecodeReport,
        validation: ModelValidation,
        greedy: SolutionRecord | None,
        exhaustive: SolutionRecord | None,
        backend,
    ) -> dict[str, Any]:
        best = decode.best_feasible or decode.best_overall

        optimum = None
        if exhaustive is not None and exhaustive.placement:
            optimum = float(exhaustive.cost)

        def gap(cost: float | None) -> float | None:
            if cost is None or optimum is None or optimum == 0.0:
                return None
            return (float(cost) - optimum) / optimum * 100.0

        solution_report = self.objective.report(best.placement) if best else None

        report: dict[str, Any] = {
            "problem": self.problem.summary(),
            "model": self.model.describe(),
            "validation": validation.to_dict(),
            "backend": {
                "key": backend.service.key,
                "title": backend.service.title,
                "kind": backend.service.kind,
                "executed_locally": backend.executes_locally,
                "note": backend.service.note,
            },
            "qaoa": {
                "layers": self.config.layers,
                "shots": self.config.shots,
                "gamma": qaoa.gammas,
                "beta": qaoa.betas,
                "expectation": qaoa.expectation,
                "evaluations": qaoa.evaluations,
                "optimizer": qaoa.method,
                "restarts": qaoa.restarts,
                "objective": self.config.objective,
                "valid_fraction": decode.valid_fraction,
                "feasible_fraction": decode.feasible_fraction,
                "probability_of_solution": decode.probability_of(best.placement) if best else 0.0,
            },
            "solution": solution_report,
            "baselines": {},
            "gaps": {},
        }

        if greedy is not None:
            report["baselines"]["greedy"] = {
                "placement": greedy.placement,
                "cost": greedy.cost,
                "feasible": greedy.feasible,
            }
            report["gaps"]["greedy_percent"] = gap(greedy.cost)

        if exhaustive is not None:
            report["baselines"]["exhaustive"] = {
                "placement": exhaustive.placement,
                "cost": exhaustive.cost,
                "feasible": exhaustive.feasible,
                "explored_states": exhaustive.explored_states,
            }

        if best is not None:
            report["gaps"]["qaoa_percent"] = gap(best.cost)

        report["top_states"] = [
            {
                "bitstring": sample.bitstring,
                "counts": sample.counts,
                "placement": sample.placement,
                "cost": sample.cost,
                "feasible": sample.feasible,
            }
            for sample in decode.samples[:10]
        ]

        return report
