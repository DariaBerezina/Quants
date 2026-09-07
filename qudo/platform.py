"""
Интеграционный слой с «Облачной платформой квантовых вычислений» (ОПКВ).

Платформа запускает прикладной модуль как контейнер-лончер и показывает
пользователю:

    * «Логи последнего выполнения» — построчный текстовый вывод в stdout
      в формате ``[YYYY-MM-DD HH:MM:SS.mmm] сообщение``;
    * «Измерение вероятностей» — гистограмма, которая строится по словарю
      счётчиков вида ``{"0100": 125, "1010": 109, ...}``, выведенному
      отдельной строкой лога;
    * файлы результатов, записанные в каталог ``/data``.

Модуль повторяет этот контракт, чтобы вывод QUDO-лончера выглядел для
платформы так же, как вывод штатных приложений (grover, gkp, ideem и т.д.).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Mapping

# Рабочий каталог платформы: лончеры получают входные файлы из /data
# и туда же складывают результаты (см. параметры input/output штатных
# приложений в руководстве пользователя ОПКВ).
PLATFORM_DATA_DIR = os.environ.get("QUDO_DATA_DIR", "/data")


def resolve_path(path: str | None, default_name: str | None = None) -> str | None:
    """
    Приводит путь к соглашению платформы.

    Относительные пути раскрываются относительно ``/data`` (или значения
    переменной окружения ``QUDO_DATA_DIR``, что удобно при локальной отладке).
    """

    if path is None:
        if default_name is None:
            return None
        path = default_name

    if os.path.isabs(path):
        return path

    return os.path.join(PLATFORM_DATA_DIR, path)


def ensure_parent_directory(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


class PlatformLogger:
    """
    Логгер в формате логов ОПКВ.

    verbose:
        0 — только ключевые сообщения (запуск, результат, ошибки);
        1 — стандартный уровень (по умолчанию);
        2 — отладочный уровень (коэффициенты модели, шаги оптимизатора).
    """

    def __init__(self, verbose: int = 1, stream=None) -> None:
        self.verbose = int(verbose)
        self.stream = stream if stream is not None else sys.stdout
        self._start = time.perf_counter()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def log(self, message: str, level: int = 1) -> None:
        if level > self.verbose:
            return
        print(f"[{self._timestamp()}] {message}", file=self.stream, flush=True)

    def error(self, message: str) -> None:
        print(f"[{self._timestamp()}] ОШИБКА: {message}", file=self.stream, flush=True)

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def report_elapsed(self, prefix: str = "Локальное время расчёта составило") -> None:
        self.log(f"{prefix} {self.elapsed():.6f} с", level=0)

    def counts(self, counts: Mapping[str, int]) -> None:
        """
        Выводит словарь счётчиков одной строкой.

        Именно эта строка используется платформой для построения гистограммы
        распределения вероятностей на странице «Квантовые вычисления».
        """

        print(json.dumps(dict(counts), ensure_ascii=False), file=self.stream, flush=True)


def write_json(path: str, payload: Any) -> str:
    ensure_parent_directory(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: str, text: str) -> str:
    ensure_parent_directory(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path
