from __future__ import annotations

import logging
import math

import blessings

terminal: blessings.Terminal = blessings.Terminal()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wordle")


def progress_bar(current: int, total: int) -> str:
    assert 0 <= current <= total
    return "|" + "■" * (80 * current // total) + " " * (80 - (80 * current // total)) + "| " + ("%.2f" % (100 * current / total)) + "%"


def turn_distribution_bars(distribution: dict[int, int]) -> str:
    results: list[str] = []
    biggest_bucket_size = max(distribution.values()) or 1

    for key in range(1, 7):
        results.append(str(key) + ":    |" + "■" * math.ceil(80 * distribution[key] / biggest_bucket_size) + " " + str(distribution[key]))

    results.append(terminal.red_bold + "Lost: |" + "■" * math.ceil(80 * distribution[0] / biggest_bucket_size) + " " + str(distribution[0]) + terminal.normal)
    results.append("Running average win: " + ("%.4f" % (sum([key * distribution[key] for key in range(1, 7)]) / sum([distribution[key] for key in range(1, 7)]))))

    return terminal.yellow + "\n".join(results) + terminal.normal
