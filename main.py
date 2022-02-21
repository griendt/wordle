#!/usr/bin/env python

from __future__ import annotations

import argparse
import collections
import copy
import logging
import multiprocessing
import time
from random import randint
from typing import Any

# Define ColorMask type for clearer type hinting
import cache
import metrics
from cli import terminal, logger, progress_bar, turn_distribution_bars
from game import Game, Color
from metrics import available_metrics

ColorMask = int

# Escape sequence to clear the terminal screen
clear: str = "\033c"


def main(metric: str, interactive: bool = False, solution: str = None, full: bool = False, hard: bool = False, starter: str = None, **kwargs):
    start_time = time.time()

    if kwargs.get("min_subprocess_chunk"):
        metrics.Metric.MINIMUM_SUBPROCESS_CHUNK_SIZE = kwargs["min_subprocess_chunk"]
    if kwargs.get("max_cpus"):
        metrics.Metric.MAX_CORES = kwargs["max_cpus"]
    if kwargs.get("log_level"):
        logger.setLevel(kwargs["log_level"])
    if kwargs.get("metric_entropy_percentile") is not None:
        metrics.PercentileEntropy.PERCENTILE = kwargs["metric_entropy_percentile"]

    with open('wordle-words.txt', 'r') as f:
        _all_solutions = [word.strip() for word in f]
        num_solutions = len(_all_solutions)

        if not kwargs.get("full_truncate_solutions"):
            # Sort the list to avoid being spoiled, but keep the list sorted in case we apply the full-truncate-solutions optimization.
            _all_solutions = sorted(_all_solutions)

    if kwargs.get("only-solution-set"):
        _all_guesses = sorted(list(_all_solutions))
    else:
        with open('wordle-fake-words.txt', 'r') as f:
            _all_guesses = sorted(list({word.strip() for word in f}.union(_all_solutions)))

    if starter is not None:
        if starter not in _all_guesses:
            raise ValueError("Unrecognized starter word")
        cache.TURN_1_GUESS = starter

    cache.TURN_2_CACHE = {}
    cache.BUCKETS_CACHE = {}
    cache.COUNTERS_PER_SOLUTION = {solution: collections.Counter(solution) for solution in _all_solutions}

    failed_words: list[str] = []
    game_options: dict[str, Any] = {
        "guesses": _all_guesses,
        "solutions": _all_solutions,
        "solution": solution,
        "hard": hard,
        "metric": available_metrics[metric],
    }

    if not full:
        if not solution:
            game_options["solution"] = _all_solutions[randint(0, len(_all_solutions) - 1)]
        elif solution not in _all_solutions:
            raise ValueError("Unrecognized solution word")

        print(Game(**game_options).play(_all_guesses, interactive))
    else:
        # Keep track of how many turns were needed for this game. Key "0" implies the game was not finished within the maximum allotted amount of turns.
        distribution = {i: 0 for i in range(Game.MAX_TURNS + 1)}

        for i, solution in enumerate(copy.deepcopy(_all_solutions)):
            metrics.Metric.TURNS_PLAYED = 0
            game_options["solution"] = solution
            game = Game(**game_options).play(_all_guesses, interactive)

            if game.is_won:
                distribution[len(game.turns)] += 1
                logger.info(game)
            else:
                distribution[0] += 1
                failed_words.append(solution)

            if kwargs.get("full_truncate_solutions"):
                cache.TURN_1_CACHE_PREVIOUS_SOLUTION = solution

            if not kwargs.get("silent") or i % 100 == 0:
                print(f"{terminal.clear}Played game {terminal.yellow}{i}{terminal.normal} with solution {terminal.bold_green}{solution}{terminal.normal}")
                print(game)
                print(progress_bar(i, num_solutions))
                print(turn_distribution_bars(distribution))

            if kwargs.get("full_truncate_solutions"):
                _all_solutions.remove(solution)

        print("Average turns per win: " + str(
            sum([key * value for key, value in distribution.items() if isinstance(key, int)]) / sum([value for value in distribution.values()])))

        if failed_words:
            print(f"Failed words: {failed_words}")

    logger.info(f"Finished in {time.time() - start_time}")


def parse_args():
    supported_args = {
        "--full": {"short": "-f", "action": "store_true", "help": "Perform a full run over all solution words. Useful for determining whether the engine can solve all games. Overrides -s and -i options."},
        "--hard": {"short": "-H", "action": "store_true", "help": "Play in 'hard mode': only guesses allowed that match all previous hints. Does not alter the solving metric."},
        "--interactive": {"short": "-i", "action": "store_true", "help": "Interactive mode: allows the user to enter guesses. Leave a guess blank to let the program decide on a guess."},
        "--metric": {"default": "Paranoid", "type": str, "help": f"Specify a metric to use for solving the game. Supported values are: {', '.join(available_metrics.keys())}"},
        "--solution": {"short": "-s", "default": None, "type": str, "help": "The solution word. If none provided, a random solution word will be chosen."},
        "--starter": {"short": "-S", "default": cache.TURN_1_GUESS, "type": str, "help": "Specify a starter word."},
        "--min-subprocess-chunk": {"type": int, "help": "Minimum chunk size for parallel multiprocessing of possible guesses.", "default": metrics.Metric.MINIMUM_SUBPROCESS_CHUNK_SIZE},
        "--max-cpus": {"type": int, "help": "Maximum amount of CPUs that may be used. Defaults to all available.", "default": multiprocessing.cpu_count()},
        "--log-level": {"type": str, "help": "Set the log level. Defaults to INFO.", "default": logging.INFO},
        "--full-truncate-solutions": {"action": "store_true", "help": "If set, and a full run is being done, words that are already seen as solutions will be truncated from the solution space in subsequent games."},
        "--only-solution-set": {"action": "store_true", "help": "If set, only words in the solution set will be played at all times."},
        "--metric-entropy-percentile": {"type": float, "default": None, "help": "The percentile (0-100) to use when using the PercentileEntropy metric. Note that 100th percentile is effectively equivalent to the Paranoid metric."},
        "--silent": {"action": "store_true", "help": "Suppress output during full runs, giving output only once every 100 games."},
    }

    parser = argparse.ArgumentParser()
    for long_arg, info in supported_args.items():
        parser.add_argument(
            *[dash_arg for dash_arg in [long_arg, info.get("short")] if dash_arg is not None],
            **{key: value for key, value in info.items() if key in ["default", "type", "help", "action"]}
        )

    args = parser.parse_args()
    return {key.strip("-").replace("-", "_"): getattr(args, key.strip("-").replace("-", "_")) for key in list(supported_args.keys())}


if __name__ == "__main__":
    RED, GREEN, YELLOW = Color.RED, Color.GREEN, Color.YELLOW
    main(**parse_args())
