from __future__ import annotations

import math
import multiprocessing
from abc import ABC, abstractmethod
from multiprocessing.pool import ApplyResult
from typing import Optional, final, Type

import cache
from buckets import get_buckets_many, get_buckets
from cli import logger

ColorMask = int
available_metrics: dict[str, Type[Metric]] = {}


class Metric(ABC):
    # How many guesses need evaluation before we decide to spawn subprocesses for parallel computation.
    MINIMUM_SUBPROCESS_CHUNK_SIZE: int = 100
    # How many CPU cores may be used when evaluating. Defaults to all available.
    MAX_CORES: Optional[int] = None
    # Some metrics have different behaviour based on which turn it is.
    TURNS_PLAYED: int = 0

    @abstractmethod
    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        raise NotImplementedError

    @final
    def __str__(self):
        return type(self).__name__

    @final
    def optimal_guesses_for_chunk(self, guess_chunk: list[str], feasible_solutions: list[str]) -> tuple[list[str], float]:
        _optimum, _optimal_guesses = math.inf, []
        for guess in guess_chunk:
            evaluation = self.evaluate(guess, feasible_solutions)
            if evaluation > _optimum:
                continue

            if evaluation < _optimum:
                _optimum = evaluation
                _optimal_guesses = [guess]
            elif evaluation == _optimum:
                _optimal_guesses.append(guess)

        return _optimal_guesses, _optimum

    @final
    def get_optimal_guesses(self, guesses: list[str], feasible_solutions: list[str], is_first_turn: bool) -> list[str]:
        logger.debug(f"Getting optimal guesses; guess list contains {len(guesses)} words and solutions contain {len(feasible_solutions)} words")

        previous_buckets_per_guess: Optional[dict[str, dict[ColorMask, int]]] = None
        if is_first_turn and cache.TURN_1_CACHE is not None and cache.TURN_1_CACHE_PREVIOUS_SOLUTION is not None:
            previous_solution, previous_buckets_per_guess = cache.TURN_1_CACHE_PREVIOUS_SOLUTION, cache.TURN_1_CACHE

            for guess, buckets in get_buckets_many(guesses, [previous_solution]).items():
                for bucket, occurrences in buckets.items():
                    previous_buckets_per_guess[guess][bucket] -= occurrences
                    if previous_buckets_per_guess[guess][bucket] == 0:
                        del previous_buckets_per_guess[guess][bucket]

        if len(guesses) <= Metric.MINIMUM_SUBPROCESS_CHUNK_SIZE or (Metric.MAX_CORES or 1) <= 1:
            optimal_guesses, optimum = self.optimal_guesses_for_chunk(guesses, feasible_solutions)
            return sorted(optimal_guesses)

        chunk_size = max(Metric.MINIMUM_SUBPROCESS_CHUNK_SIZE, len(guesses) // (Metric.MAX_CORES or multiprocessing.cpu_count()))
        chunks = [guesses[i: i + chunk_size] for i in range(0, len(guesses), chunk_size)]
        optimum, optimal_guesses = math.inf, []

        if is_first_turn and cache.TURN_1_CACHE is None:
            async_results: list[ApplyResult] = []
            pool = multiprocessing.Pool(processes=len(chunks))

            for chunk in chunks:
                async_results.append(pool.apply_async(args=(chunk, feasible_solutions)))
            pool.close()
            pool.join()

            # Merge all the buckets together
            awaited_results: list[dict[str, dict[int, int]]] = [result.get() for result in async_results]
            buckets_per_guess: dict[str, dict[int, int]] = {}
            for result in awaited_results:
                buckets_per_guess.update(result)

            cache.TURN_1_CACHE = buckets_per_guess
            cache.TURN_1_CACHE_PREVIOUS_SOLUTION = None

        if is_first_turn and cache.TURN_1_CACHE is not None and cache.TURN_1_CACHE_PREVIOUS_SOLUTION is not None:
            assert previous_buckets_per_guess is not None
            buckets_per_guess = previous_buckets_per_guess

            for (guess, buckets) in buckets_per_guess.items():
                value = self.evaluate(guess, feasible_solutions, buckets)
                if value < optimum:
                    optimum = value
                    optimal_guesses = [guess]
                elif value == optimum:
                    optimal_guesses.append(guess)

            return sorted(optimal_guesses)

        if len(chunks) > 1:
            async_results: list[ApplyResult] = []
            pool = multiprocessing.Pool(processes=len(chunks))
            for chunk in chunks:
                async_results.append(pool.apply_async(func=self.optimal_guesses_for_chunk, args=(chunk, feasible_solutions)))
            pool.close()
            pool.join()

            # Merge all the buckets together
            results = [result.get() for result in async_results]
        else:
            results = [self.optimal_guesses_for_chunk(chunks[0], feasible_solutions)]

        for (guesses, local_optimum) in results:
            if local_optimum < optimum:
                optimum = local_optimum
                optimal_guesses = guesses
            elif local_optimum == optimum:
                optimal_guesses += guesses

        logger.debug(f"Done getting optimal guesses")
        return sorted(optimal_guesses)


def register_metric(cls):
    assert issubclass(cls, Metric)
    available_metrics[str(cls())] = cls
    return cls


@register_metric
class PatternThenParanoid(Metric):
    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        if Metric.TURNS_PLAYED <= 1:
            return Pattern().evaluate(guess, feasible_solutions, buckets)

        return Paranoid().evaluate(guess, feasible_solutions, buckets)


@register_metric
class Paranoid(Metric):
    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        buckets = buckets if buckets else get_buckets(guess, feasible_solutions)
        return max(buckets.values())


@register_metric
class Pattern(Metric):
    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        buckets = buckets if buckets else get_buckets(guess, feasible_solutions)
        return -len(buckets)


@register_metric
class Deviation(Metric):
    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        buckets = buckets if buckets else get_buckets(guess, feasible_solutions)
        values = buckets.values()

        if len(values) == 1:
            # Only one possible bucket: this suggestion provides no information at all.
            return math.inf

        average_population = sum(values) / len(values)

        # Since square root is monotone, we do not need to compute it in order to compare guesses with one another.
        return sum([(value - average_population) ** 2 for value in values]) / (len(values) - 1)


@register_metric
class AverageEntropy(Metric):
    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        buckets = buckets if buckets else get_buckets(guess, feasible_solutions)
        values = buckets.values()
        information_bits = [-math.log(value / len(feasible_solutions), 2) for value in values]

        return -sum(information_bits) / len(information_bits)


@register_metric
class PercentileEntropy(Metric):
    PERCENTILE: float = 0

    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        assert 0 < self.PERCENTILE <= 100
        buckets = buckets if buckets else get_buckets(guess, feasible_solutions)
        values = buckets.values()
        information_bits = sorted([-math.log(value / len(feasible_solutions), 2) for value in values], reverse=True)

        return -information_bits[math.ceil(len(information_bits) * (self.PERCENTILE / 100)) - 1]


@register_metric
class ParanoidThenPattern(Metric):
    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        if Metric.TURNS_PLAYED <= 1:
            return Paranoid().evaluate(guess, feasible_solutions, buckets)

        return Pattern().evaluate(guess, feasible_solutions, buckets)


