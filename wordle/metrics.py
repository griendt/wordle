from __future__ import annotations

import math
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import field
from multiprocessing.pool import ApplyResult
from typing import Optional, final, Type

from wordle import cache
from wordle.buckets import get_buckets_many, get_buckets
from wordle.cli import logger
from wordle.color_mask import filter_feasible_words

ColorMask = int
available_metrics: dict[str, Type[Metric]] = {}


class Metric(ABC):
    # How many guesses need evaluation before we decide to spawn subprocesses for parallel computation.
    MINIMUM_SUBPROCESS_CHUNK_SIZE: int = 100
    # How many CPU cores may be used when evaluating. Defaults to all available.
    MAX_CORES: Optional[int] = None
    # Some metrics have different behaviour based on which turn it is.
    TURNS_PLAYED: int = 0
    # The optimal value the metric can return. If reached, no more searching is necessary.
    OPTIMAL_VALUE: float = -math.inf
    # Whether to stop searching for optimal guesses when the theoretical absolute optimum is found.
    STOP_SEARCH_ON_OPTIMAL_VALUE: bool = True

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

            if evaluation == self.OPTIMAL_VALUE and self.STOP_SEARCH_ON_OPTIMAL_VALUE:
                break

        return _optimal_guesses, _optimum

    @final
    def get_optimal_guesses(self, guesses: list[str], feasible_solutions: list[str], is_first_turn: bool, use_multiprocessing: bool = True) -> tuple[list[str], float]:
        logger.debug(f"Getting optimal guesses; guess list contains {len(guesses)} words and solutions contain {len(feasible_solutions)} words")

        previous_buckets_per_guess: Optional[dict[str, dict[ColorMask, int]]] = None
        if is_first_turn and cache.TURN_1_CACHE is not None and cache.TURN_1_CACHE_PREVIOUS_SOLUTION is not None:
            previous_solution, previous_buckets_per_guess = cache.TURN_1_CACHE_PREVIOUS_SOLUTION, cache.TURN_1_CACHE

            for guess, buckets in get_buckets_many(guesses, [previous_solution]).items():
                for bucket, occurrences in buckets.items():
                    previous_buckets_per_guess[guess][bucket] -= occurrences
                    if previous_buckets_per_guess[guess][bucket] == 0:
                        del previous_buckets_per_guess[guess][bucket]

        if len(guesses) <= Metric.MINIMUM_SUBPROCESS_CHUNK_SIZE or (Metric.MAX_CORES or 1) <= 1 or not use_multiprocessing:
            optimal_guesses, optimum = self.optimal_guesses_for_chunk(guesses, feasible_solutions)
            return sorted(optimal_guesses), optimum

        chunk_size = max(Metric.MINIMUM_SUBPROCESS_CHUNK_SIZE, len(guesses) // (Metric.MAX_CORES or multiprocessing.cpu_count()))
        chunks = [guesses[i: i + chunk_size] for i in range(0, len(guesses), chunk_size)]
        optimum, optimal_guesses = math.inf, []

        if is_first_turn and cache.TURN_1_CACHE is None:
            async_results: list[ApplyResult] = []
            pool = multiprocessing.Pool(processes=len(chunks))

            for chunk in chunks:
                async_results.append(pool.apply_async(func=self.optimal_guesses_for_chunk, args=(chunk, feasible_solutions)))
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

            return sorted(optimal_guesses), optimum

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
        return sorted(optimal_guesses), optimum


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


@register_metric
class Depth2Paranoid(Metric):
    ALL_GUESSES: list[str] = field(default_factory=list)
    OPTIMAL_VALUE = 1

    def evaluate(self, guess: str, feasible_solutions: list[str], buckets: dict[int, int] = None) -> float:
        if len(feasible_solutions) == 1 and guess in feasible_solutions:
            # If we can actually win in 1 turn, do that instead of trying a depth 2 search!
            return -math.inf

        logger.info(f"Trying {guess} with {len(feasible_solutions)} feasible solutions...")
        assert guess in self.ALL_GUESSES
        buckets = buckets or get_buckets(guess, feasible_solutions)

        best_evaluation = math.inf
        best_next_guesses = []
        for color_mask in buckets.keys():
            remaining_solutions = filter_feasible_words(feasible_solutions, guess, color_mask)
            optimal_next_guesses, evaluation = Paranoid().get_optimal_guesses(self.ALL_GUESSES, remaining_solutions, is_first_turn=False, use_multiprocessing=False)

            if evaluation < best_evaluation:
                best_evaluation = evaluation
                best_next_guesses = optimal_next_guesses

            if evaluation == 1:
                # We have a guaranteed win in 2. However, we obviously prefer a win in 1 over a win in 2.
                break

        if best_evaluation == 1:
            # We have a guaranteed win in 3. As a secondary metric, prefer words that will guarantee a win in 2 instead, by using the regular Paranoid evaluation.
            best_evaluation = Paranoid().evaluate(guess, feasible_solutions, buckets)
        else:
            # Add the size of all guesses to the evaluation, so that all words that do not have a guaranteed win in 2 score worse than words that have a guaranteed win in 2.
            best_evaluation = len(self.ALL_GUESSES) + best_evaluation

        logger.info(f"Best evaluation for guess {guess} is {best_evaluation} with words: {best_next_guesses[:5]}")
        return best_evaluation
