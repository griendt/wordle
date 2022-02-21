from __future__ import annotations

import collections
from enum import Enum
from typing import Optional, Type

import cache
import metrics
from cli import terminal, logger
from buckets import get_buckets

ColorMask = int


def color_mask_visual(color_mask: ColorMask) -> str:
    colors = ""
    for i in range(5):
        if color_mask & (1 << i):
            colors += Color.GREEN.value
        elif color_mask & (1 << (i + 5)):
            colors += Color.YELLOW.value
        else:
            colors += Color.RED.value

    return colors


class Color(Enum):
    RED = "🟥"
    GREEN = "🟩"
    YELLOW = "🟨"


class Game:
    turns: list[tuple[str, ColorMask]]
    solution: Optional[str]
    is_hard_mode: bool = False
    metric: Type[metrics.Metric] = metrics.Paranoid

    _all_guesses: list[str]
    _all_solutions: list[str]
    _feasible_solutions: list[str]
    _turn_computed: int

    MAX_TURNS = 6
    WORD_LENGTH = 5

    def __init__(self, guesses: list[str], solutions: list[str], solution: str = None, hard: bool = False, metric: Type[metrics.Metric] = metrics.Paranoid):
        self.turns = []
        self.solution = solution
        self._all_guesses = guesses
        self._all_solutions = solutions
        self._feasible_solutions = solutions
        self._turn_computed = 0
        self.is_hard_mode = hard
        self.metric = metric

        cache.COUNTERS_PER_SOLUTION = {solution: collections.Counter(solution) for solution in solutions}

    def __str__(self):
        return "\n".join([f"{turn[0]} {color_mask_visual(turn[1])}" for turn in self.turns]) + "\n" * (self.MAX_TURNS - len(self.turns)) + f"\nGuess space: {len(self._all_guesses)}, solution space: {len(self._all_solutions)}"

    @property
    def num_turns(self):
        return len(self.turns)

    @property
    def is_finished(self):
        return self.num_turns == self.MAX_TURNS or self.is_won

    @property
    def is_won(self):
        return self.turns and self.turns[-1][1] == 2 ** self.WORD_LENGTH - 1

    @staticmethod
    def _filter_words_based_on_color_mask(words: list[str], guess: str, color_mask: ColorMask, counts: dict[tuple[str, Color], int]) -> list[str]:
        for i in range(5):
            if color_mask & (1 << i):
                # This index is marked green
                words = [word for word in words if word[i] == guess[i]]
            elif color_mask & (1 << (i + 5)):
                # This index is marked yellow
                words = [word for word in words if (
                    # Amount of occurrences of the letter must be at least equal to the amount of green+yellows of that letter;
                    # the yellow hint does not exclude that there may be more occurrences in the target word.
                        len([letter for letter in word if letter == guess[i]]) >= counts[(guess[i], Color.GREEN)] + counts[(guess[i], Color.YELLOW)]
                        and word[i] != guess[i]
                )]
            else:
                # This index is marked gray
                words = [word for word in words if (
                    # Amount of occurrences of the letter should be exactly equal to the amount of greens+yellows of that letter;
                    # the white hint denotes that no more occurrences can be present in the target word.
                        len([letter for letter in word if letter == guess[i]]) == counts[(guess[i], Color.GREEN)] + counts[(guess[i], Color.YELLOW)]
                        and word[i] != guess[i]
                )]

        return words

    def _filter_feasible_solutions(self) -> None:
        """Filter the internal list of feasible solutions based on the hints given in the last played turn."""
        if not self.turns:
            return

        guess, color_mask = self.turns[-1]
        counts: dict[tuple[str, Color], int] = collections.defaultdict(lambda: 0)

        for index, letter in enumerate(guess):
            if color_mask & (1 << index) != 0:
                counts[(letter, Color.GREEN)] += 1
            elif color_mask & (1 << (index + self.WORD_LENGTH)) != 0:
                counts[(letter, Color.YELLOW)] += 1

        self._feasible_solutions = self._filter_words_based_on_color_mask(self._feasible_solutions, guess, color_mask, counts)
        if self.is_hard_mode:
            self._all_guesses = self._filter_words_based_on_color_mask(self._all_guesses, guess, color_mask, counts)

    def get_best_guesses(self):
        best_guesses = self.metric().get_optimal_guesses(self._all_guesses, self._feasible_solutions, len(self.turns) == 0)

        # If we have multiple equivalent guesses, but some lie in the solution set and some don't, then prefer the ones in the solution set.
        # They are equivalent but maybe picking a solution guess leads to a lucky win!
        if solution_best_guesses := [guess for guess in best_guesses if guess in self._feasible_solutions]:
            return solution_best_guesses

        return best_guesses

    def suggest_guess(self) -> str:
        if len(self._feasible_solutions) == 1:
            return self._feasible_solutions[0]

        if self.is_hard_mode and len(self._feasible_solutions) <= self.MAX_TURNS - self.num_turns:
            # There are fewer or equal amount of solutions left as we have turns.
            # In hard mode, this means we have a guaranteed win; in normal mode, we may decide to play a non-solution word
            # instead to get to a win faster. But in hard mode this may be a too risky operation; so, for hard mode,
            # we will simply enumerate the solution words to get a guaranteed win, even if that is not optimal for the amount
            # of turns needed to get to that win.
            return self._feasible_solutions[0]

        if self.num_turns == 5:
            # Desperado for a solution word
            return self._feasible_solutions[0]

        if self.num_turns == 0 and cache.TURN_1_GUESS is not None:
            # This is the first turn. Use pre-computed best words if available.
            return cache.TURN_1_GUESS

        if self.num_turns == 1 and self.turns[0][0] == cache.TURN_1_GUESS:
            # We have a cached result for these hints after the starter word to play for turn 2, so use that.
            if self.turns[0][1] in cache.TURN_2_CACHE.keys():
                logger.info("Using cache")
                return cache.TURN_2_CACHE[self.turns[0][1]]

            # Compute the best word to use for turn 2 and cache it for when we play more games using the same starter word.
            guess = self.get_best_guesses()[0]
            cache.TURN_2_CACHE[self.turns[0][1]] = guess
            return guess

        return self.get_best_guesses()[0]

    def play_turn(self, guess: str):
        assert self.solution is not None, "Cannot process turn without knowing the solution"
        hints = list(get_buckets(guess, [self.solution]).keys())[0]
        self.turns.append((guess, hints))
        self._filter_feasible_solutions()

        metrics.Metric.TURNS_PLAYED += 1

        if guess == self.solution:
            logger.info('Game solved!')
        elif self.num_turns == 6:
            logger.warning('Did not find a solution in time.')

    def play(self, guess_list: list[str], interactive: bool = False) -> Game:
        while not self.is_finished:
            while True:
                if interactive:
                    guess = input("Guess: ")
                    if guess == "":
                        logger.debug("Calculating suggested guess...")
                        guess = self.suggest_guess()
                        logger.debug(f"Suggested guess: {guess}")
                        break
                    elif guess in guess_list:
                        break
                else:
                    guess = self.suggest_guess()
                    break

                print("Not a valid word, try again.")

            self.play_turn(guess=guess)
            if interactive and not self.is_finished:
                print(terminal.clear + str(self) + "\n")

        return self
