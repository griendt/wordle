from __future__ import annotations

from typing import Optional, Type

from wordle import cache
from wordle import metrics
from wordle.buckets import get_buckets
from wordle.cli import terminal, logger
from wordle.color_mask import color_mask_visual, filter_feasible_words

ColorMask = int


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

    def get_best_guesses(self):
        best_guesses, optimum = self.metric().get_optimal_guesses(self._all_guesses, self._feasible_solutions, len(self.turns) == 0)

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
        color_mask: ColorMask = list(get_buckets(guess, [self.solution]).keys())[0]
        self.turns.append((guess, color_mask))
        self._feasible_solutions = filter_feasible_words(self._feasible_solutions, guess, color_mask)

        if self.is_hard_mode:
            self._all_guesses = filter_feasible_words(self._all_guesses, guess, color_mask)

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
