from __future__ import annotations

import argparse
import collections
import math
from abc import abstractmethod, ABC
from enum import Enum
from inspect import isclass
from random import randint
from typing import Optional, final, Type


class Color(Enum):
    RED = "🟥"
    GREEN = "🟩"
    YELLOW = "🟨"


class ColorMask:
    colors: list[Color]

    def __init__(self, colors: list[Color]):
        self.colors = colors

    def __str__(self):
        return "".join([color.value for color in self.colors])

    def __iter__(self):
        for color in self.colors:
            yield color

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return str(self)


class Metric(ABC):
    @abstractmethod
    def evaluate(self, guess: str, feasible_solutions: list[str]) -> float:
        raise NotImplementedError

    @final
    def get_optimal_guesses(self, guesses: list[str], feasible_solutions: list[str]) -> list[str]:
        optimum, optimal_guesses = math.inf, []
        for guess in guesses:
            evaluation = self.evaluate(guess, feasible_solutions)
            if evaluation > optimum:
                continue

            if evaluation < optimum:
                optimum = evaluation
                optimal_guesses = [guess]
            elif evaluation == optimum:
                optimal_guesses.append(guess)

        return sorted(optimal_guesses)


class Paranoid(Metric):
    def evaluate(self, guess: str, feasible_solutions: list[str]) -> float:
        return max(Game.get_bins(guess, solutions=feasible_solutions).values())


class Pattern(Metric):
    def evaluate(self, guess: str, feasible_solutions: list[str]) -> float:
        return -len(Game.get_bins(guess, solutions=feasible_solutions))


class Game:
    turns: list[tuple[str, ColorMask]]
    solution: Optional[str]
    is_hard_mode: bool = False
    metric: Type[Metric] = Paranoid

    _all_guesses: list[str]
    _all_solutions: list[str]
    _feasible_solutions: list[str]
    _turn_computed: int

    # The word to use for turn 1.
    TURN_1_GUESS: str = "raise"
    TURN_2_CACHE: dict[ColorMask, str] = None
    MAX_TURNS = 6
    WORD_LENGTH = 5

    def __init__(self, guesses: list[str], solutions: list[str], solution: str = None, hard: bool = False, metric: Type[Metric] = Paranoid):
        self.turns = []
        self.solution = solution
        self._all_guesses = guesses
        self._all_solutions = solutions
        self._feasible_solutions = solutions
        self._turn_computed = 0
        self.is_hard_mode = hard
        self.metric = metric

    def __str__(self):
        return "\n".join([f"{turn[0]} {turn[1]}" for turn in self.turns])

    @property
    def num_turns(self):
        return len(self.turns)

    @property
    def is_finished(self):
        return self.num_turns == self.MAX_TURNS or self.is_won

    @property
    def is_won(self):
        return self.turns and self.turns[-1][1] == ColorMask([GREEN] * self.WORD_LENGTH)

    @staticmethod
    def _filter_words_based_on_hint(words: list[str], guess: str, letter_index: int, hint: Color, counts: dict[(str, Color), int]) -> list[str]:
        if hint == GREEN:
            return [word for word in words if word[letter_index] == guess[letter_index]]
        if hint == RED:
            return [word for word in words if (
                # Amount of occurrences of the letter should be exactly equal to the amount of greens+yellows of that letter;
                # the white hint denotes that no more occurrences can be present in the target word.
                len([letter for letter in word if letter == guess[letter_index]]) == counts[(guess[letter_index], GREEN)] + counts[(guess[letter_index], YELLOW)]
                and word[letter_index] != guess[letter_index]
            )]
        if hint == YELLOW:
            return [word for word in words if (
                # Amount of occurrences of the letter must be at least equal to the amount of green+yellows of that letter;
                # the yellow hint does not exclude that there may be more occurrences in the target word.
                len([letter for letter in word if letter == guess[letter_index]]) >= counts[(guess[letter_index], GREEN)] + counts[(guess[letter_index], YELLOW)]
                and word[letter_index] != guess[letter_index]
                )
            ]

        raise ValueError(f"Unexpected hint: {hint}")

    @staticmethod
    def get_color_mask(guess: str, solution: str) -> ColorMask:
        colors = [RED] * 5
        counts_per_letter = collections.Counter(solution)

        for i in range(5):
            if solution[i] == guess[i]:
                colors[i] = GREEN
                counts_per_letter[guess[i]] -= 1

        for i in range(5):
            if solution[i] != guess[i] and guess[i] in solution and counts_per_letter[guess[i]] > 0:
                counts_per_letter[guess[i]] -= 1
                colors[i] = YELLOW

        return ColorMask(colors)

    @staticmethod
    def get_bins(guess: str, solutions: list[str]):
        bins = collections.defaultdict(lambda: 0)
        for solution in solutions:
            bins[Game.get_color_mask(guess, solution)] += 1

        return bins

    def _filter_feasible_solutions(self, turn_index: int = None) -> None:
        """Filter the internal list of feasible solutions based on the hints given in turn `turn_index`."""
        if turn_index is None:
            # If not specified, assume the last played turn.
            turn_index = self.num_turns - 1

        if self.num_turns < turn_index:
            # Cannot filter based on a turn that was not yet played!
            return

        guess, hints = self.turns[turn_index]

        counts: dict[(str, Color), int] = collections.defaultdict(lambda: 0)
        for letter, hint in zip(guess, hints):
            counts[(letter, hint)] += 1

        for index, hint in enumerate(hints):
            self._feasible_solutions = self._filter_words_based_on_hint(self._feasible_solutions, guess, index, hint, counts)
            if self.is_hard_mode:
                self._all_guesses = self._filter_words_based_on_hint(self._all_guesses, guess, index, hint, counts)

    def get_best_guesses(self):
        best_guesses = self.metric().get_optimal_guesses(self._all_guesses, self._feasible_solutions)

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

        if self.num_turns == 0:
            # This is the first turn. Use pre-computed best words.
            return self.TURN_1_GUESS

        if self.num_turns == 1 and self.turns[0][0] == self.TURN_1_GUESS:
            # We have a cached result for these hints after the starter word to play for turn 2, so use that.
            if self.turns[0][1] in Game.TURN_2_CACHE.keys():
                print("Using cache")
                return Game.TURN_2_CACHE[self.turns[0][1]]

            # Compute the best word to use for turn 2 and cache it for when we play more games using the same starter word.
            guess = self.get_best_guesses()[0]
            Game.TURN_2_CACHE[self.turns[0][1]] = guess
            return guess

        return self.get_best_guesses()[0]

    def play_turn(self, guess: str):
        assert self.solution is not None, "Cannot process turn without knowing the solution"
        hints = list(self.get_bins(guess, [self.solution]).keys())[0]
        self.turns.append((guess, hints))
        self._filter_feasible_solutions()

        if guess == self.solution:
            print('Game solved!')
        elif self.num_turns == 6:
            print('Did not find a solution in time.')

    def play(self, guess_list: list[str], interactive: bool = False) -> Game:
        while not self.is_finished:
            while True:
                if interactive:
                    guess = input("Guess: ")
                    if guess == "":
                        print("Calculating suggested guess...")
                        guess = self.suggest_guess()
                        print(f"Suggested guess: {guess}")
                        break
                    elif guess in guess_list:
                        break
                else:
                    guess = self.suggest_guess()
                    break

                print("Not a valid word, try again.")

            self.play_turn(guess=guess)
            if interactive:
                print(self)
                print(self._feasible_solutions)
                print("\n")

        return self


def main(interactive: bool = False, solution: str = None, full: bool = False, hard: bool = False, starter: str = None, metric: str = None):
    metric = globals()[metric.title()]
    assert issubclass(metric, Metric)

    with open('wordle-words.txt', 'r') as f:
        _all_solutions = sorted(list({word.strip() for word in f}))

    with open('wordle-fake-words.txt', 'r') as f:
        _all_guesses = sorted(list({word.strip() for word in f}.union(_all_solutions)))

    if starter is not None:
        if starter not in _all_guesses:
            raise ValueError("Unrecognized starter word")
        Game.TURN_1_GUESS = starter

    Game.TURN_2_CACHE = {}
    failed_words: list[str] = []
    game_options = {
        "guesses": _all_guesses,
        "solutions": _all_solutions,
        "solution": solution,
        "hard": hard,
        "metric": metric,
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
        for i, solution in enumerate(_all_solutions):
            print(f"Playing game {i} with solution {solution}...")
            game_options["solution"] = solution
            game = Game(**game_options).play(_all_guesses, interactive)

            if game.is_won:
                distribution[len(game.turns)] += 1
                print(game)
            else:
                distribution["Failed"] += 1
                failed_words.append(solution)

        print(distribution)
        print("Average turns per win: " + str(sum([key*value for key, value in distribution.items() if isinstance(key, int)]) / sum([value for value in distribution.values()])))

        if failed_words:
            print(f"Failed words: {failed_words}")


def parse_args():
    metrics = [name.lower() for name, cls in globals().items() if isclass(cls) and issubclass(cls, Metric) and cls != Metric]
    supported_args = {
        "--solution": dict(short="-s", default=None, type=str, help="The solution word. If none provided, a random solution word will be chosen."),
        "--starter": dict(short="-S", default=Game.TURN_1_GUESS, type=str, help="Specify a starter word."),
        "--metric": dict(short="-m", default="paranoid", type=str, help=f"Specify a metric to use for solving game. Supported values are: {', '.join(metrics)}"),
    }
    supported_flags = {
        "--interactive": dict(short="-i", action="store_true", help="Interactive mode: allows the user to enter guesses. Leave a guess blank to let the program decide on a guess."),
        "--full": dict(short="-f", action="store_true", help="Perform a full run over all solution words. Useful for determining whether the engine can solve all games. Overrides -s and -i options."),
        "--hard": dict(short="-H", action="store_true", help="Play in 'hard mode': only guesses allowed that match all previous hints. Does not alter the solving metric."),
    }

    parser = argparse.ArgumentParser()
    for long_arg, info in supported_args.items():
        parser.add_argument(info["short"], long_arg, default=info["default"], type=info["type"], help=info["help"])
    for long_arg, info in supported_flags.items():
        parser.add_argument(info["short"], long_arg, action=info["action"], help=info["help"])

    args = parser.parse_args()
    return {
        key.strip("-"): getattr(args, key.strip("-")) for key in list(supported_args.keys()) + list(supported_flags.keys())
    }


if __name__ == "__main__":
    RED, GREEN, YELLOW = Color
    main(**parse_args())
