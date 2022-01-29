from __future__ import annotations

import argparse
import collections
from enum import Enum
from random import randint
from typing import Optional


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


class Game:
    turns: list[tuple[str, ColorMask]]
    solution: Optional[str]
    is_hard_mode: bool = False

    _all_guesses: list[str]
    _all_solutions: list[str]
    _feasible_solutions: list[str]
    _turn_computed: int

    TURN_2_CACHE: dict[ColorMask, str] = None
    MAX_TURNS = 6
    WORD_LENGTH = 5

    # Precomputed to be one of the best starter words according to the "smallest worst-case bin size" metric.
    # Equivalent are: serai, arise
    _turn_1_starter: str = "raise"

    def __init__(self, guesses: list[str], solutions: list[str], solution: str = None, hard: bool = False):
        self.turns = []
        self.solution = solution
        self._all_guesses = guesses
        self._all_solutions = solutions
        self._feasible_solutions = solutions
        self._turn_computed = 0
        self.is_hard_mode = hard

    def __str__(self):
        return "\n".join([f"{turn[0]} {turn[1]}" for turn in self.turns])

    def is_finished(self):
        return len(self.turns) == self.MAX_TURNS or self.is_won()

    def _filter_feasible_solutions(self, turn_index: int = None) -> None:
        """Filter the internal list of feasible solutions based on the hints given in turn `turn_index`."""
        if turn_index is None:
            # If not specified, assume the last played turn.
            turn_index = len(self.turns) - 1

        if len(self.turns) < turn_index:
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

    def get_color_mask(self, guess: str, solution: str = None) -> ColorMask:
        if solution is None:
            solution = self.solution

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

    def get_bins(self, guess: str, solutions: list[str]):
        bins = collections.defaultdict(lambda: 0)
        for solution in solutions:
            bins[self.get_color_mask(guess, solution)] += 1

        return bins

    def get_best_guesses(self):
        biggest_bin_per_guess: dict[str, tuple[str, int]] = {}

        for guess in self._all_guesses:
            bins = self.get_bins(guess, self._feasible_solutions)
            max_bin_size = max(bins.values())
            max_bins = [bin for bin in bins.keys() if bins[bin] == max_bin_size]
            biggest_bin_per_guess[guess] = ("|".join([str(bin) for bin in max_bins]), max_bin_size)

        smallest_bin = min([entry[1] for entry in biggest_bin_per_guess.values()])
        best_words = {guess: entry for guess, entry in biggest_bin_per_guess.items() if entry[1] == smallest_bin}

        # Some words may have multiple bins that are worst-case, while others do not.
        # So as a secondary metric to the "best worst-case scenario" metric, check for words that have the least such scenarios,
        # which are then more likely overall to not get into such a worst-case scenario.
        # We can improve this more generally by taking a certain percentile of all bins rather than always the worst,
        # but this requires sorting the bins, which takes a significant performance hit.
        fewest_scenarios = min([len(result[0].split("|")) for result in best_words.values()])
        best_guesses = sorted([key for key, result in best_words.items() if len(result[0].split("|")) == fewest_scenarios])

        # If we have multiple equivalent guesses, but some lie in the solution set and some don't, then prefer the ones in the solution set.
        # They are equivalent but maybe picking a solution guess leads to a lucky win!
        if solution_best_guesses := [guess for guess in best_guesses if guess in self._feasible_solutions]:
            return solution_best_guesses

        return best_guesses

    def suggest_guess(self) -> str:
        if len(self._feasible_solutions) == 1:
            return self._feasible_solutions[0]

        if len(self.turns) == 0:
            # This is the first turn. Use pre-computed best words.
            return self._turn_1_starter

        if len(self.turns) == 5:
            # Desperado for a solution word
            return self._feasible_solutions[0]

        if len(self.turns) == 1 and self.turns[0][0] == self._turn_1_starter:
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
        elif len(self.turns) == 6:
            print('Did not find a solution in time.')

    def is_won(self):
        return self.turns and self.turns[-1][1] == ColorMask([GREEN] * self.WORD_LENGTH)

    def play(self, guess_list: list[str], interactive: bool = False) -> Game:
        while not self.is_finished():
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


def main(interactive: bool = False, solution: str = None, full: bool = False, hard: bool = False):
    with open('wordle-words.txt', 'r') as f:
        _all_solutions = sorted(list({word.strip() for word in f}))

    with open('wordle-fake-words.txt', 'r') as f:
        _all_guesses = sorted(list({word.strip() for word in f}.union(_all_solutions)))

    Game.TURN_2_CACHE = {}

    if full:
        # Keep track of how many turns were needed for this game. Key "0" implies the game was not finished within the maximum allotted amount of turns.
        distribution = {i: 0 for i in range(Game.MAX_TURNS + 1)}
        for i, solution in enumerate(_all_solutions):
            print(f"Playing game {i} with solution {solution}...")
            game = Game(guesses=_all_guesses, solutions=_all_solutions, solution=solution, hard=hard).play(_all_guesses, interactive)

            if game.is_won():
                distribution[len(game.turns)] += 1
                print(game)
            else:
                distribution[0] += 1

        print(distribution)
    else:
        if not solution:
            solution = _all_solutions[randint(0, len(_all_solutions) - 1)]
        elif solution not in _all_solutions:
            raise ValueError("Unrecognized solution word")

        game = Game(guesses=_all_guesses, solutions=_all_solutions, solution=solution, hard=hard).play(_all_guesses, interactive)
        print(game)


def parse_args():
    supported_args = {
        "--solution": dict(short="-s", default=None, type=str, help="The solution word. If none provided, a random solution word will be chosen."),
    }
    supported_flags = {
        "--interactive": dict(short="-i", action="store_true", help="Interactive mode: allows the user to enter guesses. Leave a guess blank to let the program decide on a guess."),
        "--full": dict(short="-f", action="store_true", help="Perform a full run over all solution words. Useful for determining whether the engine can solve all games. Overrides -s and -i options."),
        "--hard": dict(short="-H", action="store_true", help="Play in 'hard mode': only guesses allowed that match all previous hints. Does not alter the solving metric.")
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
