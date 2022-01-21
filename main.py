from __future__ import annotations

import argparse
import collections
import sys
from enum import Enum
from typing import Optional


class Color(Enum):
    WHITE = "🟥"
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


class Game:
    turns: list[tuple[str, ColorMask]]
    solution: Optional[str]

    _all_guesses: list[str]
    _all_solutions: list[str]
    _feasible_solutions: list[str]
    _turn_computed: int
    _turn_2_cache: dict[ColorMask, str]

    # Precomputed to be one of the best starter words according to the "smallest worst-case bin size" metric.
    # Equivalent are: serai, arise
    _turn_1_starter: str = "raise"

    def __init__(self, guesses: list[str], solutions: list[str], solution: str = None):
        self.turns = []
        self.solution = solution
        self._all_guesses = guesses
        self._all_solutions = solutions
        self._feasible_solutions = solutions
        self._turn_computed = 0
        self._turn_2_cache = {}

    def __str__(self):
        return "\n".join([f"{turn[0]} {turn[1]}" for turn in self.turns])

    def is_finished(self):
        return len(self.turns) == 6 or (self.turns and self.turns[-1][1] == ColorMask([GREEN] * 5))

    def _filter_feasible_solutions(self, turn_index: int = None) -> None:
        """Filter the internal list of feasible solutions based on the hints given in turn `turn_index`."""

        if turn_index is None:
            # If not specified, assume the last played turn.
            turn_index = len(self.turns) - 1

        if len(self.turns) < turn_index:
            # Cannot filter based on a turn that was not yet played!
            return

        solutions = self._feasible_solutions
        guess, hints = self.turns[turn_index]
        letters_seen = collections.defaultdict(lambda: 0)
        counts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        for letter, hint in zip(guess, hints):
            counts[letter][hint] += 1

        for index, hint in enumerate(hints):
            if hint == GREEN:
                solutions = [word for word in solutions if word[index] == guess[index]]
                letters_seen[guess[index]] += 1
            elif hint == WHITE:
                solutions = [
                    word for word in solutions
                    if (
                        # Amount of occurrences of the letter should be exactly equal to the amount of greens+yellows of that letter;
                        # the white hint denotes that no more occurrences can be present in the target word.
                            len([letter for letter in word if letter == guess[index]]) == counts[guess[index]][GREEN] + counts[guess[index]][YELLOW]
                            and word[index] != guess[index]
                    )
                ]
            elif hint == YELLOW:
                solutions = [
                    word for word in solutions
                    if (
                        # Amount of occurrences of the letter must be at least equal to the amount of green+yellows of that letter;
                        # the yellow hint does not exclude that there may be more occurrences in the target word.
                            len([letter for letter in word if letter == guess[index]]) >= counts[guess[index]][GREEN] + counts[guess[index]][YELLOW]
                            and word[index] != guess[index]
                    )
                ]

        self._feasible_solutions = solutions

    def get_color_mask(self, guess: str, solution: str = None) -> ColorMask:
        if solution is None:
            solution = self.solution

        colors = [WHITE] * 5
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
        return sorted(best_words.keys())

    def suggest_guess(self) -> str:
        if len(self._feasible_solutions) == 1:
            return self._feasible_solutions[0]

        if len(self.turns) == 0:
            # This is the first turn. Use pre-computed best words.
            return self._turn_1_starter

        if len(self.turns) == 5:
            # Desperado for a solution word
            return self._feasible_solutions[0]

        if len([hint for hint in self.turns[-1][1] if hint == GREEN]) == 4 and len(self._feasible_solutions) > (6 - len(self.turns)):
            # Only one letter is missing to get the solution, but we cannot simply try them all; we have too few turns to do this.
            # So we should spend one turn to try and eliminate several options at once.
            index_to_guess = 0
            for index, hint in enumerate(self.turns[-1][1]):
                if hint == WHITE.value:
                    index_to_guess = index
                    break

            possible_letters = [word[index_to_guess] for word in self._feasible_solutions]

            max_letter_count = 0
            best_guesses = {}
            for word in self._all_guesses:
                num_matches = len({letter for letter in word if letter in possible_letters})
                if num_matches == max_letter_count:
                    best_guesses[word] = 1
                elif num_matches > max_letter_count:
                    max_letter_count = num_matches
                    best_guesses = {word: 1}

            # Return a random guess out of the best guesses
            for guess in best_guesses:
                return guess

        if len(self.turns) == 2 and self.turns[0][0] == self._turn_1_starter:
            # We have a cached result for these hints after the starter word to play for turn 2, so use that.
            if self.turns[0][1] in self._turn_2_cache.keys():
                return self._turn_2_cache[self.turns[0][1]]

            # Compute the best word to use for turn 2 and cache it for when we play more games using the same starter word.
            guess = self.get_best_guesses()[0]
            self._turn_2_cache[self.turns[0][1]] = guess
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


def main(interactive: bool = False, solution: str = None):
    with open('wordle-words.txt', 'r') as f:
        _all_solutions = list({word.strip() for word in f})

    with open('wordle-fake-words.txt', 'r') as f:
        _all_guesses = list({word.strip() for word in f}.union(_all_solutions))

    if len(sys.argv) < 2:
        raise ValueError("No solution word provided")

    if solution not in _all_solutions:
        raise ValueError("Unrecognized solution word")

    game = Game(guesses=_all_guesses, solutions=_all_solutions, solution=solution)

    while not game.is_finished():
        while True:
            if interactive:
                guess = input("Guess: ")
                if guess == "":
                    print("Calculating suggested guess...")
                    guess = game.suggest_guess()
                    print(f"Suggested guess: {guess}")
                    break
                elif guess in _all_guesses:
                    break
            else:
                guess = game.suggest_guess()
                break

            print("Not a valid word, try again.")

        game.play_turn(guess=guess)
        print(game)
        print("\n")


def parse_args():
    supported_args = {
        "--solution": dict(short="-s", default=None, type=str),
    }
    supported_flags = {
        "--interactive": dict(short="-i", action="store_true"),
    }

    parser = argparse.ArgumentParser()
    for long_arg, info in supported_args.items():
        parser.add_argument(info["short"], long_arg, default=info["default"], type=info["type"])
    for long_arg, info in supported_flags.items():
        parser.add_argument(info["short"], long_arg, action=info["action"])

    args = parser.parse_args()
    return {
        key.strip("-"): getattr(args, key.strip("-")) for key in list(supported_args.keys()) + list(supported_flags.keys())
    }


if __name__ == "__main__":
    WHITE, GREEN, YELLOW = Color
    main(**parse_args())
