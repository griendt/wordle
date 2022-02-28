from __future__ import annotations

import collections
from enum import Enum


ColorMask = int


class Color(Enum):
    RED = "🟥"
    GREEN = "🟩"
    YELLOW = "🟨"


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


def filter_feasible_words(words: list[str], guess: str, color_mask: ColorMask) -> list[str]:
    """Filter the internal list of feasible solutions based on the hints given in the last played turn."""
    counts: dict[tuple[str, Color], int] = collections.defaultdict(lambda: 0)

    for index, letter in enumerate(guess):
        if color_mask & (1 << index) != 0:
            counts[(letter, Color.GREEN)] += 1
        elif color_mask & (1 << (index + len(guess))) != 0:
            counts[(letter, Color.YELLOW)] += 1

    for i in range(len(guess)):
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
