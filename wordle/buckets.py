from __future__ import annotations

from wordle import cache


def get_buckets_many(guesses: list[str], solutions: list[str]) -> dict[str, dict[int, int]]:
    buckets_per_guess: dict[str, dict[int, int]] = {}

    for guess in guesses:
        buckets_per_guess[guess] = get_buckets(guess, solutions)

    return buckets_per_guess


def get_buckets(guess: str, solutions: list[str]) -> dict[int, int]:
    buckets: dict[int, int] = {}

    for solution in solutions:
        try:
            colors = cache.BUCKETS_CACHE[(guess, solution)]
        except KeyError:
            letters_seen: dict[str, int] = {}
            colors = 0

            for i in range(len(guess)):
                guess_letter, solution_letter = guess[i], solution[i]

                if guess_letter == solution_letter:
                    colors += (1 << i)
                    letters_seen.setdefault(guess_letter, 0)
                    letters_seen[guess_letter] += 1

            for i in range(len(guess)):
                guess_letter, solution_letter = guess[i], solution[i]
                if guess_letter != solution_letter and guess_letter in solution and letters_seen.get(guess_letter, 0) < cache.COUNTERS_PER_SOLUTION[solution][guess_letter]:
                    letters_seen.setdefault(guess_letter, 0)
                    letters_seen[guess_letter] += 1
                    colors += (1 << (i + 5))

            cache.BUCKETS_CACHE[(guess, solution)] = colors

        buckets.setdefault(colors, 0)
        buckets[colors] += 1

    return buckets
