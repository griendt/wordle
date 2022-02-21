from dataclasses import field
from typing import Optional, Counter

from main import ColorMask

TURN_1_GUESS: Optional[str] = None
TURN_2_CACHE: dict[ColorMask, str] = field(default_factory=dict)

# We may keep track of computations from a previous Game, if we are doing a full run. Normally it would suffice to simply cache/hard-code a turn 1 starting word;
# but when doing a full run with the "truncate solution space" optimization flag turned on, the best starting word will actually vary over time. To avoid having to
# compute a large amount of options each Game, we may simply compute it once for the first game and tweak the stats according to the word that gets truncated for the next Game.
TURN_1_CACHE_PREVIOUS_SOLUTION: Optional[str] = None
TURN_1_CACHE: Optional[dict[str, dict[ColorMask, int]]] = None

COUNTERS_PER_SOLUTION: dict[str, Counter] = None
BUCKETS_CACHE: dict[tuple[str, str], ColorMask] = None
