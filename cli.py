from __future__ import annotations

import logging

import blessings

terminal: blessings.Terminal = blessings.Terminal()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wordle")
