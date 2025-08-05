"""
Module for storing the programs application logger
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

from tqdm import tqdm


class AppLogger(logging.Logger):

    def __init__(self, name: str, level: int | str = logging.DEBUG) -> None:
        """General helper class for"""
        super().__init__(name, level)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] (%(levelname)s): %(message)s")
        ch.setFormatter(formatter)
        self.addHandler(ch)

    def is_debug(self) -> bool:
        return self.isEnabledFor(logging.DEBUG)

    def tqdm(
        self, iterable: Iterable[Any], desc: str | None = None, ignore: bool = False
    ):
        """Only use tqdm bar while debugging"""
        if not ignore and self.isEnabledFor(logging.DEBUG):
            return tqdm(iterable, desc=desc)

        return iterable


app_logger = AppLogger("`RAG Timeline`", level=logging.DEBUG)
