"""
Module for storing the programs application logger
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Iterable

from tqdm import tqdm

def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser"""

    parser = argparse.ArgumentParser(
        prog="RAG_Timeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-d",
        "--download-feed",
        help="Pull in the latest feeds via the RSS Feed library.",
        dest="update",
        action="store_true",
    )

    return parser

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
