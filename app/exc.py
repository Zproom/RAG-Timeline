"""
Module to store custom exceptions for error handling
"""

from __future__ import annotations


class InitError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DbError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ScraperError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RagError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
