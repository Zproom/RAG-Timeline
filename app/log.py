import logging


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


app_logger = AppLogger("`RAG Timeline`", level=logging.DEBUG)
