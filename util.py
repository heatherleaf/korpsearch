
import logging
from typing import Any


class RelativeTimeFormatter(logging.Formatter):
    def __init__(self, *args:Any, divider:float=1000, **kwargs:Any):
        super().__init__(*args, **kwargs)
        self._divider = divider

    def format(self, record:logging.LogRecord) -> str:
        record.relativeCreated = record.relativeCreated / self._divider
        warningname : str = f"{record.levelname:<9s}" if record.levelno >= logging.WARNING else ""
        record.warningname = warningname  # type: ignore
        return super().format(record)


def setup_logger(format:str, timedivider:int=1000, loglevel:int=logging.WARNING):
    formatter = RelativeTimeFormatter(format, style='{', divider=timedivider)
    logging.basicConfig(level=loglevel)
    logging.root.handlers[0].setFormatter(formatter)
