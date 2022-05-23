
import logging

def bytesify(s):
    return s.encode() if isinstance(s, str) else bytes(s)


class RelativeTimeFormatter(logging.Formatter):
    def __init__(self, *args, divider=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self._divider = divider

    def format(self, record):
        record.relativeCreated = record.relativeCreated / self._divider
        record.warningname = f"{record.levelname:<9s}" if record.levelno >= logging.WARNING else ""
        return super().format(record)


def setup_logger(format, timedivider=1000, loglevel=logging.WARNING):
    formatter = RelativeTimeFormatter(format, style='{', divider=timedivider)
    logging.basicConfig(level=loglevel)
    logging.root.handlers[0].setFormatter(formatter)
