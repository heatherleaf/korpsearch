
import sys
import logging
from pathlib import Path
from typing import Any, Literal, Iterator, Iterable


ByteOrder = Literal['little', 'big']


def add_suffix(path:Path, suffix:str):
    """Add the suffix to the path, unless it's already there."""
    if path.suffix != suffix:
        path = Path(str(path) + suffix)
        # Alternatively: Path(path).with_suffix(path.suffix + suffix)
    return path



###############################################################################
## Progress bar

class NoTQDM:
    def __init__(self, iterable=None, desc=None, **kwargs):
        self._iter = None if iterable is None else iter(iterable)

    def __enter__(self):
        return self if self._iter is None else self._iter

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self if self._iter is None else self._iter

    def update(self, _add):
        pass


tqdm_bar_format = '{desc:20s} {percentage:3.0f}%|{bar}|{n_fmt:>7s}/{total_fmt} [{elapsed},{rate_fmt:>10s}{postfix}]'
# Specify a custom bar string formatting. May impact performance. 
# [default: '{l_bar}{bar}{r_bar}'], where l_bar='{desc}: {percentage:3.0f}%|' and r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]' 
# Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage, elapsed, elapsed_s, ncols, nrows, desc, unit, rate, rate_fmt, rate_noinv, rate_noinv_fmt, rate_inv, rate_inv_fmt, postfix, unit_divisor, remaining, remaining_s, eta. 
# Note that a trailing ": " is automatically removed after {desc} if the latter is empty.

try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable:Iterable=(), desc:str='', unit_scale:bool=True, leave:bool=False, **kwargs):
        return _tqdm(iterable=iterable, desc=desc.ljust(20), unit_scale=unit_scale, leave=leave, bar_format=tqdm_bar_format, **kwargs)
except ModuleNotFoundError:
    print("Module `tqdm` not found, please consider installing it.", file=sys.stderr)
    tqdm = NoTQDM  # type: ignore


###############################################################################
## Debugging 

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
