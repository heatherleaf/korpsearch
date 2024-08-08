
import os
import sys
import math
import logging
import gzip, bz2, lzma
from pathlib import Path
from typing import Any, Literal, List, Iterable, Optional, BinaryIO, Union

ByteOrder = Literal['little', 'big']


def add_suffix(path:Path, suffix:str):
    """Add the suffix to the path, unless it's already there."""
    if path.suffix != suffix:
        path = Path(str(path) + suffix)
        # Alternatively: Path(path).with_suffix(path.suffix + suffix)
    return path


def clean_up(path:Path, suffixes:List[str]):
    for suffix in suffixes:
        add_suffix(path, suffix).unlink()


def min_bytes_to_store_values(max_value:int) -> int:
    """The minimal n:o bytes needed to store values `0...max_value`"""
    return math.ceil(math.log(max_value + 1, 2) / 8)


class CompressedFileReader:
    """
    A class that can read compressed (and uncompressed) files, 
    and where you can query the original file size and the current position.
    Use e.g. like this:

    >>> with (basefile := CompressedFileReader(path)) as Reader:
    >>>     for line in (pbar := tqdm(Reader, total=basefile.file_size()):
    >>>         pbar.update(basefile.file_position() - pbar.n)
    >>>         ...do something with line...
    """
    basefile : BinaryIO
    reader : BinaryIO

    def __init__(self, path:Union[Path,str]):
        path = Path(path)
        self.basefile = binfile = open(path, 'rb')
        self.reader = (
            gzip.open(binfile, mode='rb') if path.suffix == '.gz'  else
            bz2.open(binfile, mode='rb')  if path.suffix == '.bz2' else 
            lzma.open(binfile, mode='rb') if path.suffix == '.xz'  else 
            binfile
        )   # type: ignore

    def file_position(self) -> int:
        return self.basefile.tell()

    def file_size(self) -> int:
        return os.fstat(self.basefile.fileno()).st_size

    def close(self):
        self.reader.close()
        self.basefile.close()

    def __enter__(self) -> BinaryIO:
        return self.reader

    def __exit__(self, *_):
        self.close()


###############################################################################
## Progress bar

class NoProgressBar:
    """A simple progress bar wrapper class, doing nothing at all."""
    n = 0

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


try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print("Module `tqdm` not found, please consider installing it.\n", file=sys.stderr)
    tqdm = NoProgressBar  # type: ignore

_tqdm_bar_format = '{desc:20s} {percentage:3.0f}%|{bar}|{n_fmt:>7s}/{total_fmt} [{elapsed},{rate_fmt:>10s}{postfix}]'

def progress_bar(iterable:Optional[Iterable]=None, desc:str="", **kwargs) -> tqdm:  # type: ignore
    loglevel = logging.root.getEffectiveLevel()
    if loglevel > logging.INFO:
        return NoProgressBar(iterable)  # type: ignore
    kwargs.setdefault('leave', loglevel <= logging.DEBUG)
    kwargs.setdefault('unit_scale', True)
    kwargs.setdefault('bar_format', _tqdm_bar_format)
    return tqdm(iterable=iterable, desc=desc.ljust(20), **kwargs)


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


def setup_logger(format:str, timedivider:int=1000, loglevel:int=logging.WARNING, logfile:Optional[Path]=None):
    formatter = RelativeTimeFormatter(format, style='{', divider=timedivider)
    logging.basicConfig(level=loglevel, filename=logfile)
    logging.root.handlers[0].setFormatter(formatter)



# from typing import Protocol, Generic
# from abc import abstractmethod
# 
# CT = TypeVar("CT", bound='Comparable', contravariant=True)
# 
# class Comparable(Generic[CT], Protocol):
#     """Protocol for annotating comparable types."""
#     @abstractmethod
#     def __lt__(self, other: CT) -> bool: ...
#     @abstractmethod
#     def __le__(self, other: CT) -> bool: ...
#     @abstractmethod
#     def __gt__(self, other: CT) -> bool: ...
#     @abstractmethod
#     def __ge__(self, other: CT) -> bool: ...

class ComparableWithCounter:
    val : Any
    ctr : int = 0
    def __init__(self, n:Any):
        self.val = n
    def __lt__(self, other:'ComparableWithCounter') -> bool:
        ComparableWithCounter.ctr += 1
        return self.val < other.val
    def __le__(self, other:'ComparableWithCounter') -> bool:
        ComparableWithCounter.ctr += 1
        return self.val <= other.val
    def __gt__(self, other:'ComparableWithCounter') -> bool:
        ComparableWithCounter.ctr += 1
        return self.val > other.val
    def __ge__(self, other:'ComparableWithCounter') -> bool:
        ComparableWithCounter.ctr += 1
        return self.val >= other.val
    def __eq__(self, other:object) -> bool:
        ComparableWithCounter.ctr += 1
        return isinstance(other, ComparableWithCounter) and self.val == other.val
    def __ne__(self, other:object) -> bool:
        return not (self == other)

