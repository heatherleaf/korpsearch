
import os
import sys
import re
import math
import logging
import gzip, bz2, lzma
from pathlib import Path
from typing import Any, Protocol, TypeVar, Literal, BinaryIO, Optional, NewType
from collections.abc import Iterable, Iterator, Callable
from abc import abstractmethod


###############################################################################
## Project-specific constants and functions

Feature = NewType('Feature', bytes)
FValue = NewType('FValue', bytes)

WORD = Feature(b'word')
SENTENCE = Feature(b's')

EMPTY = FValue(b'')
START = FValue(b's')


def check_feature(feature: Feature) -> None:
    assert isinstance(feature, bytes), f"Feature must be a bytestring: {feature!r}"
    assert re.match(br'^[a-z_][a-z_0-9]*$', feature), f"Ill-formed feature: {feature.decode()}"


###############################################################################
## Type definitions

ByteOrder = Literal['little', 'big']

T = TypeVar('T')


class ComparableProtocol(Protocol):
    """Protocol for annotating comparable types."""
    @abstractmethod
    def __lt__(self: 'CT', other: 'CT', /) -> bool: ...

CT = TypeVar('CT', bound=ComparableProtocol)


###############################################################################
## File/path utilities

def add_suffix(path: Path, suffix: str) -> Path:
    """Add the suffix to the path, unless it's already there."""
    if path.suffix != suffix:
        path = Path(str(path) + suffix)
        # Alternatively: Path(path).with_suffix(path.suffix + suffix)
    return path


def uncompressed_suffix(path: Path) -> str:
    if path.suffix in CompressedFileReader.compressors:
        return path.with_suffix('').suffix
    else:
        return path.suffix

def clean_up(path: Path, suffixes: list[str]):
    """For every given suffix, the given path with the suffix appended is removed from the file system."""
    for suffix in suffixes:
        try:
            add_suffix(path, suffix).unlink()
        except FileNotFoundError:
            pass


###############################################################################
## N:o bytes needed to store integer values

def get_integer_size(max_value: int) -> int:
    """The minimal n:o bytes needed to store values `0...max_value`"""
    itemsize = math.ceil(math.log(max_value + 1, 2) / 8)
    assert 1 <= itemsize <= 8
    if itemsize == 3: itemsize = 4
    if itemsize > 4: itemsize = 8
    return itemsize


TypeFormat = Literal['B', 'H', 'I', 'Q']

def get_typecode(itemsize: int) -> TypeFormat:
    """Returns the memoryview typecode for the given bytesize of unsigned integers"""
    typecodes: dict[int, TypeFormat] = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    return typecodes[itemsize]


###############################################################################
## Reading (compressed and uncompressed) files

def binsearch_lookup(start: int, end: int, key: CT, lookup: Callable[[int], CT]) -> bool:
    try:
        binsearch(start, end, key, lookup)
        return True
    except KeyError:
        return False


def binsearch(start: int, end: int, key: CT, lookup: Callable[[int], CT], error: bool = True) -> int:
    while start <= end:
        mid = (start + end) // 2
        mykey = lookup(mid)
        if mykey == key:
            return mid
        elif mykey < key:
            start = mid + 1
        else:
            end = mid - 1
    if error:
        raise KeyError(f'Key "{key}" not found')
    return -1


def binsearch_first(start: int, end: int, key: CT, lookup: Callable[[int], CT], error: bool = True) -> int:
    while start <= end:
        mid = (start + end) // 2
        if lookup(mid) < key:
            start = mid + 1
        else:
            end = mid - 1
    if error and lookup(start) != key:
        raise KeyError(f'Key "{key}" not found')
    return start


def binsearch_last(start: int, end: int, key: CT, lookup: Callable[[int], CT], error: bool = True) -> int:
    while start <= end:
        mid = (start + end) // 2
        if key < lookup(mid):
            end = mid - 1
        else:
            start = mid + 1
    if error and lookup(end) != key:
        raise KeyError(f'Key "{key}" not found')
    return end


def binsearch_range(start: int, end: int, key: CT, lookup: Callable[[int], CT], error: bool = True) -> tuple[int, int]:
    start = binsearch_first(start, end, key, lookup, error)
    end = binsearch_last(start, end, key, lookup, error)
    return start, end



###############################################################################
## Reading (compressed and uncompressed) files

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
    compressors = {
        '.gz': gzip,
        '.bz2': bz2,
        '.xz': lzma,
    }

    basefile: BinaryIO
    reader: BinaryIO

    def __init__(self, path: Path) -> None:
        path = Path(path)
        self.basefile = binfile = open(path, 'rb')
        compressor = self.compressors.get(path.suffix)
        if compressor:
            self.reader = compressor.open(binfile, mode='rb')  # type: ignore
        else:
            self.reader = binfile

    def file_position(self) -> int:
        return self.basefile.tell()

    def file_size(self) -> int:
        return os.fstat(self.basefile.fileno()).st_size

    def close(self) -> None:
        self.reader.close()
        self.basefile.close()

    def __enter__(self) -> BinaryIO:
        return self.reader

    def __exit__(self, *_: Any) -> None:
        self.close()


###############################################################################
## Progress bar

class ProgressBar(Iterable[T]):
    """A simple progress bar wrapper class, doing nothing at all."""
    n: int = 0

    def __init__(self, iterable: Optional[Iterable[T]] = None, **_: Any) -> None:
        self._iter = iter(()) if iterable is None else iter(iterable)

    def __enter__(self) -> 'ProgressBar[T]':
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def __iter__(self) -> Iterator[T]:
        return self._iter

    def update(self, n: int) -> None:
        pass


try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print("Module `tqdm` not found, please consider installing it.\n", file=sys.stderr)

_tqdm_bar_format = '{desc:20s} {percentage:3.0f}%|{bar}|{n_fmt:>7s}/{total_fmt} [{elapsed},{rate_fmt:>10s}{postfix}]'

def progress_bar(iterable: Optional[Iterable[T]] = None, desc: str = "", **kwargs: Any) -> ProgressBar[T]:
    loglevel = logging.root.getEffectiveLevel()
    if loglevel > logging.INFO:
        return ProgressBar(iterable)
    kwargs.setdefault('leave', loglevel <= logging.DEBUG)
    kwargs.setdefault('unit_scale', True)
    kwargs.setdefault('bar_format', _tqdm_bar_format)
    try:
        return tqdm(iterable=iterable, desc=desc.ljust(20), **kwargs)  # type: ignore
    except NameError:
        return ProgressBar(iterable)


###############################################################################
## Debugging 

class RelativeTimeFormatter(logging.Formatter):
    def __init__(self, *args: Any, divider: float = 1000, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._divider = divider

    def format(self, record: logging.LogRecord) -> str:
        record.relativeCreated = record.relativeCreated / self._divider
        record.warningname = f"{record.levelname:<9s}" if record.levelno >= logging.WARNING else ""
        return super().format(record)


def setup_logger(format: str, timedivider: int = 1000, loglevel: int = logging.WARNING, logfile: Optional[Path] = None) -> None:
    formatter = RelativeTimeFormatter(format, style='{', divider=timedivider)
    logging.basicConfig(level=loglevel, filename=logfile)
    logging.root.handlers[0].setFormatter(formatter)

