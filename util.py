
import os
import sys
import math
import logging
import gzip, bz2, lzma
from pathlib import Path
from typing import Any, TypeVar, Literal, BinaryIO, Optional
from collections.abc import Iterable, Iterator

ByteOrder = Literal['little', 'big']

T = TypeVar('T')


def add_suffix(path: Path, suffix: str) -> Path:
    """Add the suffix to the path, unless it's already there."""
    if path.suffix != suffix:
        path = Path(str(path) + suffix)
        # Alternatively: Path(path).with_suffix(path.suffix + suffix)
    return path


def get_integer_size(max_value: int) -> int:
    """The minimal n:o bytes needed to store values `0...max_value`"""
    elemsize = math.ceil(math.log(max_value + 1, 2) / 8)
    assert 1 <= elemsize <= 8
    if elemsize == 3: elemsize = 4
    if elemsize > 4: elemsize = 8
    return elemsize


def get_typecode(elemsize: int) -> str:
    """Returns the memoryview typecode for the given bytesize of unsigned integers"""
    typecodes = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    return typecodes[elemsize]


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
    basefile: BinaryIO
    reader: BinaryIO

    def __init__(self, path: Path) -> None:
        path = Path(path)
        self.basefile = binfile = open(path, 'rb')
        self.reader = (
            gzip.open(binfile, mode='rb') if path.suffix == '.gz'  else   # type: ignore
            bz2.open(binfile, mode='rb')  if path.suffix == '.bz2' else
            lzma.open(binfile, mode='rb') if path.suffix == '.xz'  else
            binfile
        )

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

