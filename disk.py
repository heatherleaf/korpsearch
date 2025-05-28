"""General-purpose on-disk data structures."""

import sys
import re
import json
from pathlib import Path
from mmap import mmap
import itertools
from typing import Optional, Union, Any, NewType
from collections.abc import Iterator, Iterable

from util import add_suffix, get_integer_size, get_typecode, binsearch, binsearch_range, file_size


################################################################################
## On-disk arrays of integers

class IntArray:
    array_suffix = '.ia'
    config_suffix = '.cfg'
    default_itemsize = 4

    array: memoryview
    path: Optional[Path] = None
    config: dict[str, Any]

    def __init__(self, source: Union[Path, mmap, bytearray], itemsize: int = default_itemsize) -> None:
        assert not isinstance(source, bytes), "bytes is not mutable - use bytearray instead"
        if isinstance(source, Path):
            with open(self.getconfig(source)) as configfile:
                self.config = json.load(configfile)
            assert self.config['byteorder'] == sys.byteorder, f"Cannot handle byteorder {self.config['byteorder']}"
            itemsize = self.config['itemsize']
            self.path = self.getpath(source)
            with open(self.path, 'r+b') as file:
                try:
                    source = mmap(file.fileno(), 0)
                except ValueError:  # "cannot mmap an empty file"
                    source = bytearray(0)
        self.array = memoryview(source).cast(get_typecode(itemsize))

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, i: int) -> int:
        return self.array[i]

    def __setitem__(self, i: int, value: int) -> None:
        self.array[i] = value

    def __iter__(self) -> Iterator[int]:
        yield from self.array

    def __enter__(self) -> memoryview:
        return self.array

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        obj = self.array.obj
        self.array.release()
        if isinstance(obj, mmap):
            obj.close()

    def truncate(self, newsize: int) -> None:
        obj = self.array.obj
        itemsize = self.array.itemsize
        self.array.release()
        if isinstance(obj, bytearray):
            assert self.path is None
            del obj[newsize * itemsize:]
            typecode = get_typecode(itemsize)
            self.array = memoryview(obj).cast(typecode)
        elif isinstance(obj, mmap):
            assert self.path
            obj.close()
            with open(self.path, 'r+b') as file:
                file.truncate(newsize * itemsize)
                try:
                    obj = mmap(file.fileno(), 0)
                except ValueError:  # "cannot mmap an empty file"
                    obj = bytearray(0)
        else:
            raise ValueError(f"Cannot handle memoryview object {obj}")
        self.array = memoryview(obj).cast(get_typecode(itemsize))

    @staticmethod
    def create(size: int, path: Optional[Path] = None, max_value: int = 0, itemsize: int = 0, **config: Any) -> 'IntArray':
        assert not (max_value and itemsize), "Only one of 'max_value' and 'itemsize' should be provided."
        if max_value > 0:
            itemsize = get_integer_size(max_value)
        if not itemsize:
            itemsize = IntArray.default_itemsize
        if path:
            with open(IntArray.getconfig(path), 'w') as configfile:
                print(json.dumps({
                    'itemsize': itemsize,
                    'byteorder': sys.byteorder,
                    'size': size,
                    **config,
                }), file=configfile)
            with open(IntArray.getpath(path), 'wb') as file:
                file.truncate(size * itemsize)
            return IntArray(path)
        else:
            data = bytearray(size * itemsize)
            return IntArray(data, itemsize)

    @staticmethod
    def build(path: Optional[Path], values: Iterable[int], size: int = 0, **config: Any) -> None:
        if isinstance(values, (list, tuple)):
            assert not size or size == len(values), "Wrong size"
        elif not size:
            values = list(values)
            size = len(values)
        with IntArray.create(size, path, **config) as array:
            for i, val in enumerate(values):
                array[i] = val

    @classmethod
    def disksize(cls, path: Path) -> int:
        return file_size(cls.getpath(path)) + file_size(cls.getconfig(path))

    @classmethod
    def getpath(cls, path: Path) -> Path:
        return add_suffix(path, cls.array_suffix)

    @classmethod
    def getconfig(cls, path: Path) -> Path:
        return add_suffix(cls.getpath(path), cls.config_suffix)


################################################################################
## Symbols (interned bytestrings)

Symbol = NewType('Symbol', int)
SymbolRange = tuple[Symbol, Symbol]


class SymbolCollection:
    symbols_suffix = '.symbols'

    _rawdata: mmap
    _starts: IntArray
    _preload: dict[bytes, Symbol]

    def __init__(self, path: Path, preload: bool = False) -> None:
        path = self.getpath(path)
        with open(path, 'r+b') as file:
            self._rawdata = mmap(file.fileno(), 0)
        self._starts = IntArray(path)
        assert self._starts.array[0]+1 == self._starts.array[1]
        self._preload = {}
        if preload:
            self.preload()

    def __len__(self) -> int:
        return len(self._starts) - 1

    def to_name(self, index: Symbol|int) -> bytes:
        arr = self._starts.array
        start, nextstart = arr[index], arr[index + 1]
        return self._rawdata[start : nextstart-1]

    def preload(self) -> None:
        if not self._preload:
            self._preload = {}
            for i in range(len(self)):
                self._preload[self.to_name(i)] = Symbol(i)

    def to_symbol(self, name: bytes) -> Symbol:
        try:
            if self._preload:
                return self._preload[name]
            else:
                return Symbol(binsearch(0, len(self)-1, name, lambda i: self.to_name(i)))
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"Symbol doesn't exist: {name.decode(errors='ignore')}")

    def to_symbol_range(self, prefix: bytes) -> SymbolRange:
        try:
            n = len(prefix)
            start, end = binsearch_range(0, len(self)-1, prefix, prefix, lambda i: self.to_name(i)[:n])
            return (Symbol(start), Symbol(end))
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"Symbol prefix doesn't exist: {prefix.decode(errors='ignore')}")

    def __enter__(self) -> 'SymbolCollection':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._rawdata.close()
        self._starts.close()

    def sanity_check(self) -> None:
        starts = self._starts.array
        assert starts[0] == 0 and starts[1] == 1
        old = b''
        for start, end in zip(starts[1:], starts[2:]):
            assert start < end, f"SymbolCollection position error: {start} >= {end}"
            new = self._rawdata[start : end]
            assert old < new, f"SymbolCollection order error: {old!r} >= {new!r}"
            old = new

    def finditer(self, regex: bytes, flags: int = 0) -> Iterator[Symbol]:
        regex = b'^(?:' + regex + b')$'
        flags |= re.MULTILINE
        positions = self._starts.array
        pos = 0
        for match in re.finditer(regex, self._rawdata, flags=flags):
            start, end = match.span()
            if start < end:
                pos = binsearch(pos, len(positions), start, lambda i: positions[i])
                yield Symbol(pos)


    @staticmethod
    def build(path: Path, names: Iterable[bytes]) -> None:
        names = sorted({b''} | set(names))
        assert names[0] == b''

        path = SymbolCollection.getpath(path)
        rawsize = 0
        with open(path, 'wb') as rawfile:
            for name in names:
                rawfile.write(name + b'\n')
                rawsize += len(name) + 1

        starts = list(itertools.accumulate((len(s)+1 for s in names), initial=0))
        with IntArray.create(len(starts), path, max_value=starts[-1]) as arr:
            for i, start in enumerate(starts):
                arr[i] = start
            assert arr[0] == 0 and arr[1] == 1 and arr[-1] == rawsize

    @classmethod
    def getpath(cls, path: Path) -> Path:
        return add_suffix(path, cls.symbols_suffix)


################################################################################
## On-disk arrays of symbols (interned bytestrings)

class SymbolArray:
    symbols: SymbolCollection
    array: IntArray

    def __init__(self, path: Path, preload: bool = False) -> None:
        self.array = IntArray(path)
        self.symbols = SymbolCollection(path, preload)

    def raw(self) -> memoryview:
        return self.array.array

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, i: int) -> Symbol:
        return Symbol(self.array.array[i])

    def __setitem__(self, i: int, value: Symbol) -> None:
        self.array[i] = value

    def __iter__(self) -> Iterator[Symbol]:
        for i in self.array:
            yield Symbol(i)

    def __enter__(self) -> 'SymbolArray':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.array.close()
        self.symbols.close()

    def sanity_check(self) -> None:
        self.symbols.sanity_check()


    @staticmethod
    def create(path: Path, names: Iterable[bytes], max_size: int) -> 'SymbolArray':
        SymbolCollection.build(path, names)
        collection = SymbolCollection(path, preload=True)
        IntArray.create(max_size, path, max_value = len(collection)-1)
        return SymbolArray(path, preload=True)

