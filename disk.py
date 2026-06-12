"""General-purpose on-disk data structures."""

import sys
import re
import json
from pathlib import Path
from mmap import mmap
from typing import Any, NamedTuple
from collections.abc import Iterator, Iterable
from dataclasses import dataclass

from util import add_suffix, get_integer_size, get_typecode, binsearch, binsearch_range


################################################################################
## On-disk arrays of integers

class IntArray:
    array_suffix = '.ia'
    default_itemsize = 4

    array: memoryview
    path: Path | None = None
    config: dict[str, Any]

    def __init__(self, source: Path|mmap|bytearray, itemsize: int = default_itemsize) -> None:
        assert not isinstance(source, bytes), "bytes is not mutable - use bytearray instead"
        if isinstance(source, Path):
            with open(self.getconfigpath(source)) as configfile:
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

    def slice(self, j: int, k: int) -> memoryview:
        return self.array[j:k]

    def __setitem__(self, i: int, value: int) -> None:
        self.array[i] = value

    def __iter__(self) -> Iterator[int]:
        yield from self.array

    def __enter__(self) -> memoryview:
        return self.array

    def __exit__(self, *_: Any) -> None:
        self.close()

    def getconfig(self) -> dict[str, Any]:
        return self.config

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
        if self.path:
            config = self.getconfig()
            config['size'] = newsize
            with open(IntArray.getconfigpath(self.path), 'w') as configfile:
                print(json.dumps(config), file=configfile)

    def sanity_check(self, sorted: bool = False) -> None:
        if sorted:
            prev = -1
            for val in self:
                assert prev <= val, "Sorted IntArray is not sorted"
        assert len(self) == self.getconfig()['size'], "Size mismatch"

    @staticmethod
    def create(size: int, path: Path|None = None, max_value: int = 0, itemsize: int = 0, **config: Any) -> 'IntArray':
        assert not (max_value and itemsize), "Only one of 'max_value' and 'itemsize' should be provided."
        if max_value > 0:
            itemsize = get_integer_size(max_value)
        if not itemsize:
            itemsize = IntArray.default_itemsize
        if path:
            with open(IntArray.getconfigpath(path), 'w') as configfile:
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
    def build(path: Path|None, values: Iterable[int], size: int = 0, **config: Any) -> None:
        if isinstance(values, (list, tuple)):
            if size:
                assert size == len(values), "Wrong size"
            else:
                size = len(values)
        elif not size:
            values = list(values)
            size = len(values)
        array = IntArray.create(size, path, **config)
        i = -1
        for i, val in enumerate(values):
            array[i] = val
        realsize = i + 1
        if realsize < size:
            array.truncate(realsize)
        array.close()


    @staticmethod
    def getpath(path: Path) -> Path:
        return add_suffix(path, IntArray.array_suffix)

    @staticmethod
    def getconfigpath(path: Path) -> Path:
        return add_suffix(IntArray.getpath(path), '.cfg')


################################################################################
## On-disk arrays of bytestrings
## Note: if you want to use .finditer(), the strings *must not* contain newlines \n

class BytesArray:
    _starts: IntArray
    _rawdata: mmap | bytearray

    def __init__(self, path: Path) -> None:
        startspath, rawpath = self.getpaths(path)
        self._starts = IntArray(startspath)
        with open(rawpath, 'r+b') as file:
            try:
                self._rawdata = mmap(file.fileno(), 0)
            except ValueError:  # "cannot mmap an empty file"
                self._rawdata = bytearray(0)

    def __len__(self) -> int:
        return len(self._starts) - 1

    def __getitem__(self, i: int) -> bytes:
        arr = self._starts.array
        start, end = arr[i], arr[i+1]
        return self._rawdata[start:end-1]

    def slice(self, j: int, k: int) -> list[bytes]:
        arr = self._starts.array
        raw = self._rawdata
        return [raw[start:end-1] for i in range(j, k) for start, end in [(arr[i], arr[i+1])]]

    def __iter__(self) -> Iterator[bytes]:
        raw = self._rawdata
        start = -1
        for end in self._starts:
            if start >= 0:
                yield raw[start:end-1]
            start = end

    def __setitem__(self, i: int, value: bytes) -> None:
        raise TypeError("'BytesArray' does not support item assignment")

    def __enter__(self) -> 'BytesArray':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._starts.close()
        if isinstance(self._rawdata, mmap):
            self._rawdata.close()

    def finditer(self, regex: bytes, flags: int = 0) -> Iterator[int]:
        regex = b'^(?:' + regex + b')$'
        flags |= re.MULTILINE
        positions = self._starts.array
        pos = 0
        for match in re.finditer(regex, self._rawdata, flags=flags):
            start, end = match.span()
            if start < end:
                yield binsearch(pos, len(positions), start, lambda i: positions[i])

    def sanity_check(self, accept_newlines: bool = False) -> None:
        starts = self._starts.array
        rawdata = self._rawdata
        assert starts[0] == 0 and starts[-1] == len(self._rawdata)
        for start, end in zip(starts, starts[1:]):
            assert start <= end, f"'BytesArray' position error: {start} > {end}"
            if not accept_newlines:
                assert b'\n' not in rawdata[start:end-1], f"'BytesArray' value contains newline '\\n'"

    @staticmethod
    def build(path: Path, values: Iterable[bytes]) -> int:
        startspath, rawpath = BytesArray.getpaths(path)
        pos = 0
        starts: list[int] = [pos]
        with open(rawpath, 'wb') as rawfile:
            for val in values:
                rawfile.write(val + b'\n')
                pos += len(val) + 1
                starts.append(pos)
        IntArray.build(startspath, starts, len(starts), max_value=starts[-1])
        return len(starts) - 1

    @staticmethod
    def getpaths(path: Path) -> tuple[Path, Path]:
        return (add_suffix(path, '.starts'), add_suffix(path, '.rawdata'))


################################################################################
## Symbols (interned bytestrings)

Symbol = int

class SymbolRange(NamedTuple):
    start: Symbol
    end: Symbol

@dataclass(frozen=True, order=True, init=False)
class SymbolList:
    symbols: tuple[Symbol, ...]
    def __init__(self, symbols: Iterable[Symbol]):
        # We need to use __setattr__ because the class is frozen:
        object.__setattr__(self, 'symbols', tuple(symbols))

Symbols = Symbol | SymbolRange | SymbolList


class SymbolCollection:
    _bytesarray: BytesArray
    _preload: dict[bytes, Symbol]

    def __init__(self, path: Path, preload: bool = False) -> None:
        self._bytesarray = BytesArray(path)
        self._preload = {}
        if preload:
            self.preload()

    def __len__(self) -> int:
        return len(self._bytesarray)

    def preload(self) -> None:
        if not self._preload:
            self._preload = {}
            for i in range(len(self)):
                self._preload[self._bytesarray[i]] = i

    def to_name(self, index: Symbol) -> bytes:
        return self._bytesarray[index]

    def to_symbol(self, name: bytes) -> Symbol:
        try:
            if self._preload:
                return self._preload[name]
            else:
                ba = self._bytesarray
                return binsearch(0, len(self)-1, name, lambda i: ba[i])
        except (KeyError, IndexError, ValueError):
            raise ValueError(f"Symbol doesn't exist: {name.decode(errors='ignore')}")

    def find_prefix(self, prefix: bytes) -> Symbols:
        try:
            n = len(prefix)
            ba = self._bytesarray
            start, end = binsearch_range(0, len(self)-1, prefix, prefix, lambda i: ba[i][:n])
            return SymbolRange(start, end)
        except (KeyError, IndexError, ValueError):
            return SymbolList([])

    def find_regex(self, regex: bytes, flags: int = 0) -> Symbols:
        return SymbolList(self._bytesarray.finditer(regex, flags))

    def __enter__(self) -> 'SymbolCollection':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._bytesarray.close()

    def sanity_check(self) -> None:
        self._bytesarray.sanity_check()
        old = None
        for name in self._bytesarray:
            if old is not None:
                assert old < name, f"'SymbolCollection' order error: {old!r} >= {name!r}"
            old = name

    @staticmethod
    def build(path: Path, names: Iterable[bytes]) -> int:
        names = sorted({b''} | set(names))
        assert names[0] == b''
        return BytesArray.build(path, names)


################################################################################
## On-disk arrays of symbols (interned bytestrings)

class SymbolArray:
    symbols: SymbolCollection
    array: IntArray

    def __init__(self, path: Path, preload: bool = False) -> None:
        arrpath, symspath = SymbolArray.getpaths(path)
        self.array = IntArray(arrpath)
        self.symbols = SymbolCollection(symspath, preload)

    def raw(self) -> 'memoryview[Symbol]':
        return self.array.array

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, i: int) -> Symbol:
        return self.array.array[i]

    def __setitem__(self, i: int, value: Symbol) -> None:
        self.array[i] = value

    def __iter__(self) -> Iterator[Symbol]:
        yield from self.array

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
        arrpath, symspath = SymbolArray.getpaths(path)
        nsymbols = SymbolCollection.build(symspath, names)
        IntArray.create(max_size, path, max_value=nsymbols)
        return SymbolArray(arrpath, preload=True)

    @staticmethod
    def getpaths(path: Path) -> tuple[Path, Path]:
        return (path, add_suffix(path, '.symbols'))


################################################################################
## On-disk maps/dicts from integers to bytestrings

class IntBytesMap:
    _keys: IntArray
    _values: BytesArray

    def __init__(self, path: Path):
        keyspath, valspath = self.getpaths(path)
        self._keys = IntArray(keyspath)
        self._values = BytesArray(valspath)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: int) -> bytes:
        pos = binsearch(0, len(self._keys)-1, key, lambda k: self._keys[k])
        return self._values[pos]

    def slice(self, start_key: int, end_key: int) -> list[bytes]:
        start, end = binsearch_range(0, len(self._keys)-1, start_key, end_key, lambda k: self._keys[k])
        return self._values.slice(start, end+1)

    def get_key_position(self, key: int) -> int:
        return binsearch(0, len(self._keys), key, lambda k: self._keys[k])

    def __setitem__(self, key: int, value: bytes) -> None:
        raise TypeError("'IntBytesMap' does not support item assignment")

    def __enter__(self) -> 'IntBytesMap':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._keys.close()
        self._values.close()

    def sanity_check(self) -> None:
        self._keys.sanity_check(sorted=True)
        self._values.sanity_check(accept_newlines=True)
        assert len(self._keys) == len(self._values), "Different lengths for keys and values"

    @staticmethod
    def build(path: Path, keys: Iterable[int], values: Iterable[bytes], **xargs: Any) -> None:
        # Note: the 'keys' must be sorted!
        keyspath, valspath = IntBytesMap.getpaths(path)
        IntArray.build(keyspath, keys, **xargs)
        BytesArray.build(valspath, values)

    @staticmethod
    def getpaths(path: Path) -> tuple[Path, Path]:
        return (add_suffix(path, '.keys'), add_suffix(path, '.values'))
