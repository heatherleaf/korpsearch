"""General-purpose on-disk data structures."""

import sys
import json
from pathlib import Path
from mmap import mmap
import itertools
from functools import total_ordering
from typing import overload, BinaryIO, Optional, Union, Any
from collections.abc import Iterator, Iterable, MutableSequence

from util import add_suffix, get_integer_size, get_typecode, binsearch


################################################################################
## On-disk arrays of numbers

class DiskIntArray:
    array_suffix = '.ia'
    config_suffix = '.cfg'
    default_itemsize = 4

    array: memoryview
    path: Optional[Path] = None

    def __init__(self, source: Union[Path, mmap, bytearray], itemsize: int = default_itemsize) -> None:
        assert not isinstance(source, bytes), "bytes is not mutable - use bytearray instead"
        if isinstance(source, Path):
            with open(self.getconfig(source)) as configfile:
                config = json.load(configfile)
            assert config['byteorder'] == sys.byteorder, f"Cannot handle byteorder {config['byteorder']}"
            itemsize = config['itemsize']
            self.path = self.getpath(source)
            with open(self.path, 'r+b') as file:
                try:
                    source = mmap(file.fileno(), 0)
                except ValueError:  # "cannot mmap an empty file"
                    source = bytearray(0)
        self.array = memoryview(source).cast(get_typecode(itemsize))

    def __len__(self) -> int:
        return len(self.array)

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
    def create(size: int, path: Optional[Path] = None, max_value: int = 0, itemsize: int = 0) -> 'DiskIntArray':
        assert not (max_value and itemsize), "Only one of 'max_value' and 'itemsize' should be provided."
        if max_value > 0: 
            itemsize = get_integer_size(max_value)
        if not itemsize:
            itemsize = DiskIntArray.default_itemsize
        if path:
            with open(DiskIntArray.getconfig(path), 'w') as configfile:
                json.dump({
                    'itemsize': itemsize,
                    'byteorder': sys.byteorder,
                }, configfile)
            with open(DiskIntArray.getpath(path), 'wb') as file:
                file.truncate(size * itemsize)
            return DiskIntArray(path)
        else:
            data = bytearray(size * itemsize)
            return DiskIntArray(data, itemsize)

    @classmethod
    def getpath(cls, path: Path) -> Path:
        return add_suffix(path, cls.array_suffix)

    @classmethod
    def getconfig(cls, path: Path) -> Path:
        return add_suffix(cls.getpath(path), cls.config_suffix)


################################################################################
## On-disk array of fixed-width byte sequences

class DiskFixedBytesArray(MutableSequence[bytes]):
    itemsize: int
    _file: BinaryIO
    _mmap: mmap
    _len: int

    def __init__(self, path: Path, itemsize: int) -> None:
        with open(path, 'r+b') as file:
            self._mmap = mmap(file.fileno(), 0)
        self.itemsize = itemsize
        self._len = len(self._mmap) // self.itemsize
        assert len(self._mmap) % self.itemsize == 0, \
            f"Array length ({len(self._mmap)}) is not divisible by itemsize ({self.itemsize})"

    def __len__(self) -> int:
        return self._len

    def __setitem__(self, index: Union[int,slice], value: Union[bytes,Iterable[bytes]]) -> None:
        assert isinstance(index, int) and isinstance(value, bytes), "DiskFixedBytesArray cannot handle slices"
        itemsize = self.itemsize
        start = index * itemsize
        assert len(value) == itemsize
        self._mmap[start : start+itemsize] = value

    def __delitem__(self, index: Union[int,slice]) -> None:
        # Required to be a MutableSequence, but mmap arrays cannot change size
        raise NotImplementedError("DiskFixedBytesArray cannot change size")

    def insert(self, index: int, value: bytes) -> None:
        # Required to be a MutableSequence, but mmap arrays cannot change size
        raise NotImplementedError("DiskFixedBytesArray cannot change size")

    @overload
    def __getitem__(self, index: int) -> bytes: pass
    @overload
    def __getitem__(self, index: slice) -> MutableSequence[bytes]: pass
    def __getitem__(self, index: Union[int,slice]) -> Union[bytes, MutableSequence[bytes]]:
        if isinstance(index, slice):
            return self._slice(index)  # type: ignore
        itemsize = self.itemsize
        start = index * itemsize
        return self._mmap[start : start+itemsize]

    def _slice(self, sl: slice) -> Iterator[bytes]:
        array = self._mmap
        itemsize = self.itemsize
        start, stop, step = sl.indices(len(self))
        for i in range(start * itemsize, stop * itemsize, step * itemsize):
            yield array[i : i+itemsize]

    def __iter__(self) -> Iterator[bytes]:
        return self._slice(slice(None))

    def __enter__(self) -> 'DiskFixedBytesArray':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._mmap.close()


################################################################################
## String interning

class StringCollection:
    strings_suffix = '.strings'

    strings: mmap
    starts: DiskIntArray

    _intern: dict[bytes, int]

    def __init__(self, path: Path, preload: bool = False) -> None:
        path = self.getpath(path)
        with open(path, 'r+b') as file:
            self.strings = mmap(file.fileno(), 0)
        self.starts = DiskIntArray(path)
        assert self.starts.array[0] == self.starts.array[1]
        self._intern = {}
        if preload:
            self.preload()

    def __len__(self) -> int:
        return len(self.starts) - 1

    def from_index(self, index: int) -> 'InternedString':
        return InternedString(self, index)

    def preload(self) -> None:
        if not self._intern:
            self._intern = {}
            for i in range(len(self)):
                self._intern[bytes(self.from_index(i))] = i

    def intern(self, string: Union[bytes,'InternedString']) -> 'InternedString':
        if isinstance(string, InternedString):
            assert string.db is self
            return string
        if self._intern:
            index = self._intern[string]
        else:
            index = binsearch(0, len(self)-1, string, lambda i: bytes(self.from_index(i)))
        return self.from_index(index)

    def __enter__(self) -> 'StringCollection':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.strings.close()
        self.starts.close()

    def sanity_check(self) -> None:
        starts = self.starts.array
        assert starts[0] == starts[1] == 0
        old = b''
        for start, end in zip(starts[1:], starts[2:]):
            assert start < end, f"StringCollection position error: {start} >= {end}"
            new = self.strings[start : end]
            assert old < new, f"StringCollection order error: {old!r} >= {new!r}"
            old = new


    @staticmethod
    def build(path: Path, strings: Iterable[bytes]) -> None:
        stringset = set(strings)
        stringset.add(b'')
        stringlist = sorted(stringset)
        assert stringlist[0] == b''

        path = StringCollection.getpath(path)
        with open(path, 'wb') as stringsfile:
            for string in stringlist: 
                stringsfile.write(string)

        starts = list(itertools.accumulate((len(s) for s in stringlist), initial=0))
        with DiskIntArray.create(len(starts), path, max_value=starts[-1]) as arr:
            for i, start in enumerate(starts):
                arr[i] = start
            assert arr[0] == arr[1] == 0

    @classmethod
    def getpath(cls, path: Path) -> Path:
        return add_suffix(path, cls.strings_suffix)


@total_ordering
class InternedString:
    __slots__ = ['db', 'index']
    db: StringCollection
    index: int

    def __init__(self, db: StringCollection, index: int) -> None:
        # We cannot use 'self.db = db', because this will call ' __setattr__'
        # which in turn will raise an exception
        object.__setattr__(self, "db", db)
        object.__setattr__(self, "index", index)

    def __bytes__(self) -> bytes:
        arr = self.db.starts.array
        start = arr[self.index]
        nextstart = arr[self.index + 1]
        return self.db.strings[start : nextstart]

    def __str__(self) -> str:
        return bytes(self).decode()

    def __repr__(self) -> str:
        return f"InternedString({self})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, InternedString) and self.db is other.db:
            return self.index == other.index
        else:
            raise TypeError(f"Comparing InternedString against {type(other)}")

    def __lt__(self, other: 'InternedString') -> bool:
        if self.db is other.db:
            return self.index < other.index
        else:
            raise TypeError(f"Comparing InternedString against {type(other)}")

    def __bool__(self) -> bool:
        # arr = self.db.starts.array
        # return arr[self.index] < arr[self.index + 1]
        return self.index > 0

    def __hash__(self) -> int:
        return hash(self.index)

    def __setattr__(self, _field: str, _value: object) -> None:
        raise TypeError("InternedString is read-only")

    def __delattr__(self, _field: str) -> None:
        raise TypeError("InternedString is read-only")


################################################################################
## On-disk arrays of interned strings

class DiskStringArray:
    _strings: StringCollection
    _array: DiskIntArray

    def __init__(self, path: Path, preload: bool = False) -> None:
        self._array = DiskIntArray(path)
        self._strings = StringCollection(path, preload)

    def raw(self) -> memoryview:
        return self._array.array

    def intern(self, x: bytes) -> InternedString:
        return self._strings.intern(x)

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, i: int) -> InternedString:
        return self._strings.from_index(self._array.array[i])

    def __setitem__(self, i: int, value: bytes) -> None:
        self._array.array[i] = self._strings.intern(value).index

    def __iter__(self) -> Iterator[InternedString]:
        yield from map(self._strings.from_index, self._array.array)

    def __enter__(self) -> 'DiskStringArray':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._array.close()
        self._strings.close()

    def sanity_check(self) -> None:
        self._strings.sanity_check()


    @staticmethod
    def create(path: Path, strings: Iterable[bytes], max_size: int) -> 'DiskStringArray':
        StringCollection.build(path, strings)
        collection = StringCollection(path, preload=True)
        DiskIntArray.create(max_size, path, max_value = len(collection)-1)
        return DiskStringArray(path, preload=True)

