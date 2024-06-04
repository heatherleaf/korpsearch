"""General-purpose on-disk data structures."""

import os
import sys
import json
from pathlib import Path
from mmap import mmap
# from array import array
import itertools
from functools import total_ordering
from typing import overload, BinaryIO, Optional, Union, Any
from collections.abc import Iterator, Iterable, Sequence, MutableSequence

from util import add_suffix, get_integer_size, get_typecode


################################################################################
## On-disk arrays of numbers

class DiskIntArray(Sequence[int]):
    array_suffix = '.ia'
    config_suffix = '.ia-cfg'

    array: memoryview
    path: Optional[Path]
    itemsize: int

    _file: BinaryIO
    _bytearray: Union[mmap, bytearray, bytes]
    _length: int

    _append_ptr: int

    def __init__(self, source: Union[Path, bytes, bytearray], itemsize: int = 4) -> None:
        if isinstance(source, (bytes, bytearray)):
            self.path = None
            self._bytearray = source
            self.itemsize = itemsize
        else:
            self.path = self.getpath(source)
            self._file = open(self.path, 'r+b')
            with open(self.getconfig(self.path)) as configfile:
                config = json.load(configfile)
            self.itemsize = config['itemsize']
            assert config['byteorder'] == sys.byteorder, f"Cannot handle byteorder {config['byteorder']}"
            try:
                self._bytearray = mmap(self._file.fileno(), 0)
            except ValueError:  # "cannot mmap an empty file"
                self._bytearray = bytearray(b'')
        self._length = len(self._bytearray) // self.itemsize
        assert len(self._bytearray) % self.itemsize == 0, \
            f"Array length ({len(self._bytearray)}) is not divisible by itemsize ({self.itemsize})"
        self.array = memoryview(self._bytearray).cast(get_typecode(self.itemsize))

    def __len__(self) -> int:
        return len(self.array)

    @overload
    def __getitem__(self, index: int) -> int: pass
    @overload
    def __getitem__(self, index: slice) -> memoryview: pass
    def __getitem__(self, index: Union[int,slice]) -> Union[int, memoryview]:
        return self.array[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self.array)

    def reset_append(self) -> None:
        self._append_ptr = 0

    def append(self, value: int) -> None:
        self.array[self._append_ptr] = value
        self._append_ptr += 1

    def extend(self, values: Iterator[int]) -> None:
        for val in values:
            self.append(val)

    def truncate_append(self) -> None:
        try:
            self._file.truncate(self._append_ptr * self.itemsize)
        except AttributeError:
            return
        self.array.release()
        try:
            self._bytearray.close()  # type: ignore
        except AttributeError:
            pass
        try:
            self._bytearray = mmap(self._file.fileno(), 0)
        except ValueError:  # "cannot mmap an empty file"
            self._bytearray = bytearray(b'')
        self.array = memoryview(self._bytearray).cast(get_typecode(self.itemsize))

    def __enter__(self) -> memoryview:
        return self.array

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.array.release()
        try:
            self._bytearray.close()  # type: ignore
        except AttributeError:
            pass
        try:
            self._file.close()
        except AttributeError:
            pass

    @classmethod
    def getpath(cls, path: Path) -> Path:
        return add_suffix(path, cls.array_suffix)

    @classmethod
    def getconfig(cls, path: Path) -> Path:
        return cls.getpath(path).with_suffix(cls.config_suffix)


class DiskIntArrayBuilder:
    path: Path
    itemsize: int
    _file: BinaryIO

    def __init__(self, path: Path, max_value: int = 0) -> None:
        self.path = DiskIntArray.getpath(path)
        self._file = open(self.path, 'wb')
        if max_value > 0: 
            self.itemsize = get_integer_size(max_value)
        else:
            self.itemsize = 4  # default: 4-byte integers

        with open(DiskIntArray.getconfig(self.path), 'w') as configfile:
            json.dump({
                'itemsize': self.itemsize,
                'byteorder': sys.byteorder,
            }, configfile)

    def append(self, value: int) -> None:
        self._file.write(value.to_bytes(self.itemsize, byteorder=sys.byteorder))

    def extend(self, values: Iterator[int]) -> None:
        for val in values:
            self.append(val)

    def __setitem__(self, index: int, value: int) -> None:
        self._file.seek(index * self.itemsize)
        self.append(value)
        self._file.seek(0, os.SEEK_END)

    def __len__(self) -> int:
        return self._file.tell() // self.itemsize

    def __enter__(self) -> 'DiskIntArrayBuilder':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    @staticmethod
    def build(path: Path, values: Iterable[int], max_value: int = 0) -> None:
        if not max_value:
            values = list(values)
            max_value = max(values)
        with DiskIntArrayBuilder(path, max_value) as builder:
            for value in values: 
                builder.append(value)


################################################################################
## On-disk array of fixed-width byte sequences

class DiskFixedBytesArray(MutableSequence[bytes]):
    itemsize: int
    _file: BinaryIO
    _mmap: mmap
    _len: int

    def __init__(self, path: Path, itemsize: int) -> None:
        self._file = open(path, 'r+b')
        self._mmap = mmap(self._file.fileno(), 0)
        self.itemsize = itemsize
        self._len = len(self._mmap) // self.itemsize
        assert len(self._mmap) % self.itemsize == 0, \
            f"Array length ({len(self._mmap)}) is not divisible by itemsize ({self.itemsize})"

    def __len__(self) -> int:
        return self._len

    def __setitem__(self, index: Union[int,slice], value: Union[bytes,Iterable[bytes]]) -> None:
        assert isinstance(index, int) and isinstance(value, bytes), "Cannot handle slices"
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
        self._file.close()


################################################################################
## String interning

class StringCollection:
    strings_suffix = '.strings'
    starts_suffix  = '.starts'

    strings: mmap
    starts: Union[memoryview, DiskIntArray]

    _path: Path
    _stringsfile: BinaryIO
    _startsarray: DiskIntArray
    _intern: dict[bytes, int]

    def __init__(self, path: Path) -> None:
        self._path = add_suffix(path, self.strings_suffix)
        self._stringsfile = open(self._path, 'r+b')
        self.strings = mmap(self._stringsfile.fileno(), 0)
        self._startsarray = DiskIntArray(self._path.with_suffix(self.starts_suffix))
        self.starts = self._startsarray.array
        self._intern = {}
        assert self.starts[0] == self.starts[1]

    def __len__(self) -> int:
        return len(self.starts)-1

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
            return self.from_index(self._intern[string])

        lo = 0
        hi = len(self)-1
        while lo <= hi:
            mid = (lo + hi) // 2
            here = bytes(self.from_index(mid))
            if string < here:
                hi = mid-1
            elif string > here:
                lo = mid+1
            else:
                return self.from_index(mid)
        raise KeyError(f"StringCollection: string '{str(string)}' not found in database")

    def __enter__(self) -> 'StringCollection':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.strings.close()
        self._stringsfile.close()
        self._startsarray.close()


class StringCollectionBuilder:
    @staticmethod
    def build(path: Path, strings: Iterable[bytes]) -> None:
        stringset = set(strings)
        stringset.add(b'')
        stringlist = sorted(stringset)
        assert not stringlist[0]

        path = add_suffix(path, StringCollection.strings_suffix)
        with open(path, 'wb') as stringsfile:
            for string in stringlist: 
                stringsfile.write(string)

        DiskIntArrayBuilder.build(
            path.with_suffix(StringCollection.starts_suffix),
            itertools.accumulate((len(s) for s in stringlist), initial=0)
        )


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
        start = self.db.starts[self.index]
        nextstart = self.db.starts[self.index+1]
        return self.db.strings[start:nextstart]

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
        return self.db.starts[self.index] < self.db.starts[self.index+1]

    def __hash__(self) -> int:
        return hash(self.index)

    def __setattr__(self, _field: str, _value: object) -> None:
        raise TypeError("InternedString is read-only")

    def __delattr__(self, _field: str) -> None:
        raise TypeError("InternedString is read-only")


################################################################################
## On-disk arrays of interned strings

class DiskStringArray(Sequence[InternedString]):
    strings_suffix = '.sa'

    strings: StringCollection
    _array: DiskIntArray

    def __init__(self, path: Path) -> None:
        self._array = DiskIntArray(path)
        self.strings = StringCollection(add_suffix(path, self.strings_suffix))

    def raw(self) -> DiskIntArray:
        return self._array

    def intern(self, x: bytes) -> InternedString:
        return self.strings.intern(x)

    def __len__(self) -> int:
        return len(self._array)

    @overload
    def __getitem__(self, i: int) -> InternedString: pass
    @overload
    def __getitem__(self, i: slice) -> Sequence[InternedString]: pass
    def __getitem__(self, i: Union[int,slice]) -> Union[InternedString, Sequence[InternedString]]:
        if isinstance(i, slice):
            return self._slice(i)  # type: ignore
        return self.strings.from_index(self._array[i])

    def _slice(self, slice: slice) -> Iterator[InternedString]:
        return map(self.strings.from_index, self._array[slice])

    def __iter__(self) -> Iterator[InternedString]:
        yield from self._slice(slice(None))

    def __enter__(self) -> 'DiskStringArray':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._array.close()
        self.strings.close()


class DiskStringArrayBuilder:
    _path: Path
    _strings: StringCollection
    _builder: DiskIntArrayBuilder

    def __init__(self, path: Path, strings: Iterable[bytes]) -> None:
        self._path = path
        strings_path: Path = add_suffix(path, DiskStringArray.strings_suffix)
        StringCollectionBuilder.build(strings_path, strings)
        self._strings = StringCollection(strings_path)
        self._strings.preload()
        self._builder = DiskIntArrayBuilder(path, max_value=len(self._strings)-1)

    def append(self, value: bytes) -> None:
        self._builder.append(self._strings.intern(value).index)

    def __enter__(self) -> 'DiskStringArrayBuilder':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._builder.close()
        self._strings.close()

    @staticmethod
    def build(path: Path, values: Iterable[bytes], strings: Optional[Iterable[bytes]] = None) -> None:
        if strings is None:
            values = strings = list(values)

        with DiskStringArrayBuilder(path, strings) as builder:
            for value in values:
                builder.append(value)
