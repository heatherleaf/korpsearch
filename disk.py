"""General-purpose on-disk data structures."""

import os
import sys
import json
from pathlib import Path
from mmap import mmap
from array import array
import itertools
from functools import total_ordering
from typing import overload, BinaryIO, Optional, Union
from collections.abc import Iterator, Iterable, Sequence, MutableSequence

from util import ByteOrder, add_suffix, min_bytes_to_store_values


################################################################################
## On-disk arrays of numbers

class DiskIntArray(Sequence[int]):
    typecodes = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}

    array_suffix = '.ia'
    config_suffix = '.ia-cfg'

    array: Union[memoryview, 'DiskIntArray']
    path: Optional[Path]

    _file: BinaryIO
    _mmap: Union[mmap, bytearray, bytes]
    _mview: memoryview
    _length: int
    _elemsize: int
    _byteorder: ByteOrder

    _append_ptr: int

    def __init__(self, path: Path) -> None:
        for n, c in self.typecodes.items(): assert array(c).itemsize == n

        self.path = self.getpath(path)
        with open(self.getconfig(self.path)) as configfile:
            config = json.load(configfile)
        self._file = open(self.path, 'r+b')
        self._elemsize = config['elemsize']
        self._byteorder = config['byteorder']
        try:
            self._mmap = mmap(self._file.fileno(), 0)
        except ValueError:  # "cannot mmap an empty file"
            self._mmap = bytearray(b'')

        self._length = len(self._mmap) // self._elemsize
        assert len(self._mmap) % self._elemsize == 0, \
            f"Array length ({len(self._mmap)}) is not divisible by elemsize ({self._elemsize})"

        if self._elemsize in self.typecodes and self._byteorder == sys.byteorder:
            typecode = self.typecodes[self._elemsize]
            self._mview = memoryview(self._mmap).cast(typecode)
            self.array = self._mview
        else:
            self.array = self

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, index: int) -> int: pass
    @overload
    def __getitem__(self, index: slice) -> Union[memoryview, Sequence[int]]: pass
    def __getitem__(self, index: Union[int,slice]) -> Union[int, memoryview, Sequence[int]]:
        try:
            return self._mview[index]
        except AttributeError:
            pass
        if isinstance(index, slice):
            return self._slice(index)  # type: ignore
        start = index * self._elemsize
        return int.from_bytes(self._mmap[start : start+self._elemsize], byteorder=self._byteorder)

    def _slice(self, sl: slice) -> Iterator[int]:
        array = self._mmap
        elemsize = self._elemsize
        byteorder = self._byteorder
        start, stop, step = sl.indices(len(self))
        for i in range(start * elemsize, stop * elemsize, step * elemsize):
            yield int.from_bytes(array[i : i+elemsize], byteorder=byteorder)

    def __iter__(self) -> Iterator[int]:
        try:
            return iter(self._mview)
        except AttributeError:
            return self._slice(slice(None))

    def reset_append(self) -> None:
        self._append_ptr = 0

    def append(self, value: int) -> None:
        try:
            self._mview[self._append_ptr] = value
        except AttributeError:
            if isinstance(self._mmap, bytes):
                raise TypeError("Cannot append to a bytestring")
            i = self._append_ptr * self._elemsize
            self._mmap[i : i+self._elemsize] = value.to_bytes(self._elemsize, byteorder=self._byteorder)
        self._append_ptr += 1

    def extend(self, values: Iterator[int]) -> None:
        for val in values:
            self.append(val)

    def truncate_append(self) -> None:
        self._file.truncate(self._append_ptr * self._elemsize)
        try:
            self._mview.release()
        except AttributeError:
            pass
        try:
            self._mmap.close()  # type: ignore
        except AttributeError:
            pass
        try:
            self._mmap = mmap(self._file.fileno(), 0)
        except ValueError:  # "cannot mmap an empty file"
            self._mmap = bytearray(b'')
        self._length = len(self._mmap) // self._elemsize
        if self._elemsize in self.typecodes and self._byteorder == sys.byteorder:
            typecode = self.typecodes[self._elemsize]
            self._mview = memoryview(self._mmap).cast(typecode)
            self.array = self._mview
        else:
            self.array = self

    def __enter__(self) -> Union[memoryview, 'DiskIntArray']:
        return self.array

    def __exit__(self, *_) -> None:
        self.close()

    def close(self):
        try:
            self._mview.release()
        except AttributeError:
            pass
        try:
            self._mmap.close()  # type: ignore
        except AttributeError:
            pass
        try:
            self._file.close()
        except AttributeError:
            pass

    @classmethod
    def getpath(cls, path:Path) -> Path:
        return add_suffix(path, cls.array_suffix)

    @classmethod
    def getconfig(cls, path:Path) -> Path:
        return cls.getpath(path).with_suffix(cls.config_suffix)



class LowlevelIntArray(DiskIntArray):
    def __init__(self, bytemap: bytes, elemsize: int, byteorder: ByteOrder) -> None:
        self.path = None
        self._elemsize = elemsize
        self._byteorder = byteorder
        self._mmap = bytemap
        self._length = len(self._mmap) // self._elemsize
        assert len(self._mmap) % self._elemsize == 0, \
            f"Array length ({len(self._mmap)}) is not divisible by elemsize ({self._elemsize})"
        if self._elemsize in self.typecodes and self._byteorder == sys.byteorder:
            typecode = self.typecodes[self._elemsize]
            self._mview = memoryview(self._mmap).cast(typecode)
            self.array = self._mview
        else:
            self.array = self



class DiskIntArrayBuilder:
    path: Path

    _byteorder: ByteOrder
    _elemsize: int
    _file: BinaryIO

    def __init__(self, path: Path, max_value: int = 0, 
                 byteorder: ByteOrder = sys.byteorder, use_memoryview: bool = False) -> None:
        self._byteorder = byteorder

        if max_value == 0: max_value = 2**32-1  # default: 4-byte integers
        self._elemsize = min_bytes_to_store_values(max_value)

        if use_memoryview:
            if byteorder != sys.byteorder:
                raise ValueError(f"DiskIntArrayBuilder: memoryview requires byteorder to be '{sys.byteorder}'")
            if self._elemsize == 3: self._elemsize = 4
            if self._elemsize > 4 and self._elemsize <= 8: self._elemsize = 8
            if self._elemsize > 8:
                raise ValueError('DiskIntArrayBuilder: memoryview does not support self._elemsize > 8')

        self.path = DiskIntArray.getpath(path)
        self._file = open(self.path, 'wb')

        with open(DiskIntArray.getconfig(self.path), 'w') as configfile:
            json.dump({
                'elemsize': self._elemsize,
                'byteorder': self._byteorder,
            }, configfile)

    def append(self, value: int) -> None:
        self._file.write(value.to_bytes(self._elemsize, byteorder=self._byteorder))

    def extend(self, values: Iterator[int]) -> None:
        for val in values:
            self.append(val)

    def __setitem__(self, index: int, value: int):
        self._file.seek(index * self._elemsize)
        self.append(value)
        self._file.seek(0, os.SEEK_END)

    def __len__(self) -> int:
        return self._file.tell() // self._elemsize

    def __enter__(self) -> 'DiskIntArrayBuilder':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    @staticmethod
    def build(path: Path, values: Iterable[int], max_value: int = 0, 
              byteorder: ByteOrder = sys.byteorder, use_memoryview: bool = False) -> None:
        if not max_value:
            values = list(values)
            max_value = max(values)

        with DiskIntArrayBuilder(path, max_value=max_value, byteorder=byteorder, use_memoryview=use_memoryview) as builder:
            for value in values: 
                builder.append(value)


################################################################################
## On-disk array of fixed-width byte sequences

class DiskFixedBytesArray(MutableSequence[bytes]):
    _file: BinaryIO
    _mmap: mmap
    _elemsize: int
    _len: int

    def __init__(self, path: Path, elemsize: int) -> None:
        self._file = open(path, 'r+b')
        self._mmap = mmap(self._file.fileno(), 0)
        self._elemsize = elemsize
        self._len = len(self._mmap) // self._elemsize
        assert len(self._mmap) % self._elemsize == 0, \
            f"Array length ({len(self._mmap)}) is not divisible by elemsize ({self._elemsize})"

    def __len__(self) -> int:
        return self._len

    def __setitem__(self, index: Union[int,slice], value: Union[bytes,Iterable[bytes]]) -> None:
        assert isinstance(index, int) and isinstance(value, bytes), "Cannot handle slices"
        elemsize = self._elemsize
        start = index * elemsize
        assert len(value) == elemsize
        self._mmap[start : start+elemsize] = value

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
        elemsize = self._elemsize
        start = index * elemsize
        return self._mmap[start : start+elemsize]

    def _slice(self, sl: slice) -> Iterator[bytes]:
        array = self._mmap
        elemsize = self._elemsize
        start, stop, step = sl.indices(len(self))
        for i in range(start * elemsize, stop * elemsize, step * elemsize):
            yield array[i : i+elemsize]

    def __iter__(self) -> Iterator[bytes]:
        return self._slice(slice(None))

    def __enter__(self) -> 'DiskFixedBytesArray':
        return self

    def __exit__(self, *_) -> None:
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

    def __init__(self, path: Path):
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

    def __exit__(self, *_) -> None:
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
            itertools.accumulate((len(s) for s in stringlist), initial=0),
            use_memoryview = True
        )


@total_ordering
class InternedString:
    __slots__ = ['db', 'index']
    db: StringCollection
    index: int

    def __init__(self, db:StringCollection, index:int):
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

    def _slice(self, slice:slice) -> Iterator[InternedString]:
        return map(self.strings.from_index, self._array[slice])

    def __iter__(self) -> Iterator[InternedString]:
        yield from self._slice(slice(None))

    def __enter__(self) -> 'DiskStringArray':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._array.close()
        self.strings.close()


class DiskStringArrayBuilder:
    _path: Path
    _strings: StringCollection
    _builder: DiskIntArrayBuilder

    def __init__(self, path: Path, strings: Iterable[bytes], use_memoryview: bool = False) -> None:
        self._path = path
        strings_path: Path = add_suffix(path, DiskStringArray.strings_suffix)
        StringCollectionBuilder.build(strings_path, strings)
        self._strings = StringCollection(strings_path)
        self._strings.preload()
        self._builder = DiskIntArrayBuilder(path, max_value=len(self._strings)-1, use_memoryview=use_memoryview)

    def append(self, value: bytes) -> None:
        self._builder.append(self._strings.intern(value).index)

    def __enter__(self) -> 'DiskStringArrayBuilder':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._builder.close()
        self._strings.close()

    @staticmethod
    def build(path: Path, values: Iterable[bytes], strings: Optional[Iterable[bytes]] = None, use_memoryview: bool = False):
        if strings is None:
            values = strings = list(values)

        with DiskStringArrayBuilder(path, strings, use_memoryview) as builder:
            for value in values:
                builder.append(value)
