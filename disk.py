"""General-purpose on-disk data structures."""

import os
import sys
import json
import math
from pathlib import Path
from mmap import mmap
from array import array
import itertools
from functools import total_ordering
from typing import overload, Dict, BinaryIO, Union, Iterator, Optional, Iterable

from util import ByteOrder, add_suffix


################################################################################
## On-disk arrays of numbers

DiskIntArrayType = Union[memoryview, 'SlowDiskIntArray']

def DiskIntArray(path : Path) -> DiskIntArrayType:
    typecodes : Dict[int,str] = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    for n, c in typecodes.items(): assert array(c).itemsize == n

    path = add_suffix(path, SlowDiskIntArray.array_suffix)
    with open(path.with_suffix(SlowDiskIntArray.config_suffix)) as configfile:
        config = json.load(configfile)
    file : BinaryIO = open(path, 'r+b')
    elemsize : int = config['elemsize']
    byteorder : ByteOrder = config['byteorder']

    arr : mmap = mmap(file.fileno(), 0)
    if elemsize in typecodes and byteorder == sys.byteorder:
        return memoryview(arr).cast(typecodes[elemsize])
    else:
        return SlowDiskIntArray(arr, elemsize, byteorder)


class SlowDiskIntArray:
    array_suffix = '.ia'
    config_suffix = '.ia-cfg'

    _array : mmap
    _elemsize : int
    _byteorder : ByteOrder
    _length : int

    def __init__(self, array:mmap, elemsize:int, byteorder:ByteOrder):
        self._array = array
        self._elemsize = elemsize
        self._byteorder = byteorder
        self._length = len(array) // elemsize
        assert len(array) % elemsize == 0, f"Array length ({len(array)}) is not divisible by elemsize ({elemsize})"

    def __len__(self) -> int:
        return self._length

    @overload
    def __getitem__(self, i:int) -> int: pass
    @overload
    def __getitem__(self, i:slice) -> Iterator[int]: pass
    def __getitem__(self, i:Union[int,slice]):
        if isinstance(i, slice):
            return self._slice(i)
        if not isinstance(i, int):
            raise TypeError("SlowDiskIntArray: invalid array index type")
        if i < 0 or i >= len(self):
            raise IndexError("SlowDiskIntArray: array index out of range")
        start = i * self._elemsize
        return int.from_bytes(self._array[start:start+self._elemsize], byteorder=self._byteorder)

    def _slice(self, slice:slice) -> Iterator[int]:
        start, stop, step = slice.indices(len(self))
        array = self._array
        elemsize = self._elemsize
        byteorder = self._byteorder

        start *= elemsize
        stop *= elemsize
        step *= elemsize
        for i in range(start, stop, step):
            yield int.from_bytes(array[i:i+elemsize], byteorder=byteorder)

    def __iter__(self) -> Iterator[int]:
        return self._slice(slice(None))


class DiskIntArrayBuilder:
    _byteorder : ByteOrder
    _elemsize : int
    _path : Path
    _file : BinaryIO

    def __init__(self, path:Path, max_value:int=0, byteorder:ByteOrder=sys.byteorder, use_memoryview:bool=False):
        self._byteorder = byteorder

        if max_value == 0: max_value = 2**32-1  # default: 4-byte integers
        self._elemsize = self._min_bytes_to_store_values(max_value)

        if use_memoryview:
            if byteorder != sys.byteorder:
                raise ValueError(f"DiskIntArrayBuilder: memoryview requires byteorder to be '{sys.byteorder}'")
            if self._elemsize == 3: self._elemsize = 4
            if self._elemsize > 4 and self._elemsize <= 8: self._elemsize = 8
            if self._elemsize > 8:
                raise RuntimeError('DiskIntArrayBuilder: memoryview does not support self._elemsize > 8')

        self._path = add_suffix(path, SlowDiskIntArray.array_suffix)
        self._file : BinaryIO = open(self._path, 'wb')

        with open(self._path.with_suffix(SlowDiskIntArray.config_suffix), 'w') as configfile:
            json.dump({
                'elemsize': self._elemsize,
                'byteorder': self._byteorder,
            }, configfile)

    def append(self, value:int):
        self._file.write(value.to_bytes(self._elemsize, byteorder=self._byteorder)) # type: ignore

    def __setitem__(self, k:int, value:int):
        self._file.seek(k * self._elemsize)
        self.append(value)
        self._file.seek(0, os.SEEK_END)

    def __len__(self) -> int:
        return self._file.tell() // self._elemsize

    def close(self):
        self._file.close()

    @staticmethod
    def _min_bytes_to_store_values(max_value:int) -> int:
        return math.ceil(math.log(max_value + 1, 2) / 8)

    @staticmethod
    def build(path:Path, values:Iterable[int], max_value:int=0, byteorder:ByteOrder=sys.byteorder, use_memoryview:bool=False):
        if max_value is None:
            values = list(values)
            max_value = max(values)

        builder = DiskIntArrayBuilder(path, max_value=max_value, byteorder=byteorder, use_memoryview=use_memoryview)
        for value in values: 
            builder.append(value)
        builder.close()


################################################################################
## String interning

class StringCollection:
    strings_suffix = '.strings'
    starts_suffix  = '.starts'

    strings : mmap

    _path : Path
    _stringsfile : BinaryIO
    _startsarray : DiskIntArrayType
    _intern : Dict[bytes, int]

    def __init__(self, path:Path):
        self._path = add_suffix(path, self.strings_suffix)
        self._stringsfile = open(self._path, 'r+b')
        self.strings = mmap(self._stringsfile.fileno(), 0)
        self._startsarray = DiskIntArray(self._path.with_suffix(self.starts_suffix))
        self._intern = {}

    def __len__(self) -> int:
        return len(self._startsarray)-1

    def from_index(self, index:int) -> 'InternedString':
        return InternedString(self, index)

    def preload(self):
        if not self._intern:
            self._intern = {}
            for i in range(len(self)):
                self._intern[bytes(self.from_index(i))] = i

    def intern(self, string:Union[bytes,'InternedString']) -> 'InternedString':
        if isinstance(string, InternedString):
            assert string.db is self
            return string

        if self._intern:
            return self.from_index(self._intern[string])

        lo : int = 0
        hi : int = len(self)-1
        while lo <= hi:
            mid : int = (lo + hi) // 2
            here : bytes = bytes(self.from_index(mid))
            if string < here:
                hi = mid-1
            elif string > here:
                lo = mid+1
            else:
                return self.from_index(mid)
        raise KeyError(f"StringCollection: string '{str(string)}' not found in database")


class StringCollectionBuilder:
    @staticmethod
    def build(path:Path, strings:Iterable[bytes]):
        stringlist = sorted(set(strings))

        path = add_suffix(path, StringCollection.strings_suffix)
        with open(path, 'wb') as stringsfile:
            for string in stringlist: 
                stringsfile.write(string)

        stringlist.insert(0, b'')  # this is to emulate the 'initial' keyword in accumulate, which was introduced in Python 3.8
        DiskIntArrayBuilder.build(
            path.with_suffix(StringCollection.starts_suffix),
            itertools.accumulate((len(s) for s in stringlist)),
            use_memoryview = True
        )


@total_ordering
class InternedString:
    __slots__ = ['db', 'index']
    db : StringCollection
    index : int

    def __init__(self, db:StringCollection, index:int):
        object.__setattr__(self, "db", db)
        object.__setattr__(self, "index", index)

    def __bytes__(self) -> bytes:
        start : int = self.db._startsarray[self.index]
        nextstart : int = self.db._startsarray[self.index+1]
        return self.db.strings[start:nextstart]

    def __str__(self):
        return bytes(self).decode()

    def __repr__(self):
        return f"InternedString({self})"

    def __eq__(self, other:object) -> bool:
        if isinstance(other, InternedString) and self.db is other.db:
            return self.index == other.index
        else:
            raise TypeError(f"Comparing InternedString against {type(other)}")

    def __lt__(self, other:'InternedString') -> bool:
        if isinstance(other, InternedString) and self.db is other.db:
            return self.index < other.index
        else:
            raise TypeError(f"Comparing InternedString against {type(other)}")

    def __hash__(self) -> int:
        return hash(self.index)

    def __setattr__(self, _field:str, _value:object):
        raise TypeError("InternedString is read-only")

    def __delattr__(self, _field:str):
        raise TypeError("InternedString is read-only")


################################################################################
## On-disk arrays of interned strings

class DiskStringArray:
    strings_suffix = '.sa'

    strings : StringCollection
    _array : DiskIntArrayType

    def __init__(self, path:Path):
        self._array = DiskIntArray(path)
        self.strings = StringCollection(add_suffix(path, self.strings_suffix))

    def raw(self) -> DiskIntArrayType:
        return self._array

    def intern(self, x:bytes) -> InternedString:
        return self.strings.intern(x)

    def __len__(self):
        return len(self._array)

    @overload
    def __getitem__(self, i:int) -> InternedString: pass
    @overload
    def __getitem__(self, i:slice) -> Iterator[InternedString]: pass
    def __getitem__(self, i:Union[int,slice]):
        if isinstance(i, slice):
            return self._slice(i)
        if not isinstance(i, int):
            raise TypeError("invalid array index type")
        return self.strings.from_index(self._array[i])

    def _slice(self, slice:slice) -> Iterator[InternedString]:
        return map(self.strings.from_index, self._array[slice])

    def __iter__(self) -> Iterator[InternedString]:
        yield from self._slice(slice(None))


class DiskStringArrayBuilder:
    _path : Path
    _strings : StringCollection
    _builder : DiskIntArrayBuilder

    def __init__(self, path:Path, strings:Iterable[bytes], use_memoryview:bool=False):
        self._path = path
        strings_path : Path = add_suffix(path, DiskStringArray.strings_suffix)
        StringCollectionBuilder.build(strings_path, strings)
        self._strings = StringCollection(strings_path)
        self._strings.preload()
        self._builder = DiskIntArrayBuilder(path, max_value=len(self._strings)-1, use_memoryview=use_memoryview)

    def append(self, value:bytes):
        self._builder.append(self._strings.intern(value).index)

    def close(self):
        self._builder.close()

    @staticmethod
    def build(path:Path, values:Iterable[bytes], strings:Optional[Iterable[bytes]]=None, use_memoryview:bool=False):
        if strings is None:
            values = strings = list(values)

        builder = DiskStringArrayBuilder(path, strings, use_memoryview)
        for value in values:
            builder.append(value)
        builder.close()
