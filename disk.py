"""General-purpose on-disk data structures."""

import os
import sys
import json
import math
from pathlib import Path
import mmap
from array import array
import itertools
from functools import total_ordering
from typing import overload, Dict, BinaryIO, Union, Iterator, Optional, Iterable

def open_readonly_mmap(file : BinaryIO) -> mmap.mmap:
    try:
        return mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
    except TypeError:
        # prot is not available on Windows
        return mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)


def add_suffix(path:Path, suffix:str):
    # Alternatively: Path(path).with_suffix(path.suffix + suffix)
    return Path(str(path) + suffix)


################################################################################
## On-disk arrays of numbers

CONFIG_SUFFIX = '.config.json'

DiskIntArrayType = Union[memoryview, 'SlowDiskIntArray']

def DiskIntArray(path : Path) -> DiskIntArrayType:
    typecodes : Dict[int,str] = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    for n, c in typecodes.items(): assert array(c).itemsize == n

    # path = Path(path)
    file : BinaryIO = open(path, 'rb')
    with open(add_suffix(path, CONFIG_SUFFIX)) as configfile:
        config = json.load(configfile)
    elemsize : int = config['elemsize']
    byteorder : str = config['byteorder']

    arr : mmap.mmap = open_readonly_mmap(file)
    if elemsize in typecodes and byteorder == sys.byteorder:
        return memoryview(arr).cast(typecodes[elemsize])
    else:
        return SlowDiskIntArray(arr, elemsize, byteorder)


class SlowDiskIntArray:
    def __init__(self, array:mmap.mmap, elemsize:int, byteorder:str):
        self._array : mmap.mmap = array
        self._elemsize : int = elemsize
        self._byteorder : str = byteorder
        self._length : int = len(array) // elemsize
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
        start : int = i * self._elemsize
        return int.from_bytes(self._array[start:start+self._elemsize], byteorder=self._byteorder) # type: ignore

    def _slice(self, slice:slice) -> Iterator[int]:
        start, stop, step = slice.indices(len(self))
        array : mmap.mmap = self._array
        elemsize : int = self._elemsize
        byteorder : str = self._byteorder

        start *= elemsize
        stop *= elemsize
        step *= elemsize
        for i in range(start, stop, step):
            yield int.from_bytes(array[i:i+elemsize], byteorder=byteorder) # type: ignore

    def __iter__(self) -> Iterator[int]:
        return self._slice(slice(None))


class DiskIntArrayBuilder:
    def __init__(self, path:Path, max_value:int=0, byteorder:str=sys.byteorder, use_memoryview:bool=False):
        if max_value == 0: max_value = 2**32-1
        self._path : Path = path
        self._byteorder : str = byteorder

        self._elemsize : int = self._min_bytes_to_store_values(max_value)
        if use_memoryview:
            if byteorder != sys.byteorder:
                raise ValueError(f"DiskIntArrayBuilder: memoryview requires byteorder to be '{sys.byteorder}'")
            if self._elemsize == 3: self._elemsize = 4
            if self._elemsize > 4 and self._elemsize <= 8: self._elemsize = 8
            if self._elemsize > 8:
                raise RuntimeError('DiskIntArrayBuilder: memoryview does not support self._elemsize > 8')

        with open(add_suffix(path, CONFIG_SUFFIX), 'w') as configfile:
            json.dump({
                'elemsize': self._elemsize,
                'byteorder': self._byteorder,
            }, configfile)
        self._file : BinaryIO = open(path, 'wb')

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
    def build(path:Path, values:Iterable[int], max_value:int=0, byteorder:str=sys.byteorder, use_memoryview:bool=False):
        if max_value is None:
            values = list(values)
            max_value = max(values)

        builder = DiskIntArrayBuilder(path, max_value=max_value, byteorder=byteorder, use_memoryview=use_memoryview)
        for value in values: 
            builder.append(value)
        builder.close()
        return DiskIntArray(path)


################################################################################
## String interning

STARTS_SUFFIX  = '.starts'

class StringCollection:
    strings : mmap.mmap
    starts : DiskIntArrayType

    def __init__(self, path:Path):
        self._path : Path = path
        stringsfile : BinaryIO = open(path, 'rb')
        self.strings = open_readonly_mmap(stringsfile)
        self.starts = DiskIntArray(add_suffix(path, STARTS_SUFFIX))
        self._intern : Dict[bytes, int] = {}

    def __len__(self) -> int:
        return len(self.starts)-1

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
    def build(path:Path, strings:Iterable[bytes]) -> StringCollection:
        stringlist = sorted(set(strings))

        with open(path, 'wb') as stringsfile:
            for string in stringlist: 
                stringsfile.write(string)

        stringlist.insert(0, b'')  # this is to emulate the 'initial' keyword in accumulate, which was introduced in Python 3.8
        DiskIntArrayBuilder.build(
            add_suffix(path, STARTS_SUFFIX),
            itertools.accumulate((len(s) for s in stringlist)),
            use_memoryview = True
        )

        return StringCollection(path)


@total_ordering
class InternedString:
    __slots__ = ['db', 'index']
    db : StringCollection
    index : int

    def __init__(self, db:StringCollection, index:int):
        object.__setattr__(self, "db", db)
        object.__setattr__(self, "index", index)

    def __bytes__(self) -> bytes:
        start : int = self.db.starts[self.index]
        nextstart : int = self.db.starts[self.index+1]
        return self.db.strings[start:nextstart]

    def __str__(self):
        return bytes(self).decode('utf-8')

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

STRINGS_SUFFIX = '.strings'

class DiskStringArray:
    strings : StringCollection

    def __init__(self, path:Path):
        self._array : DiskIntArrayType = DiskIntArray(path)
        self.strings = StringCollection(add_suffix(path, STRINGS_SUFFIX))

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
    def __init__(self, path:Path, strings:Iterable[bytes], use_memoryview:bool=False):
        self._path : Path = path
        strings_path : Path = add_suffix(path, STRINGS_SUFFIX)
        StringCollectionBuilder.build(strings_path, strings)
        self._strings : StringCollection = StringCollection(strings_path)
        self._strings.preload()
        self._builder : DiskIntArrayBuilder = \
            DiskIntArrayBuilder(path, max_value=len(self._strings)-1, use_memoryview=use_memoryview)

    def append(self, value:bytes):
        self._builder.append(self._strings.intern(value).index)

    # def close(self):
    #     self._builder.close()

    @staticmethod
    def build(path:Path, values:Iterable[bytes], strings:Optional[Iterable[bytes]]=None, use_memoryview:bool=False):
        if strings is None:
            values = strings = list(values)

        builder = DiskStringArrayBuilder(path, strings, use_memoryview)
        for value in values:
            builder.append(value)
        # builder.close()

        return DiskStringArray(path)
