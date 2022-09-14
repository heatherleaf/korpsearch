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
from types import TracebackType
from abc import abstractmethod

from util import ByteOrder, add_suffix


################################################################################
## On-disk arrays of numbers

class DiskIntArray:
    array_suffix = '.ia'
    config_suffix = '.ia-cfg'

    array : Union[memoryview, 'SlowDiskIntArray']

    _file : Optional[BinaryIO]
    _mmap : mmap
    _elemsize : int
    _byteorder : ByteOrder

    def __new__(cls, 
        path : Optional[Path] = None, 
        bytemap : Optional[mmap] = None, 
        elemsize : Optional[int] = None, 
        byteorder : Optional[ByteOrder] = None,
    ):
        if path is None:
            assert bytemap is not None and elemsize is not None and byteorder is not None
            file = None

        else:
            assert bytemap is None and elemsize is None and byteorder is None
            path = add_suffix(path, cls.array_suffix)
            with open(path.with_suffix(cls.config_suffix)) as configfile:
                config = json.load(configfile)
            elemsize = config['elemsize']
            byteorder = config['byteorder']
            file = open(path, 'r+b')
            bytemap = mmap(file.fileno(), 0)

        if elemsize in FastDiskIntArray.typecodes and byteorder == sys.byteorder:
            subclass = FastDiskIntArray
        else:
            subclass = SlowDiskIntArray

        self = super(cls, subclass).__new__(subclass)  # type: ignore
        self._file = file
        self._mmap = bytemap
        self._elemsize = elemsize
        self._byteorder = byteorder
        return self

    @abstractmethod
    def __len__(self) -> int: pass

    @abstractmethod
    def __setitem__(self, i:int, val:int): pass

    @overload
    @abstractmethod
    def __getitem__(self, i:int) -> int: pass
    @overload
    @abstractmethod
    def __getitem__(self, i:slice) -> Union[memoryview, Iterator[int]]: pass
    @abstractmethod
    def __getitem__(self, i): pass  # type: ignore

    @abstractmethod
    def __iter__(self) -> Iterator[int]: pass

    def __enter__(self) -> Union[memoryview, 'SlowDiskIntArray']:
        return self.array

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        if isinstance(self.array, memoryview):
            self.array.release()
        self._mmap.close()
        if self._file is not None:
            self._file.close()


class FastDiskIntArray(DiskIntArray):
    typecodes = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}

    def __init__(self, *args, **kwargs):
        for n, c in self.typecodes.items(): assert array(c).itemsize == n
        assert self._elemsize in self.typecodes and self._byteorder == sys.byteorder
        typecode = self.typecodes[self._elemsize]
        self.array = memoryview(self._mmap).cast(typecode)

    def __len__(self):
        return len(self.array)

    def __setitem__(self, i, val):
        self.array[i] = val

    def __getitem__(self, i):
        return self.array[i]

    def __iter__(self):
        return iter(self.array)


class SlowDiskIntArray(DiskIntArray):
    _length : int

    def __init__(self, *args, **kwargs):
        assert len(self._mmap) % self._elemsize == 0, \
            f"Array length ({len(self._mmap)}) is not divisible by elemsize ({self._elemsize})"
        self._length = len(self._mmap) // self._elemsize
        self.array = self

    def __len__(self):
        return self._length

    def __setitem__(self, i, val):
        elemsize = self._elemsize 
        pos = i * elemsize
        self._mmap[pos : pos+elemsize] = val.to_bytes(length=elemsize, byteorder=self._byteorder)

    @overload
    def __getitem__(self, i:int) -> int: pass
    @overload
    def __getitem__(self, i:slice) -> Iterator[int]: pass
    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._slice(i)
        if not isinstance(i, int):
            raise TypeError("DiskIntArray: invalid array index type")
        if i < 0 or i >= len(self):
            raise IndexError("DiskIntArray: array index out of range")
        start = i * self._elemsize
        return int.from_bytes(self._mmap[start : start+self._elemsize], byteorder=self._byteorder)

    def _slice(self, sl:slice) -> Iterator[int]:
        array = self._mmap
        elemsize = self._elemsize
        byteorder = self._byteorder
        start, stop, step = sl.indices(len(self))
        for i in range(start * elemsize, stop * elemsize, step * elemsize):
            yield int.from_bytes(array[i : i+elemsize], byteorder=byteorder)

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

        self._path = add_suffix(path, DiskIntArray.array_suffix)
        self._file : BinaryIO = open(self._path, 'wb')

        with open(self._path.with_suffix(DiskIntArray.config_suffix), 'w') as configfile:
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

    def __enter__(self) -> 'DiskIntArrayBuilder':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

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

        with DiskIntArrayBuilder(path, max_value=max_value, byteorder=byteorder, use_memoryview=use_memoryview) as builder:
            for value in values: 
                builder.append(value)


################################################################################
## String interning

class StringCollection:
    strings_suffix = '.strings'
    starts_suffix  = '.starts'

    strings : mmap
    starts : Union[memoryview, SlowDiskIntArray]

    _path : Path
    _stringsfile : BinaryIO
    _startsarray : DiskIntArray
    _intern : Dict[bytes, int]

    def __init__(self, path:Path):
        self._path = add_suffix(path, self.strings_suffix)
        self._stringsfile = open(self._path, 'r+b')
        self.strings = mmap(self._stringsfile.fileno(), 0)
        self._startsarray = DiskIntArray(self._path.with_suffix(self.starts_suffix))
        self.starts = self._startsarray.array
        self._intern = {}
        assert self.starts[0] == self.starts[1]

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

    def __enter__(self) -> 'StringCollection':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        self.strings.close()
        self._stringsfile.close()
        self._startsarray.close()


class StringCollectionBuilder:
    @staticmethod
    def build(path:Path, strings:Iterable[bytes]):
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

    def __bool__(self) -> bool:
        return self.db.starts[self.index] < self.db.starts[self.index+1]

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
    _array : DiskIntArray

    def __init__(self, path:Path):
        self._array = DiskIntArray(path)
        self.strings = StringCollection(add_suffix(path, self.strings_suffix))

    def raw(self) -> DiskIntArray:
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

    def __enter__(self) -> 'DiskStringArray':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        self._array.close()
        self.strings.close()


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

    def __enter__(self) -> 'DiskStringArrayBuilder':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        self._builder.close()
        self._strings.close()

    @staticmethod
    def build(path:Path, values:Iterable[bytes], strings:Optional[Iterable[bytes]]=None, use_memoryview:bool=False):
        if strings is None:
            values = strings = list(values)

        with DiskStringArrayBuilder(path, strings, use_memoryview) as builder:
            for value in values:
                builder.append(value)
