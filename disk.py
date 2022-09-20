"""General-purpose on-disk data structures."""

import os
import sys
import json
from pathlib import Path
from mmap import mmap
from array import array
import itertools
from functools import total_ordering
from typing import overload, Dict, BinaryIO, Union, Iterator, Optional, Iterable, Sequence, MutableSequence
from types import TracebackType
from abc import abstractmethod

from util import ByteOrder, add_suffix, min_bytes_to_store_values


################################################################################
## On-disk arrays of numbers


class DiskIntArray(Sequence):
    typecodes = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}

    array_suffix = '.ia'
    config_suffix = '.ia-cfg'

    array : Union[memoryview, 'DiskIntArray']

    _file : BinaryIO
    _mmap : mmap
    _mview : memoryview
    _length : int
    _elemsize : int
    _byteorder : ByteOrder

    def __init__(self, 
        path : Optional[Path] = None, 
        bytemap : Optional[mmap] = None, 
        elemsize : Optional[int] = None, 
        byteorder : Optional[ByteOrder] = None,
    ):
        for n, c in self.typecodes.items(): assert array(c).itemsize == n

        if path is None:
            assert bytemap is not None and elemsize is not None and byteorder is not None
            self._elemsize = elemsize
            self._byteorder = byteorder
            self._mmap = bytemap

        else:
            assert bytemap is None and elemsize is None and byteorder is None
            path = add_suffix(path, self.array_suffix)
            with open(path.with_suffix(self.config_suffix)) as configfile:
                config = json.load(configfile)
            self._file = open(path, 'r+b')
            self._elemsize = config['elemsize']
            self._byteorder = config['byteorder']
            self._mmap = mmap(self._file.fileno(), 0)

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
    def __getitem__(self, i:int) -> int: pass
    @overload
    def __getitem__(self, i:slice) -> Union[memoryview, Iterator[int]]: pass
    def __getitem__(self, i): # type: ignore
        try:
            return self._mview[i]
        except AttributeError:
            pass
        if isinstance(i, slice):
            return self._slice(i)
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
        try:
            return iter(self._mview)
        except AttributeError:
            return self._slice(slice(None))

    def __enter__(self) -> Union[memoryview, 'DiskIntArray']:
        return self.array

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        try:
            self._mview.release()
        except AttributeError:
            pass
        self._mmap.close()
        try:
            self._file.close()
        except AttributeError:
            pass


class DiskIntArrayBuilder:
    _byteorder : ByteOrder
    _elemsize : int
    _path : Path
    _file : BinaryIO

    def __init__(self, path:Path, max_value:int=0, byteorder:ByteOrder=sys.byteorder, use_memoryview:bool=False):
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
    def build(path:Path, values:Iterable[int], max_value:int=0, byteorder:ByteOrder=sys.byteorder, use_memoryview:bool=False):
        if max_value is None:
            values = list(values)
            max_value = max(values)

        with DiskIntArrayBuilder(path, max_value=max_value, byteorder=byteorder, use_memoryview=use_memoryview) as builder:
            for value in values: 
                builder.append(value)


################################################################################
## On-disk array of fixed-width byte sequences

class DiskFixedBytesArray(MutableSequence):
    _file : BinaryIO
    _mmap : mmap
    _elemsize : int
    _len : int

    def __init__(self, path:Path, elemsize:int):
        self._file = open(path, 'r+b')
        self._mmap = mmap(self._file.fileno(), 0)
        self._elemsize = elemsize
        self._len = len(self._mmap) // self._elemsize
        assert len(self._mmap) % self._elemsize == 0, \
            f"Array length ({len(self._mmap)}) is not divisible by elemsize ({self._elemsize})"

    def __len__(self) -> int:
        return self._len

    def __setitem__(self, i:int, val:bytes):
        elemsize = self._elemsize
        start = i * elemsize
        assert len(val) == elemsize
        self._mmap[start : start+elemsize] = val

    def __delitem__(self, i:int):
        # Required to be a MutableSequence, but mmap arrays cannot change size
        raise NotImplementedError

    def insert(self, i:int, val:bytes):
        # Required to be a MutableSequence, but mmap arrays cannot change size
        raise NotImplementedError

    @overload
    def __getitem__(self, i:int) -> bytes: pass
    @overload
    def __getitem__(self, i:slice) -> Iterator[bytes]: pass
    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._slice(i)
        elemsize = self._elemsize
        start = i * elemsize
        return self._mmap[start : start+elemsize]

    def _slice(self, sl:slice) -> Iterator[bytes]:
        array = self._mmap
        elemsize = self._elemsize
        start, stop, step = sl.indices(len(self))
        for i in range(start * elemsize, stop * elemsize, step * elemsize):
            yield array[i : i+elemsize]

    def __iter__(self) -> Iterator[bytes]:
        return self._slice(slice(None))

    def __enter__(self) -> 'DiskFixedBytesArray':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        self._mmap.close()
        self._file.close()


################################################################################
## String interning

class StringCollection:
    strings_suffix = '.strings'
    starts_suffix  = '.starts'

    strings : mmap
    starts : Union[memoryview, DiskIntArray]

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

class DiskStringArray(Sequence):
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
