
from pathlib import Path
from argparse import Namespace
from typing import BinaryIO
from mmap import mmap
from abc import abstractmethod
import json
import logging

from pyroaring import BitMap

from disk import IntArray, Symbol, IntBytesMap
from corpus import Corpus
from index import Template, TemplateLiteral, Index, UnaryIndex, Instance
import sort
from util import progress_bar

# Possible sorting alternatives, the first is the default:
SORTER_CHOICES = ['roaring', 'tmpfile', 'internal', 'cython']


def build_index(corpus: Corpus, template: Template, args: Namespace) -> None:
    index_path = Index.indexpath(corpus, template)
    index_path.parent.mkdir(exist_ok=True)
    if len(template) == 1:
        build_unary_index(corpus, index_path, template, args)
    elif len(template) == 2:
        build_binary_index(corpus, index_path, template, args)
    else:
        raise ValueError(f"Cannot build indexes of length {len(template)}: {template}")
    logging.info(f"Built index for {template}")



def build_unary_index(corpus: Corpus, index_path: Path, template: Template, args: Namespace) -> None:
    logging.info(f"Building unary index: {template}, using sorter {args.sorter}")

    assert len(template) == 1, f"UnaryIndex templates must have length 1: {template}"
    assert not template.literals, f"Cannot build UnaryIndex from templates with literals: {template}"

    tmpl = template.template[0]
    assert tmpl.offset == template.min_offset() == template.max_offset() == 0
    features = corpus.tokens[tmpl.feature].raw()

    collector: Collector
    if args.sorter == 'internal':
        collector = ListCollector(2)
    elif args.sorter == 'roaring':
        collector = RoaringCollector()
    elif args.sorter == 'cython':
        collector = CythonCollector(2, index_path.parent / 'cindex.tmp', args)
    else:
        collector = TmpfileCollector(2, index_path.parent / 'index.tmp', args)

    index_size = 0
    for pos, value in enumerate(progress_bar(features, desc="Collecting positions")):
        if value:
            collector.append2(value, pos)
            index_size += 1

    collector.finalise(index_path, index_size)
    with open(Index.getconfigpath(index_path), "w") as configfile:
        print(json.dumps({
            'size': index_size,
        }), file=configfile)


def build_binary_index(corpus: Corpus, index_path: Path, template: Template, args: Namespace) -> None:
    logging.info(f"Building binary index: {template}, using sorter {args.sorter}")

    assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
    assert all(lit.negative for lit in template.literals), f"Cannot handle positive template literals: {template}"

    tmpl1, tmpl2 = template.template
    assert tmpl1.offset == template.min_offset() == 0
    assert tmpl2.offset == template.max_offset()
    features1 = corpus.tokens[tmpl1.feature].raw()
    features2 = corpus.tokens[tmpl2.feature].raw()
    min_freq = args.min_frequency

    unary1 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl1.feature)]))
    unary2 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl2.feature)]))

    # Optimisation: cache index lookups, because the size of the vocabulary
    # is much smaller than the corpus size.
    cache1: dict[int, bool] = {}
    cache2: dict[int, bool] = {}
    def unary_min_frequency(unary: UnaryIndex, unary_key: Symbol, cache: dict[int, bool]) -> bool:
        if unary_key not in cache:
            size = len(unary.search(Instance([unary_key])))
            cache[unary_key] = size >= min_freq
        return cache[unary_key]

    collector: Collector
    if args.sorter == 'internal':
        collector = ListCollector(3)
    elif args.sorter == 'roaring':
        collector = RoaringCollector()
    elif args.sorter == 'cython':
        collector = CythonCollector(3, index_path.parent / 'cindex.tmp', args)
    else:
        collector = TmpfileCollector(3, index_path.parent / 'index.tmp', args)

    # Optimisation: use the underlying memoryview for each SymbolArray,
    # and don't bother with looking up symbols - use the ints directly instead.
    test_literals = [(corpus.tokens[lit.feature].raw(), lit.offset, lit.value) for lit in template.literals]

    index_size = 0
    skipped_instances = 0
    for pos in progress_bar(range(len(corpus) - template.max_offset()), desc="Collecting positions"):
        val1, val2 = features1[pos], features2[pos + tmpl2.offset]
        if val1 and val2 and all(lfeat[pos + loffset] != lval for (lfeat, loffset, lval) in test_literals):
            if min_freq and not (
                        unary_min_frequency(unary1, val1, cache1) and
                        unary_min_frequency(unary2, val2, cache2)
                    ):
                skipped_instances += 1
            else:
                collector.append3(val1, val2, pos)
                index_size += 1
    if skipped_instances:
        logging.debug(f"Skipped {skipped_instances} low-frequency instances")

    collector.finalise(index_path, index_size)
    with open(Index.getconfigpath(index_path), "w") as configfile:
        print(json.dumps({
            'size': index_size,
            'min_frequency': min_freq,
            'skipped_instances': skipped_instances,
        }), file=configfile)


###############################################################################
# Special classes for collecting, sorting and building the final index
#  - ListCollector collects the intermediate result in a Python list
#  - TmpfileCollector stores the intermediate result in a temporary binary file
#  - FasterCollector calls C functions (via Cython) for the sorting
#
# Note for the constants 4 and 32 below: we assume 32-bit (4-byte) unsigned integers

class Collector:
    @abstractmethod
    def append2(self, a: int, b: int) -> None: ...
    def append3(self, a: int, b: int, c: int) -> None: ...
    def finalise(self, index_path: Path, index_size: int) -> None: ...


# How big a set must be to be stored as a serialized BitMap
# The value has been empirically tested to reduce file sizes
BIGSET_LIMIT = 1000

class RoaringCollector(Collector):
    bitmaps: dict[int, BitMap]

    def __init__(self) -> None:
        self.bitmaps = {}

    def append2(self, a: int, b: int) -> None:
        self.bitmaps.setdefault(a, BitMap()).add(b)

    def append3(self, a: int, b: int, c: int) -> None:
        ab = (a << 32) + b
        self.bitmaps.setdefault(ab, BitMap()).add(c)

    def finalise(self, index_path: Path, index_size: int) -> None:
        logging.debug(f"Sorting bitmap map")
        bitmaps = sorted(self.bitmaps.items(), key = lambda b:b[0])
        max_value = bitmaps[-1][0]
        logging.debug(f"Creating index file for big sets")
        bigset_keys = (key for (key, bitmap) in bitmaps if len(bitmap) >= BIGSET_LIMIT)
        bigset_values = (bitmap.serialize() for (_, bitmap) in bitmaps if len(bitmap) >= BIGSET_LIMIT)
        IntBytesMap.build(index_path, bigset_keys, bigset_values, size=index_size, max_value=max_value)
        logging.debug(f"Creating index file for small sets")
        smallsets = (n for (_, bitmap) in bitmaps if len(bitmap) < BIGSET_LIMIT for n in bitmap)
        IntArray.build(index_path, smallsets, size=index_size)


class ListCollector(Collector):
    rows: list[int]
    rowsize: int

    def __init__(self, rowsize: int) -> None:
        self.rows = []
        self.rowsize = rowsize

    def append2(self, a: int, b: int) -> None:
        # assert self.rowsize == 2
        value = (a << 32) + b
        self.rows.append(value)

    def append3(self, a: int, b: int, c: int) -> None:
        # assert self.rowsize == 3
        value = ((a << 32) + b << 32) + c
        self.rows.append(value)

    def finalise(self, index_path: Path, index_size: int) -> None:
        assert index_size == len(self.rows)
        logging.debug(f"Sorting {index_size} rows")
        self.rows.sort()

        logging.debug(f"Creating index file")
        # Keep the least significant 4 bytes (32-bits)
        positions = (row & 0xFFFFFFFF for row in self.rows)
        IntArray.build(index_path, positions, size=index_size)


class TmpfileCollector(Collector):
    path: Path
    file: BinaryIO
    rowsize: int
    args: Namespace

    def __init__(self, rowsize: int, path: Path, args: Namespace) -> None:
        self.rowsize = rowsize
        self.path = path
        self.file = open(path, 'wb')
        self.args = args

    def append2(self, a: int, b: int) -> None:
        # assert self.rowsize == 2
        value = (a << 32) + b
        self.file.write(value.to_bytes(8, 'big'))

    def append3(self, a: int, b: int, c: int) -> None:
        # assert self.rowsize == 3
        value = ((a << 32) + b << 32) + c
        self.file.write(value.to_bytes(12, 'big'))

    def finalise(self, index_path: Path, index_size: int) -> None:
        rowbytes = self.rowsize * 4
        nr_rows = self.file.tell() // rowbytes
        self.file.close()

        assert index_size == nr_rows
        logging.debug(f"Sorting {index_size} rows")
        with open(self.path, 'r+b') as file:
            with mmap(file.fileno(), 0) as bytes_mmap:
                sort.quicksort(bytes_mmap, rowbytes, cutoff = self.args.cutoff or 1_000_000)

        logging.debug(f"Creating index file")
        with IntArray.create(index_size, index_path) as suffix_array:
            with open(self.path, 'rb') as file:
                for i in range(index_size):
                    row = file.read(rowbytes)
                    suffix_array[i] = int.from_bytes(row[-4:], 'big')

        if not self.args.keep_tmpfiles:
            self.path.unlink()


class CythonCollector(TmpfileCollector):
    def finalise(self, index_path: Path, index_size: int) -> None:
        nr_rows = self.file.tell() // (self.rowsize * 4)
        self.file.close()
        assert index_size == nr_rows

        from faster_index_builder import finalise  # type: ignore
        finalise(self.path, index_size, self.rowsize, index_path)

        if not self.args.keep_tmpfiles:
            self.path.unlink()

