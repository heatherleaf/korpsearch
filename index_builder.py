
from pathlib import Path
from argparse import Namespace
from typing import BinaryIO
from mmap import mmap
from abc import abstractmethod
import logging

from disk import DiskIntArray
from corpus import Corpus
from index import Template, TemplateLiteral, Index, UnaryIndex
import sort
from util import progress_bar, binsearch_first

# Possible sorting alternatives, the first is the default:
SORTER_CHOICES = ['tmpfile', 'internal', 'cython']

# Possible pivot selectors, used by the 'tmpfile' sorter:
PIVOT_SELECTORS = {
    'random': sort.random_pivot,
    'first': sort.take_first_pivot,
    'central': sort.take_first_pivot,
    'median3': sort.median_of_three,
    'ninther': sort.tukey_ninther,
}


def build_index(corpus: Corpus, template: Template, args: Namespace) -> None:
    index_path = Index.indexpath(corpus, template)
    index_path.parent.mkdir(exist_ok=True)
    if len(template) == 1:
        build_unary_index(corpus, index_path, template, args)
    elif len(template) == 2: 
        build_binary_index(corpus, index_path, template, args)
    else:
        raise ValueError(f"Cannot build indexes of length {len(template)}: {template}")
    with DiskIntArray(index_path) as suffix_array:
        logging.info(f"Built index for {template}, with {len(suffix_array)} elements")



def build_unary_index(corpus: Corpus, index_path: Path, template: Template, args: Namespace) -> None:
    logging.info(f"Building unary index: {template}, using sorter {args.sorter}")

    assert len(template) == 1, f"UnaryIndex templates must have length 1: {template}"
    assert not template.literals, f"Cannot build UnaryIndex from templates with literals: {template}"

    tmpl = template.template[0]
    assert tmpl.offset == 0
    features = corpus.tokens[tmpl.feature].raw()

    collector: Collector
    if args.sorter == 'internal':
        collector = ListCollector(2)
    elif args.sorter == 'cython':
        collector = CythonCollector(2, index_path.parent / 'cindex.tmp', args)
    else:
        collector = TmpfileCollector(2, index_path.parent / 'index.tmp', args)

    for pos, value in enumerate(progress_bar(features, desc="Collecting positions")):
        if value:
            collector.append2(value, pos)

    collector.finalise(index_path)



def build_binary_index(corpus: Corpus, index_path: Path, template: Template, args: Namespace) -> None:
    logging.info(f"Building binary index: {template}, using sorter {args.sorter}")

    assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
    assert all(lit.negative for lit in template.literals), f"Cannot handle positive template literals: {template}"

    tmpl1, tmpl2 = template.template
    assert tmpl1.offset == 0
    assert tmpl2.offset == template.maxdelta()
    features1 = corpus.tokens[tmpl1.feature].raw()
    features2 = corpus.tokens[tmpl2.feature].raw()
    min_freq = args.min_frequency

    unary1 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl1.feature)]))
    unary2 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl2.feature)]))

    # Optimisation: cache index lookups, because the size of the vocabulary 
    # is much smaller than the corpus size.
    cache1: dict[int, bool] = {}
    cache2: dict[int, bool] = {}
    def unary_min_frequency(unary: UnaryIndex, unary_key: int, cache: dict[int, bool]) -> bool:
        if unary_key not in cache:
            search_key = unary.search_key()
            start = binsearch_first(0, len(unary)-1, unary_key, search_key)
            end = start + min_freq - 1
            cache[unary_key] = (end < len(unary) and search_key(end) == unary_key)
        return cache[unary_key]

    collector: Collector
    if args.sorter == 'internal':
        collector = ListCollector(3)
    elif args.sorter == 'cython':
        collector = CythonCollector(3, index_path.parent / 'cindex.tmp', args) 
    else:
        collector = TmpfileCollector(3, index_path.parent / 'index.tmp', args)

    # Optimisation: use the underlying memoryview for each DiskStringArray, 
    # and don't bother with looking up InternedStrings - use the ints directly instead.
    test_literals = [(corpus.tokens[lit.feature].raw(), lit.offset, lit.value) for lit in template.literals]

    skipped_instances = 0
    for pos in progress_bar(range(len(corpus) - template.maxdelta()), desc="Collecting positions"):
        val1, val2 = features1[pos], features2[pos + tmpl2.offset]
        if val1 and val2 and all(lfeat[pos + loffset] != lval for (lfeat, loffset, lval) in test_literals):
            if min_freq and not (
                        unary_min_frequency(unary1, val1, cache1) and
                        unary_min_frequency(unary2, val2, cache2)
                    ):
                skipped_instances += 1
            else:
                collector.append3(val1, val2, pos)
    if skipped_instances:
        logging.debug(f"Skipped {skipped_instances} low-frequency instances")

    collector.finalise(index_path)


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
    def finalise(self, index_path: Path) -> None: ...


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

    def finalise(self, index_path: Path) -> None:
        nr_rows = len(self.rows)

        logging.debug(f"Sorting {nr_rows} rows")
        self.rows.sort()

        logging.debug(f"Creating index file")
        with DiskIntArray.create(nr_rows, index_path) as suffix_array:
            for i, row in enumerate(self.rows):
                # Keep the least significant 4 bytes (32-bits)
                suffix_array[i] = row & 0xFFFFFFFF


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

    def finalise(self, index_path: Path) -> None:
        rowbytes = self.rowsize * 4
        nr_rows = self.file.tell() // rowbytes
        self.file.close()

        logging.debug(f"Sorting {nr_rows} rows")
        with open(self.path, 'r+b') as file:
            with mmap(file.fileno(), 0) as bytes_mmap:
                sort.quicksort(
                    bytes_mmap,
                    rowbytes,
                    pivotselector = PIVOT_SELECTORS.get(self.args.pivot_selector, sort.random_pivot), 
                    cutoff = self.args.cutoff or 1_000_000,
                )

        logging.debug(f"Creating index file")
        with DiskIntArray.create(nr_rows, index_path) as suffix_array:
            with open(self.path, 'rb') as file:
                for i in range(nr_rows):
                    row = file.read(rowbytes)
                    suffix_array[i] = int.from_bytes(row[-4:], 'big')

        if not self.args.keep_tmpfiles:
            self.path.unlink()


class CythonCollector(TmpfileCollector):
    def finalise(self, index_path: Path) -> None:
        nr_rows = self.file.tell() // (self.rowsize * 4)
        self.file.close()

        from faster_index_builder import finalise  # type: ignore
        finalise(self.path, nr_rows, self.rowsize, index_path)

        if not self.args.keep_tmpfiles:
            self.path.unlink()

