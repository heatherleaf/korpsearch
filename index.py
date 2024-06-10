
from typing import NamedTuple, Any
from collections.abc import Iterator, Callable, Collection, Sequence
from functools import total_ordering
from pathlib import Path
from argparse import Namespace
import shutil
import logging
import subprocess

from disk import InternedString, DiskFixedBytesArray, DiskIntArray
from corpus import Corpus
from indexset import IndexSet
import sort
from util import progress_bar, ByteOrder, binsearch_first, binsearch_range


# Possible sorting alternatives, the first is the default:
SORTER_CHOICES = ['tmpfile', 'internal', 'java', 'lmdb', 'multikey']

# Possible pivot selectors, used by the 'tmpfile' and 'java' sorters:
PIVOT_SELECTORS = {
    'random': sort.random_pivot,
    'first': sort.take_first_pivot,
    'central': sort.take_first_pivot,
    'median3': sort.median_of_three,
    'ninther': sort.tukey_ninther,
}


################################################################################
## Literals, templates and instances

class Literal(NamedTuple):
    negative: bool
    offset: int
    feature: str
    value: InternedString

    def __str__(self) -> str:
        return f"{self.feature}:{self.offset}{'#' if self.negative else '='}{self.value}"

    @staticmethod
    def parse(corpus: Corpus, litstr: str) -> 'Literal':
        try:
            feature, rest = litstr.split(':')
            assert feature.replace('_','').isalnum()
            try:
                offset, value = rest.split('=')
                return Literal(False, int(offset), feature.lower(), corpus.intern(feature, value.encode()))
            except ValueError:
                offset, value = rest.split('#')
                return Literal(True, int(offset), feature.lower(), corpus.intern(feature, value.encode()))
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed literal: {litstr}")


class TemplateLiteral(NamedTuple):
    offset: int
    feature: str

    def __str__(self) -> str:
        return f"{self.feature}:{self.offset}"

    @staticmethod
    def parse(litstr: str) -> 'TemplateLiteral':
        try:
            feature, offset = litstr.split(':')
            assert feature.replace('_','').isalnum()
            return TemplateLiteral(int(offset), feature.lower())
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed template literal: {litstr}")


@total_ordering
class Template:
    template: tuple[TemplateLiteral,...]
    literals: tuple[Literal,...]

    def __init__(self, template: Sequence[TemplateLiteral], literals: Collection[Literal] = []) -> None:
        self.template = tuple(template)
        self.literals = tuple(sorted(set(literals)))
        try:
            assert self.template == tuple(sorted(set(self.template))), f"Unsorted template"
            assert self.literals == tuple(sorted(literals)),           f"Duplicate literal(s)"
            assert len(self.template) > 0,                             f"Empty template"
            assert min(t.offset for t in self.template) == 0,          f"Minimum offset must be 0"
            if self.literals:
                assert all(lit.negative for lit in self.literals),     f"Positive template literal(s)"
                assert all(0 <= lit.offset <= self.maxdelta() 
                           for lit in self.literals),                  f"Literal offset must be within 0...{self.maxdelta()}"
        except AssertionError:
            raise ValueError(f"Invalid template: {self}")

    def maxdelta(self) -> int:
        return max(t.offset for t in self.template)

    def __str__(self) -> str:
        return '+'.join(map(str, self.template + self.literals))

    def querystr(self) -> str:
        offsets = [lit.offset for lit in self.template] + [lit.offset for lit in self.literals]
        min_offset = min(offsets)
        max_offset = max(offsets)
        tokens: list[str] = []
        for offset in range(min_offset, max_offset+1):
            tok = ','.join('?' + l.feature for l in self.template if l.offset == offset)
            lit = ','.join(f'{l.feature}{"≠" if l.negative else "="}"{l.value}"' 
                           for l in self.literals if l.offset == offset)
            if lit:
                tokens.append(tok + '|' + lit)
            else:
                tokens.append(tok)
        return ''.join('[' + tok + ']' for tok in tokens)

    def __iter__(self) -> Iterator[TemplateLiteral]:
        return iter(self.template)

    def __len__(self) -> int:
        return len(self.template)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Template) and \
            (self.template, self.literals) == (other.template, other.literals)

    def __lt__(self, other: 'Template') -> bool:
        return (len(self), self.template, self.literals) < (len(other), other.template, other.literals)

    def __hash__(self) -> int:
        return hash((self.template, self.literals))

    @staticmethod
    def parse(corpus: Corpus, template_str: str) -> 'Template':
        try:
            literals: list[Literal] = []
            template: list[TemplateLiteral] = []
            for litstr in template_str.split('+'):
                try:
                    literals.append(Literal.parse(corpus, litstr))
                except ValueError:
                    template.append(TemplateLiteral.parse(litstr))
            return Template(template, literals)
        except (ValueError, AssertionError):
            raise ValueError(
                "Ill-formed template - it should be on the form pos:0 or word:0+pos:2: " + template_str
            )


@total_ordering
class Instance:
    values: tuple[InternedString,...]

    def __init__(self, values: Sequence[InternedString]) -> None:
        assert len(values) > 0
        self.values = tuple(values)

    def __str__(self) -> str:
        return '+'.join(map(str, self.values))

    def __iter__(self) -> Iterator[InternedString]:
        yield from self.values

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Instance) and self.values == other.values

    def __lt__(self, other: 'Instance') -> bool:
        return self.values < other.values

    def __hash__(self) -> int:
        return hash(self.values)


################################################################################
## Inverted sentence index
## Implemented as a sorted array of interned strings
## This is a kind of modified suffix array - a "pruned" SA if you like

class Index:
    dir_suffix: str = '.indexes'
    corpus: Corpus
    template: Template
    index: DiskIntArray
    path: Path

    def __init__(self, corpus: Corpus, template: Template) -> None:
        self.corpus = corpus
        self.template = template
        self.path = self.indexpath(corpus, template)
        self.index = DiskIntArray(self.path)

    def __str__(self) -> str:
        return self.__class__.__name__ + ':' + str(self.template) 

    def __len__(self) -> int:
        return len(self.index)

    def __enter__(self) -> 'Index':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self.index.close()

    def search(self, instance: Instance, offset: int = 0) -> IndexSet:
        set_start, set_end = self.lookup_instance(instance)
        set_size = set_end - set_start + 1
        iset = IndexSet(self.index, path=self.path, start=set_start, size=set_size, offset=offset)
        return iset

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        raise NotImplementedError("Must be overridden by a subclass")

    @staticmethod
    def indexpath(corpus: Corpus, template: Template) -> Path:
        basepath = corpus.path.with_suffix(Index.dir_suffix)
        return basepath / str(template) / str(template)

    @staticmethod
    def get(corpus: Corpus, template: Template) -> 'Index':
        if len(template) == 1:
            return UnaryIndex(corpus, template)
        elif len(template) == 2: 
            return BinaryIndex(corpus, template)
        else:
            raise ValueError(f"Cannot handle indexes of length {len(template)}: {template}")

    @staticmethod
    def build(corpus: Corpus, template: Template, args: Namespace) -> None:
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


class UnaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 1, f"UnaryIndex templates must have length 1: {template}"
        super().__init__(corpus, template)

    def search_key(self) -> Callable[[int], InternedString]:
        tmpl = self.template.template[0]
        features = self.corpus.tokens[tmpl.feature]
        offset = tmpl.offset
        index = self.index.array
        return lambda k: features[index[k] + offset]

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 1, f"UnaryIndex instance must have length 1: {instance}"
        return binsearch_range(0, len(self)-1, instance.values[0], self.search_key())


class BinaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
        super().__init__(corpus, template)

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 2, f"BinaryIndex instance must have length 2: {instance}"
        tmpl1, tmpl2 = self.template.template
        features1, features2 = self.corpus.tokens[tmpl1.feature], self.corpus.tokens[tmpl2.feature]
        offset1, offset2 = tmpl1.offset, tmpl2.offset
        index = self.index.array
        def search_key(k: int) -> tuple[InternedString, InternedString]:
            return (features1[index[k] + offset1], features2[index[k] + offset2])
        return binsearch_range(0, len(self)-1, instance.values, search_key)


###################################################################################################
## Different ways of building different indexes


# We need big-endian byte order (i.e., most-significant byte first).
# Because then we can treat a tuple of integers as a bytestring (when comparing them).
SORTING_BYTEORDER: ByteOrder = 'big'

# Index sets should only consist of 4-byte (unsigned) integers.
BYTESIZE = 4


def build_unary_index(corpus: Corpus, index_path: Path, template: Template, args: Namespace) -> None:
    assert len(template) == 1, f"UnaryIndex templates must have length 1: {template}"
    assert not template.literals, f"Cannot build UnaryIndex from templates with literals: {template}"

    logging.debug(f"Building simple unary index for {template} @ {index_path}, using sorter '{args.sorter}'")
    index_size = len(corpus)
    bitsize = BYTESIZE * 8
    rowsize = BYTESIZE * 2
    tmpl: TemplateLiteral = template.template[0]
    features = corpus.tokens[tmpl.feature]

    def collect_positions(collect: RowAdder) -> None:
        for pos in progress_bar(range(index_size), desc="Collecting positions"):
            instance_value = features[pos + tmpl.offset]
            if instance_value:
                row = (instance_value.index << bitsize) + pos
                collect(row.to_bytes(rowsize, SORTING_BYTEORDER))

    collect_and_sort_positions(collect_positions, index_path, index_size, BYTESIZE, rowsize, args)


def build_binary_index(corpus: Corpus, index_path: Path, template: Template, args: Namespace) -> None:
    assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
    assert all(lit.negative for lit in template.literals), f"Cannot handle positive template literals: {template}"

    logging.debug(f"Building binary index for {template} @ {index_path}, using sorter '{args.sorter}'")

    index_size = len(corpus) - template.maxdelta()
    bytesize = 4
    bitsize = bytesize * 8
    rowsize = bytesize * 3

    def test_literals(pos: int) -> bool:
        return all(
            corpus.tokens[lit.feature][pos+lit.offset] != lit.value
            for lit in template.literals
        )

    tmpl1, tmpl2 = template.template
    features1, features2 = corpus.tokens[tmpl1.feature], corpus.tokens[tmpl2.feature]

    unary1 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl1.feature)]))
    unary2 = UnaryIndex(corpus, Template([TemplateLiteral(0, tmpl2.feature)]))

    def unary_min_frequency(unary: UnaryIndex, unary_key: InternedString) -> bool:
        search_key = unary.search_key()
        start = binsearch_first(0, len(unary)-1, unary_key, search_key)
        end = start + args.min_frequency - 1
        return end < len(unary2) and search_key(end) == unary_key

    def collect_positions(collect: RowAdder) -> None:
        skipped_instances = 0
        for pos in progress_bar(range(index_size), desc="Collecting positions"):
            val1, val2 = features1[pos + tmpl1.offset], features2[pos + tmpl2.offset]
            if val1 and val2 and test_literals(pos):
                if args.min_frequency and not (
                            unary_min_frequency(unary1, val1) and
                            unary_min_frequency(unary2, val2)
                        ):
                    skipped_instances += 1
                else:
                    row = (((val1.index << bitsize) + val2.index) << bitsize) + pos
                    collect(row.to_bytes(rowsize, SORTING_BYTEORDER))
        if skipped_instances:
            logging.info(f"Skipped {skipped_instances} low-frequency instances")

    collect_and_sort_positions(collect_positions, index_path, index_size, bytesize, rowsize, args)


RowAdder = Callable[[bytes], Any]
RowCollector = Callable[[RowAdder], None]

def collect_and_sort_positions(collect_positions: RowCollector, index_path: Path, 
                               index_size: int, bytesize: int, rowsize: int, args: Namespace) -> None:
    if args.sorter == 'internal':
        collect_and_sort_internally(collect_positions, index_path, index_size, bytesize, args)
    elif args.sorter == 'lmdb':
        collect_and_sort_lmdb(collect_positions, index_path, index_size, bytesize, args)
    else:
        collect_and_sort_tmpfile(collect_positions, index_path, index_size, bytesize, rowsize, args)


def collect_and_sort_internally(collect_positions: RowCollector, index_path: Path, 
                                index_size: int, bytesize: int, args: Namespace) -> None:
    tmplist: list[bytes] = []
    collect_positions(tmplist.append)
    logging.debug(f"Sorting {len(tmplist)} rows.")
    tmplist.sort()
    logging.debug(f"Creating suffix array")
    with DiskIntArray.create(len(tmplist), index_path, max_value=index_size) as suffix_array:
        for i, row in enumerate(tmplist):
            pos = int.from_bytes(row[-bytesize:], SORTING_BYTEORDER)
            suffix_array[i] = pos


def collect_and_sort_tmpfile(collect_positions: RowCollector, index_path: Path, 
                             index_size: int, bytesize: int, rowsize: int, args: Namespace) -> None:
    tmpfile = index_path.parent / 'index.tmp'
    with open(tmpfile, 'wb') as file:
        collect_positions(file.write)
        nr_rows = file.tell() // rowsize

    logging.debug(f"Sorting {nr_rows} rows.")

    if args.sorter == 'java':
        pivotselector = args.pivot_selector or 'random'
        cmd = ['java', '-jar', 'DiskFixedSizeArray.jar', 
               str(tmpfile), str(rowsize), pivotselector, str(args.cutoff or 1_000_000)]
        subprocess.run(cmd)

    elif args.sorter == 'multikey':
        from mmap import mmap
        with open(tmpfile, 'r+b') as file:
            mview = memoryview(mmap(file.fileno(), 0))
        assert len(mview) % rowsize == 0, \
            f"File size ({len(mview)}) is not divisible by rowsize ({rowsize})"
        sort.multikeysort(mview, rowsize)

    else:
        with DiskFixedBytesArray(tmpfile, rowsize) as bytes_array:
            sort.quicksort(
                bytes_array,
                pivotselector = PIVOT_SELECTORS.get(args.pivot_selector, sort.random_pivot), 
                cutoff = args.cutoff or 1_000_000,
            )

    logging.debug(f"Creating suffix array")
    with DiskIntArray.create(nr_rows, index_path, max_value=index_size) as suffix_array:
        with open(tmpfile, 'rb') as IN:
            i = 0
            while (row := IN.read(rowsize)):
                pos = int.from_bytes(row[-bytesize:], SORTING_BYTEORDER)
                suffix_array[i] = pos
                i += 1

    if not args.keep_tmpfiles:
        tmpfile.unlink()


def collect_and_sort_lmdb(collect_positions: RowCollector, index_path: Path, 
                          index_size: int, bytesize: int, args: Namespace) -> None:
    import lmdb  # type: ignore
    tmpdir = index_path.parent / 'index.tmpdb'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    env: Any = lmdb.open(str(tmpdir), map_size=1_000_000_000_000)  # type: ignore
    with env.begin(write=True) as db:
        collect_positions(lambda row: db.put(row, b''))
    logging.debug(f"Creating suffix array")
    with env.begin() as db:
        nrows = db.stat()['entries']
        with DiskIntArray.create(nrows, index_path, max_value=index_size) as suffix_array:
            for i, (row, _) in enumerate(db.cursor()):
                pos = int.from_bytes(row[-bytesize:], SORTING_BYTEORDER)
                suffix_array[i] = pos
    env.close()
    if not args.keep_tmpfiles:
        shutil.rmtree(tmpdir, ignore_errors=True)

