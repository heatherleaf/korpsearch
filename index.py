
from typing import Tuple, List, Iterator, Callable, Union, Collection, Sequence, NamedTuple
from functools import total_ordering
from types import TracebackType
from pathlib import Path
import shutil
import logging
import subprocess

from disk import DiskIntArray, DiskIntArrayBuilder, InternedString, DiskFixedBytesArray
from corpus import Corpus
from indexset import IndexSet
import sort
from util import progress_bar, min_bytes_to_store_values


# Possible sorting alternatives, the first is the default:
SORTER_CHOICES = ['tmpfile', 'internal', 'java', 'lmdb']

################################################################################
## Literals, templates and instances

class Literal(NamedTuple):
    negative : bool
    offset : int
    feature : str
    first_value : InternedString
    last_value : InternedString

    def __str__(self):
        if self.is_prefix():
            return f"{self.feature}:{self.offset}{'#' if self.negative else '='}{self.first_value}-{self.last_value}"
        else:
            return f"{self.feature}:{self.offset}{'#' if self.negative else '='}{self.first_value}"

    def is_prefix(self):
        return self.first_value != self.last_value

    # unoptimized version for use with Query.check_position
    def check_position(self, corpus_tokens, pos):
        value = corpus_tokens[self.feature][pos + self.offset]
        return (value >= self.first_value and value <= self.last_value) != self.negative

    @staticmethod
    def parse(corpus:Corpus, litstr:str) -> 'Literal':
        try:
            feature, rest = litstr.split(':')
            assert feature.replace('_','').isalnum()
            try:
                offset, value = rest.split('=')
                return Literal(False, int(offset), feature.lower(), corpus.intern(feature, value.encode()), corpus.intern(feature, value.encode()))
            except ValueError:
                offset, value = rest.split('#')
                return Literal(True, int(offset), feature.lower(), corpus.intern(feature, value.encode()), corpus.intern(feature, value.encode()))
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed literal: {litstr}")


class DisjunctiveGroup(NamedTuple):
    negative : bool
    offset : int
    features : Tuple[str]
    literals : Tuple[Literal]

    def __str__(self):
        string = ""
        for elem in self.literals: string += str(elem) + "|"
        return string[:-1]

    # unoptimized version for use with Query.check_position
    def check_position(self, corpus_tokens, pos):
        return any(lit.check_position(corpus_tokens, pos) for lit in self.literals)

    def is_prefix(self):
        return any(lit.is_prefix() for lit in self.literals)

    @staticmethod
    def create(literals:Tuple[Literal]):
        negative = any(lit.negative for lit in literals)
        offset = min(lit.offset for lit in literals)
        features = tuple(lit.feature for lit in literals)
        return DisjunctiveGroup(negative, offset, features, literals)


class TemplateLiteral(NamedTuple):
    offset : int
    feature : str

    def __str__(self):
        return f"{self.feature}:{self.offset}"

    @staticmethod
    def parse(litstr:str) -> 'TemplateLiteral':
        try:
            feature, offset = litstr.split(':')
            assert feature.replace('_','').isalnum()
            return TemplateLiteral(int(offset), feature.lower())
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed template literal: {litstr}")


@total_ordering
class Template:
    template : Tuple[TemplateLiteral,...]
    literals : Tuple[Literal,...]

    def __init__(self, template:Sequence[TemplateLiteral], literals:Collection[Literal]=[]):
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

    def maxdelta(self):
        return max(t.offset for t in self.template)

    def __str__(self) -> str:
        return '+'.join(map(str, self.template + self.literals))

    def querystr(self) -> str:
        min_offset = min(l.offset for lits in [self.template, self.literals] for l in lits)
        max_offset = max(l.offset for lits in [self.template, self.literals] for l in lits)
        tokens: List[str] = []
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

    def __eq__(self, other:'Template') -> bool:
        return isinstance(other, Template) and \
            (self.template, self.literals) == (other.template, other.literals)

    def __lt__(self, other:'Template') -> bool:
        return (len(self), self.template, self.literals) < (len(other), other.template, other.literals)

    def __hash__(self) -> int:
        return hash((self.template, self.literals))

    @staticmethod
    def parse(corpus:Corpus, template_str:str) -> 'Template':
        try:
            template = []
            literals = []
            for litstr in template_str.split('+'):
                try:
                    literals.append(Literal.parse(corpus, litstr))
                except ValueError:
                    template.append(TemplateLiteral.parse(litstr))
            return Template(template, literals)
        except (ValueError, AssertionError):
            raise ValueError(
                "Ill-formed template - it should be on the form pos:0 or word:0+pos:2 "
                "or pos:0+lemma:1+sentence:1#S: " + template_str
            )


@total_ordering
class Instance:
    values : Tuple[InternedString,...]

    def __init__(self, values : Sequence[Sequence[InternedString]]):
        assert len(values) > 0
        self.values = tuple(values)

    def __str__(self) -> str:
        return '+'.join(map(str, self.values))

    def __iter__(self) -> Iterator[InternedString]:
        yield from self.values

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other:'Instance') -> bool:
        return isinstance(other, Instance) and self.values == other.values

    def __lt__(self, other:'Instance') -> bool:
        return self.values < other.values

    def __hash__(self) -> int:
        return hash(self.values)


################################################################################
## Inverted sentence index
## Implemented as a sorted array of interned strings
## This is a kind of modified suffix array - a "pruned" SA if you like

class Index:
    dir_suffix : str = '.indexes'

    template : Template
    index : DiskIntArray
    search_key : Callable[[int], Union[InternedString, Tuple[InternedString,...]]]
    # Typing note: as optimisation we use the value (s) instead of a 1-tuple (s,) so the
    # return type is a union of a value and a tuple. But then Pylance can't infer the correct
    # type, so we have to write "# type: ignore" on some lines below.

    def __init__(self, corpus:Corpus, template:Template):
        self.corpus = corpus
        self.template = template
        indexpath = self.indexpath(corpus, template)
        self.index = index = DiskIntArray(indexpath)

        if len(self.template) == 1:
            [tmpl] = list(template)
            self.search_key = lambda k: \
                corpus.tokens[tmpl.feature][index[k] + tmpl.offset]
        elif len(self.template) == 2:
            [tmpl1, tmpl2] = list(template)
            self.search_key = lambda k: (
                corpus.tokens[tmpl1.feature][index[k] + tmpl1.offset],
                corpus.tokens[tmpl2.feature][index[k] + tmpl2.offset],
            )
        else:
            # The above two are just optimisations of the following generic search key:
            self.search_key = lambda k: tuple(
                corpus.tokens[tmpl.feature][index[k] + tmpl.offset] 
                for tmpl in template
            )

    def __str__(self) -> str:
        return self.__class__.__name__ + ':' + str(self.template) 

    def __len__(self) -> int:
        return len(self.index)

    def search(self, instance:Instance, offset:int=0) -> IndexSet:
        set_start, set_end = self.lookup_instance(instance)
        set_size = set_end - set_start + 1
        iset = IndexSet(self.index, set_start, set_size, offset=offset)
        return iset

    def lookup_instance(self, instance:Instance) -> Tuple[int, int]:
        search_key = self.search_key
        instance_key = instance.values
        if len(instance_key) == 1: instance_key = instance_key[0]
        else: instance_key = tuple(zip(*instance_key))

        start, end = 0, len(self)-1
        instance_key_first = instance_key[0]
        while start <= end:
            mid = (start + end) // 2
            key = search_key(mid)
            if key < instance_key_first:  # type: ignore
                start = mid + 1
            else:
                end = mid - 1
        first_index = start
        if search_key(first_index) != instance_key_first:
            raise KeyError(f'Instance "{instance}" not found')

        end = len(self) - 1
        instance_key_last = instance_key[1]
        while start <= end:
            mid = (start + end) // 2
            key = search_key(mid)
            if key <= instance_key_last:  # type: ignore
                start = mid + 1
            else:
                end = mid - 1
        last_index = end
        if search_key(last_index) != instance_key_last:
            raise KeyError(f'Instance "{instance}" not found')

        return first_index, last_index

    def __enter__(self) -> 'Index':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        self.index.close()

    @staticmethod
    def indexpath(corpus:Corpus, template:Template):
        basepath = corpus.path.with_suffix(Index.dir_suffix)
        return basepath / str(template) / str(template)

    @staticmethod
    def build(corpus:Corpus, template:Template, min_frequency:int=0, keep_tmpfiles:bool=False, sorter:str=SORTER_CHOICES[0]):
        index_path : Path = Index.indexpath(corpus, template)
        index_path.parent.mkdir(exist_ok=True)
        if len(template) == 1 and not template.literals:
            build_simple_unary_index(corpus, index_path, template, keep_tmpfiles, sorter)
        else:
            build_general_index(corpus, index_path, template, min_frequency, keep_tmpfiles, sorter)

        with DiskIntArray(index_path) as suffix_array:
            logging.info(f"Built index for {template}, with {len(suffix_array)} elements")


###################################################################################################
## Different ways of building different indexes

def build_simple_unary_index(corpus:Corpus, index_path:Path, template:Template, keep_tmpfiles:bool, sorter:str):
    logging.debug(f"Building simple unary index for {template} @ {index_path}, using sorter '{sorter}'")
    assert len(template) == 1 and not template.literals
    index_size = len(corpus)
    bytesize = min_bytes_to_store_values(index_size)
    rowsize = bytesize * (1 + len(template))
    tmpl : TemplateLiteral = template.template[0]

    def collect_positions(collect):
        for pos in progress_bar(range(index_size), desc="Collecting positions"):
            instance_value = corpus.tokens[tmpl.feature][pos+tmpl.offset]
            if instance_value:
                collect(
                    instance_value.index.to_bytes(bytesize, 'big') +
                    pos.to_bytes(bytesize, 'big')
                )

    collect_and_sort_positions(collect_positions, index_path, index_size, bytesize, rowsize, keep_tmpfiles, sorter)


def build_general_index(corpus:Corpus, index_path:Path, template:Template, min_frequency:int, keep_tmpfiles:bool, sorter:str):
    logging.debug(f"Building index for {template} @ {index_path}, using sorter '{sorter}'")
    index_size = len(corpus) - template.maxdelta()
    bytesize = min_bytes_to_store_values(index_size)
    rowsize = bytesize * (1 + len(template))

    unary_indexes : List[Index] = []
    if min_frequency > 0 and len(template) > 1:
        unary_indexes = [
            Index(corpus, Template([TemplateLiteral(0, tmpl.feature)])) 
            for tmpl in template
        ]

    def unary_min_frequency(unary, unary_key, min_frequency) -> bool:
        # This is an optimised version of:
        # >>> start, end = unary.lookup_instance(x); return end-start >= min_frequency
        searchkey = unary.search_key
        start, end = 0, len(unary)-1
        while start <= end:
            mid = (start + end) // 2
            key = searchkey(mid)
            if key < unary_key: 
                start = mid + 1
            else:
                end = mid - 1
        end = start + min_frequency - 1
        return end < len(unary.index) and searchkey(end) == unary_key

    assert all(lit.negative for lit in template.literals), \
        f"Cannot handle positive template literals: {template}"
    if len(template.literals) == 0:
        test_literals = lambda pos: True
    elif len(template.literals) == 1:
        [(_neg1, off1, feat1, val1, _)] = template.literals
        test_literals = lambda pos: corpus.tokens[feat1][pos+off1] != val1
    elif len(template.literals) == 2:
        [(_neg1, off1, feat1, val1, _), (_neg2, off2, feat2, val2, _)] = template.literals
        test_literals = lambda pos: \
            corpus.tokens[feat1][pos+off1] != val1 and corpus.tokens[feat2][pos+off2] != val2
    else:
        # The above three are just optimisations of the following generic test function:
        test_literals = lambda pos: all(
            corpus.tokens[lit.feature][pos+lit.offset] != lit.value
            for lit in template.literals
        )

    def collect_positions(collect):
        skipped_instances = 0
        for pos in progress_bar(range(index_size), desc="Collecting positions"):
            instance_values = [corpus.tokens[tmpl.feature][pos+tmpl.offset] for tmpl in template]
            if all(instance_values) and test_literals(pos):
                if unary_indexes and not all(
                            unary_min_frequency(unary, val, min_frequency)
                            for val, unary in zip(instance_values, unary_indexes)
                        ):
                    skipped_instances += 1
                else:
                    collect(
                        b''.join(val.index.to_bytes(bytesize, 'big') for val in instance_values) +
                        pos.to_bytes(bytesize, 'big')
                    )
        if skipped_instances:
            logging.info(f"Skipped {skipped_instances} low-frequency instances")

    collect_and_sort_positions(collect_positions, index_path, index_size, bytesize, rowsize, keep_tmpfiles, sorter)


def collect_and_sort_positions(collect_positions, index_path, index_size, bytesize, rowsize, keep_tmpfiles, sorter):
    if sorter == 'internal':
        collect_and_sort_internally(collect_positions, index_path, index_size, bytesize)
    elif sorter == 'lmdb':
        collect_and_sort_lmdb(collect_positions, index_path, index_size, bytesize, keep_tmpfiles)
    else:
        collect_and_sort_tmpfile(collect_positions, index_path, index_size, bytesize, rowsize, keep_tmpfiles, sorter)


def collect_and_sort_internally(collect_positions, index_path, index_size, bytesize):
    tmplist = []
    collect_positions(tmplist.append)
    logging.debug(f"Sorting {len(tmplist)} rows.")
    tmplist.sort()
    logging.debug(f"Creating suffix array")
    with DiskIntArrayBuilder(index_path, max_value=index_size) as suffix_array:
        for row in tmplist:
            pos = int.from_bytes(row[-bytesize:], 'big')
            suffix_array.append(pos)


def collect_and_sort_tmpfile(collect_positions, index_path, index_size, bytesize, rowsize, keep_tmpfiles, sorter):
    tmpfile = index_path.parent / 'index.tmp'
    with open(tmpfile, 'wb') as OUT:
        collect_positions(OUT.write)
        nr_rows = OUT.tell() // rowsize

    logging.debug(f"Sorting {nr_rows} rows.")
    if sorter == 'java':
        subprocess.run(['java', '-jar', 'DiskFixedSizeArray.jar', tmpfile, str(rowsize), 'random', '100000'])
    else:
        with DiskFixedBytesArray(tmpfile, rowsize) as bytes_array:
            sort.quicksort(
                bytes_array,
                pivotselector = sort.random_pivot, # take_first_pivot, median_of_three, tukey_ninther
                cutoff = 5_000_000,
            )

    logging.debug(f"Creating suffix array")
    with DiskIntArrayBuilder(index_path, max_value=index_size) as suffix_array:
        with open(tmpfile, 'rb') as IN:
            while (row := IN.read(rowsize)):
                pos = int.from_bytes(row[-bytesize:], 'big')
                suffix_array.append(pos)

    if not keep_tmpfiles:
        tmpfile.unlink()


def collect_and_sort_lmdb(collect_positions, index_path, index_size, bytesize, keep_tmpfiles):
    import lmdb
    tmpdir = index_path.parent / 'index.tmpdb'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    env = lmdb.open(str(tmpdir), map_size=1_000_000_000_000)
    with env.begin(write=True) as DB:
        collect_positions(lambda row: DB.put(row, b''))
    logging.debug(f"Creating suffix array")
    with env.begin() as DB:
        with DiskIntArrayBuilder(index_path, max_value=index_size) as suffix_array:
            for row, _ in DB.cursor():
                pos = int.from_bytes(row[-bytesize:], 'big')
                suffix_array.append(pos)
    env.close()
    if not keep_tmpfiles:
        shutil.rmtree(tmpdir, ignore_errors=True)

