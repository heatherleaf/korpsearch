
from typing import Tuple, List, Iterator, Callable, Union, Collection, Sequence, NamedTuple
from functools import total_ordering
from types import TracebackType
import logging

from disk import DiskIntArray, DiskIntArrayBuilder, InternedString, DiskFixedBytesArray
from corpus import Corpus
from indexset import IndexSet
import sort
from util import progress_bar


################################################################################
## Literals, templates and instances

class Literal(NamedTuple):
    negative : bool
    offset : int
    feature : str
    value : InternedString

    def __str__(self):
        return f"{self.feature}:{self.offset}{'#' if self.negative else '='}{self.value}"

    @staticmethod
    def parse(corpus:Corpus, litstr:str) -> 'Literal':
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
        assert self.template == tuple(sorted(set(self.template))), f"Unsorted template: {self}"
        assert self.literals == tuple(sorted(literals)),           f"Duplicate literal(s): {self}"
        assert len(self.template) > 0,                             f"Empty template: {self}"
        assert min(t.offset for t in self.template) == 0,          f"Minimum offset must be 0: {self}"
        assert all(lit.negative for lit in self.literals),         f"Positive template literal(s): {self}"

    def maxdelta(self):
        return max(t.offset for t in self.template)

    def __str__(self) -> str:
        return '+'.join(map(str, self.template + self.literals))

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

    def __init__(self, values : Sequence[InternedString]):
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

        start, end = 0, len(self)-1
        while start <= end:
            mid = (start + end) // 2
            key = search_key(mid)
            if key < instance_key:  # type: ignore
                start = mid + 1
            else:
                end = mid - 1
        first_index = start
        if search_key(first_index) != instance_key:
            raise KeyError(f'Instance "{instance}" not found')

        end = len(self) - 1
        while start <= end:
            mid = (start + end) // 2
            key = search_key(mid)
            if key <= instance_key:  # type: ignore
                start = mid + 1
            else:
                end = mid - 1
        last_index = end
        assert search_key(last_index) == instance_key

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
    def build(corpus:Corpus, template:Template, min_frequency:int=0, keep_tmpfiles:bool=False):
        logging.debug(f"Building index for {template}")
        index_path = Index.indexpath(corpus, template)
        index_path.parent.mkdir(exist_ok=True)

        maxdelta = template.maxdelta()
        index_size = len(corpus) - maxdelta

        unary_indexes : List[Index] = []
        if min_frequency > 0 and len(template) > 1:
            unary_indexes = [
                Index(corpus, Template([TemplateLiteral(0, tmpl.feature)])) 
                for tmpl in template
            ]

        def unary_min_frequency(unary, unary_key, min_frequency) -> bool:
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

        tmpfile = index_path.parent / 'index.tmp'
        bytesize = DiskIntArrayBuilder._min_bytes_to_store_values(index_size)
        rowsize = bytesize * (1 + len(template))

        with open(tmpfile, 'wb') as OUT:
            skipped_instances : int = 0
            for pos in progress_bar(range(index_size), desc="Collecting positions"):
                instance_values = [corpus.tokens[tmpl.feature][pos+tmpl.offset] for tmpl in template]
                if all(instance_values) and all(
                            # We can only handle negative literals:
                            corpus.tokens[lit.feature][pos+lit.offset] != lit.value
                            for lit in template.literals
                        ):
                    if unary_indexes and not all(
                                unary_min_frequency(unary, val, min_frequency)
                                for val, unary in zip(instance_values, unary_indexes)
                            ):
                        skipped_instances += 1
                    else:
                        for val in instance_values:
                            OUT.write(val.index.to_bytes(bytesize, 'big'))
                        OUT.write(pos.to_bytes(bytesize, 'big'))
            if skipped_instances:
                logging.info(f"Skipped {skipped_instances} low-frequency instances")
            nr_rows = OUT.tell() // rowsize

        # bsort is really fast, but it hangs on some files... https://github.com/yoyyyyo/bsort
        # subprocess.run(['bsort/bsort', '-k', str(rowsize), '-r', str(rowsize), tmpfile])
        logging.debug(f"Sorting {nr_rows} rows")
        with DiskFixedBytesArray(tmpfile, rowsize) as bytes_array:
            sort.quicksort(
                bytes_array,
                # pivotselector = sort.random_pivot, 
                pivotselector = sort.median_of_three,
                # pivotselector = sort.tukey_ninther,
                cutoff = 100_000,
            )

        logging.debug(f"Creating suffix array")
        with DiskIntArrayBuilder(index_path, max_value=index_size) as suffix_array:
            with open(tmpfile, 'rb') as IN:
                while (row := IN.read(rowsize)):
                    pos = int.from_bytes(row[-bytesize:], 'big')
                    suffix_array.append(pos)

        if not keep_tmpfiles:
            tmpfile.unlink()

        logging.info(f"Built index for {template}, with {nr_rows} rows")

