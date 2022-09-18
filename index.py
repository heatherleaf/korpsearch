
from pathlib import Path
from typing import Tuple, List, Iterator, Callable, Union, Sequence, NamedTuple
from functools import total_ordering
from types import TracebackType
import logging

from disk import DiskIntArray, DiskIntArrayBuilder, InternedString
from corpus import Corpus
from indexset import IndexSet
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

    def __init__(self, template:Sequence[TemplateLiteral], literals:Sequence[Literal]=[]):
        self.template = tuple(sorted(template))
        self.literals = tuple(sorted(literals))
        assert all(sorted(x) == sorted(set(x)) for x in (template, literals)), f"Duplicate literal(s): {self}"
        assert len(template) > 0,                                              f"Empty template: {self}"
        assert min(t.offset for t in template) == 0,                           f"Minimum offset must be 0: {self}"
        assert all(lit.negative for lit in literals),                          f"Positive template literal(s): {self}"

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
    def build(corpus:Corpus, template:Template, min_frequency:int=0, use_sqlite:bool=False, keep_tmpfiles:bool=False):
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

        def all_unary_min_frequency(instance_values) -> bool:
            return all(
                unary_min_frequency(unary, val, min_frequency)
                for val, unary in zip(instance_values, unary_indexes)
            )

        assert all(lit.negative for lit in template.literals), \
            f"Cannot handle positive template literals: {template}"

        def generate_positions():
            with progress_bar(range(index_size), desc="Collecting positions") as pbar:
                for pos in pbar:
                    instance_values = [corpus.tokens[tmpl.feature][pos+tmpl.offset] for tmpl in template]
                    if all(instance_values) and all(
                                corpus.tokens[lit.feature][pos+lit.offset] != lit.value
                                for lit in template.literals
                            ):
                        yield pos, instance_values

        if use_sqlite:
            dbfile = index_path.parent / 'index.db.tmp'
            with IndexBuilderDB(dbfile, len(template)+1, keep_dbfile=keep_tmpfiles) as con:
                skipped_instances : int = 0
                def generate_db_rows() -> Iterator[Tuple[int,...]]:
                    nonlocal skipped_instances
                    for pos, instance_values in generate_positions():
                        if unary_indexes and not all_unary_min_frequency(instance_values):
                            skipped_instances += 1
                        else:
                            yield tuple(value.index for value in instance_values) + (pos,)

                con.insert_rows(generate_db_rows())
                if skipped_instances:
                    logging.info(f"Skipped {skipped_instances} low-frequency instances")

                nr_rows, nr_instances = con.count_rows_and_instances()
                logging.debug(f" --> created instance database, {nr_rows} rows, {nr_instances} instances")

                with DiskIntArrayBuilder(index_path, max_value=index_size) as suffix_array:
                    for row in progress_bar(con.row_iterator(), "Creating index", total=nr_rows):
                        pos = row[-1]
                        suffix_array.append(pos)

                logging.info(f"Built index for {template}, with {nr_rows} rows, {nr_instances} instances")

        else:
            with DiskIntArrayBuilder(index_path, max_value=index_size) as suffix_array:
                if unary_indexes:
                    skipped_instances : int = 0
                    for pos, instance_values in generate_positions():
                        if unary_indexes and not all_unary_min_frequency(instance_values):
                            skipped_instances += 1
                        else:
                            suffix_array.append(pos)
                    if skipped_instances:
                        logging.debug(f"Skipped {skipped_instances} low-frequency instances")
                else:
                    for pos, _ in generate_positions():
                        suffix_array.append(pos)
                nr_rows = len(suffix_array)

            # sort the suffix array
            lookups = [(corpus.tokens[tmpl.feature], tmpl.offset) for tmpl in template]
            if len(template) == 1:
                [(lookup1, offset1)] = lookups
                assert offset1 == 0   # delta for the first feature should always be 0
                sortkey = lambda pos: (lookup1[pos], pos)
            elif len(template) == 2:
                [(lookup1, offset1), (lookup2, offset2)] = lookups
                assert offset1 == 0   # delta for the first feature should always be 0
                sortkey = lambda pos: (lookup1[pos], lookup2[pos+offset2], pos)
            else:
                # the provious sortkeys above are just optimisations of this generic one:
                sortkey = lambda pos: (tuple(lookup[pos+offset] for lookup, offset in lookups), pos)

            with DiskIntArray(index_path) as suffix_array:
                import sort
                sort.quicksort(
                    suffix_array,
                    key = sortkey, 
                    pivotselector = sort.random_pivot, 
                    # pivotselector = sort.median_of_three,
                    # pivotselector = sort.tukey_ninther,
                    cutoff = 100_000,
                )

            logging.info(f"Built index for {template}, with {nr_rows} rows")


################################################################################
## SQLite database for building the index

import sqlite3

class IndexBuilderDB:
    dbfile : Path
    keep_dbfile : bool
    con : sqlite3.Connection
    template : Template
    columns : List[str]

    def __init__(self, dbfile:Path, width:int, keep_dbfile:bool=False):
        self.dbfile = dbfile
        self.keep_dbfile = keep_dbfile
        self.con = sqlite3.connect(dbfile)

        # Switch off journalling etc to speed up database creation
        self.con.execute('pragma synchronous = off')
        self.con.execute('pragma journal_mode = off')

        # Create a table with `width` columns
        self.columns = [f"c{i}" for i in range(width)]
        column_types = [c + ' int' for c in self.columns]
        self.con.execute(f"""
            create table builder(
                {','.join(column_types)},
                primary key ({','.join(self.columns)})
            ) without rowid
        """)

    def insert_rows(self, row_iterator):
        places = ['?' for _ in self.columns]
        self.con.executemany(f"insert or ignore into builder values({','.join(places)})", row_iterator)

    def row_iterator(self) -> Iterator[Tuple[int,...]]:
        return self.con.execute(f"select * from builder order by {','.join(self.columns)}")

    def count_rows_and_instances(self) -> Tuple[int, int]:
        nr_rows = self.con.execute(f"select count(*) from builder").fetchone()[0]
        nr_instances = self.con.execute(f"select count(*) from (select distinct {','.join(self.columns[:-1])} from builder)").fetchone()[0]
        return nr_rows, nr_instances

    def close(self):
        self.con.close()
        if not self.keep_dbfile:
            self.dbfile.unlink()

    def __enter__(self) -> 'IndexBuilderDB':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

