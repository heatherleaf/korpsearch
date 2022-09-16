
from pathlib import Path
from typing import Tuple, List, Iterator, Callable, Union, Sequence
from types import TracebackType
import logging

from disk import DiskIntArray, DiskIntArrayBuilder, InternedString
from corpus import Corpus
from indexset import IndexSet
from util import progress_bar


################################################################################
## Templates and instances

class Template:
    _feature_positions : Tuple[Tuple[str,int],...]

    def __init__(self, feature_positions:Sequence[Tuple[str,int]]):
        assert len(feature_positions) > 0
        assert feature_positions[0][-1] == 0
        assert all(pos >= 0 for _, pos in feature_positions)
        self._feature_positions = tuple(feature_positions)

    def maxdelta(self):
        return max(pos for _feat, pos in self)

    def __bytes__(self) -> bytes:
        return str(self).encode()

    def __str__(self) -> str:
        return '+'.join(f"{feat}:{pos}" for feat, pos in self)

    def __iter__(self) -> Iterator[Tuple[str,int]]:
        yield from self._feature_positions

    def __len__(self) -> int:
        return len(self._feature_positions)

    @staticmethod
    def parse(template_str:str) -> 'Template':
        template = [tuple(feat_dist.split(':')) for feat_dist in template_str.split('+')]
        try:
            return Template([(feat, int(dist)) for (feat, dist) in template])
        except (ValueError, AssertionError):
            raise ValueError("Ill-formed template: it should be on the form pos:0 or word:0+pos:2")


class Instance:
    _values : Tuple[InternedString,...]

    def __init__(self, values : Sequence[InternedString]):
        assert len(values) > 0
        self._values = tuple(values)

    def values(self) -> Tuple[InternedString,...]:
        return self._values

    def __bytes__(self) -> bytes:
        return b' '.join(map(bytes, self._values))

    def __str__(self) -> str:
        return '+'.join(map(str, self._values))

    def __iter__(self) -> Iterator[InternedString]:
        yield from self._values

    def __len__(self) -> int:
        return len(self._values)


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
            [(tmpl_feat, tmpl_delta)] = list(template)
            self.search_key = lambda k: \
                corpus.tokens[tmpl_feat][index[k] + tmpl_delta]
        elif len(self.template) == 2:
            [(tmpl_feat1, tmpl_delta1), (tmpl_feat2, tmpl_delta2)] = list(template)
            self.search_key = lambda k: (
                corpus.tokens[tmpl_feat1][index[k] + tmpl_delta1],
                corpus.tokens[tmpl_feat2][index[k] + tmpl_delta2],
            )
        else:
            # The above two are just optimisations of the following generic search key:
            self.search_key = lambda k: tuple(
                corpus.tokens[feat][index[k] + delta] 
                for feat, delta in template
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
        instance_key = instance.values()
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
    def build(corpus:Corpus, template:Template, min_frequency:int=0, no_sentence_break:bool=True, use_sqlite:bool=False, keep_tmpfiles:bool=False):
        logging.debug(f"Building index for {template}...")
        index_path = Index.indexpath(corpus, template)
        index_path.parent.mkdir(exist_ok=True)

        maxdelta = template.maxdelta()
        index_size = len(corpus) - maxdelta

        unary_indexes : List[Index] = []
        if min_frequency > 0 and len(template) > 1:
            unary_indexes = [
                Index(corpus, Template([(feat, 0)])) 
                for (feat, _pos) in template
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

        def generate_positions():
            with progress_bar(range(index_size), desc="Collecting positions") as pbar:
                if no_sentence_break and maxdelta > 0:
                    # Don't generate instances that cross sentence borders
                    sentences = corpus.sentences()
                    sent = next(sentences)
                    start, stop = sent.start, sent.stop-maxdelta
                    for pos in pbar:
                        if pos >= stop:
                            sent = next(sentences)
                            start, stop = sent.start, sent.stop-maxdelta
                        if start <= pos:
                            instance_values = [corpus.tokens[feat][pos+i] for (feat, i) in template]
                            if all(instance_values):
                                yield pos, instance_values
                else:
                    for pos in pbar:
                        instance_values = [corpus.tokens[feat][pos+i] for (feat, i) in template]
                        if all(instance_values):
                            yield pos, instance_values

        if use_sqlite:
            dbfile = index_path.parent / 'index.db.tmp'
            with IndexBuilderDB(dbfile, len(template)+1, keep_dbfile=keep_tmpfiles) as con:
                skipped_instances : int = 0
                def generate_db_rows() -> Iterator[Tuple[int, ...]]:
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
            feature_positions = list(template)
            feat, delta = feature_positions[0]
            assert delta == 0   # delta for the first feature should always be 0
            text1 = corpus.tokens[feat]
            if len(feature_positions) == 1:
                sortkey = lambda pos: (text1[pos], pos)
            elif len(feature_positions) == 2:
                feat, delta = feature_positions[1]
                text2 = corpus.tokens[feat]
                sortkey = lambda pos: (text1[pos], text2[pos+delta], pos)
            else:
                # the provious sortkeys above are just optimisations of this generic one:
                sortkey = lambda pos: (tuple(corpus.tokens[feat][pos+delta] for feat, delta in feature_positions), pos)

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

