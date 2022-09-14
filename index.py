
from pathlib import Path
from typing import Tuple, List, Iterator, Callable, Union
from types import TracebackType
import logging

from disk import DiskIntArray, DiskIntArrayBuilder, InternedString
from corpus import Corpus
from indexset import IndexSet
from util import progress_bar


################################################################################
## Templates and instances

class Template:
    def __init__(self, *feature_positions:Tuple[str,int]):
        assert len(feature_positions) > 0
        assert feature_positions[0][-1] == 0
        self._feature_positions : Tuple[Tuple[str,int],...] = feature_positions

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
            return Template(*[(feat, int(dist)) for (feat, dist) in template])
        except (ValueError, AssertionError):
            raise ValueError("Ill-formed template: it should be on the form pos:0 or word:0+pos:2")


class Instance:
    def __init__(self, *values : InternedString):
        self._values : Tuple[InternedString,...] = values

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

class Index:
    dir_suffix : str = '.indexes'

    template : Template
    keys : List[DiskIntArray]
    index : DiskIntArray
    sets : DiskIntArray
    search_key : Callable[[int], Union[int, Tuple[int,...]]]
    # Typing note: as optimisation we use the value (s) instead of a 1-tuple (s,) so the
    # return type is a union of a value and a tuple. But then Pylance can't infer the correct
    # type, so we have to write "# type: ignore" on some lines below.

    def __init__(self, corpus:Corpus, template:Template):
        self.template = template
        paths = self.indexpaths(corpus, template)
        self.keys = [DiskIntArray(path) for path in paths['keys']]
        self.index = DiskIntArray(paths['index'])
        self.sets = DiskIntArray(paths['sets'])

        assert len(self.template) == len(self.keys)
        if len(self.keys) == 1:
            [keyarr] = self.keys
            self.search_key = lambda k: keyarr[k]
        elif len(self.keys) == 2:
            [keyarr1, keyarr2] = self.keys
            self.search_key = lambda k: (keyarr1[k], keyarr2[k])
        else:
            # The above two are just optimisations of the following generic search key:
            self.search_key = lambda k: tuple(
                keyarray[k] for keyarray in self.keys
            )

    def __str__(self) -> str:
        return self.__class__.__name__ + ':' + str(self.template) 

    def __len__(self) -> int:
        return len(self.index)

    def search(self, instance:Instance, offset:int=0) -> IndexSet:
        set_start : int = self.lookup_instance(instance)
        set_size : int = self.sets[set_start]
        return IndexSet(self.sets, set_start+1, set_size)

    def lookup_instance(self, instance:Instance) -> int:
        search_key = self.search_key
        instance_key = tuple(s.index for s in instance)
        if len(instance_key) == 1: instance_key = instance_key[0]

        start, end = 0, len(self)-1
        while start <= end:
            mid = (start + end) // 2
            key = search_key(mid)
            if key == instance_key:
                return self.index[mid]
            elif key < instance_key:  # type: ignore
                start = mid + 1
            else:
                end = mid - 1
        raise KeyError(f'Instance "{instance}" not found')

    def __enter__(self) -> 'Index':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

    def close(self):
        for k in self.keys: k.close()
        self.sets.close()
        self.index.close()

    @staticmethod
    def indexpaths(corpus, template):
        basedir = corpus.path.with_suffix(Index.dir_suffix)
        basepath = basedir / str(template) / str(template)
        return {
            'base': basepath,
            'keys': [basepath.with_suffix(f'.key:{feature}{pos}') for feature, pos in template],
            'index': basepath.with_suffix('.index'),
            'sets': basepath.with_suffix('.sets'),
        }

    @staticmethod
    def build(corpus:Corpus, template:Template, keep_tmpfiles:bool=False, min_frequency:int=0, use_sqlite:bool=True):
        logging.debug(f"Building index for {template}...")
        paths = Index.indexpaths(corpus, template)
        paths['base'].parent.mkdir(exist_ok=True)

        unary_indexes : List[Index] = []
        if min_frequency > 0 and len(template) > 1:
            unary_indexes = [Index(corpus, Template((feat,0))) for (feat, _pos) in template]

        dbfile : Path = paths['base'].with_suffix('.db.tmp')
        with IndexBuilderDB(dbfile, len(template)+1, keep_dbfile=keep_tmpfiles) as con:
            skipped_instances : int = 0
            def generate_db_rows() -> Iterator[Tuple[int, ...]]:
                nonlocal skipped_instances
                with progress_bar(corpus.sentences(), "Building database", total=corpus.num_sentences()) as pbar_sentences:
                    for n, sentence in enumerate(pbar_sentences, 1):
                        for pos in range(sentence.start, sentence.stop - template.maxdelta()):
                            instance_values = [corpus.words[feat][pos+i] for (feat, i) in template]
                            if unary_indexes and any(
                                        len(unary.search(Instance(val))) < min_frequency 
                                        for val, unary in zip(instance_values, unary_indexes)
                                    ):
                                skipped_instances += 1
                            else:
                                yield tuple(value.index for value in instance_values) + (n,)

            con.insert_rows(generate_db_rows())
            if skipped_instances:
                logging.debug(f"Skipped {skipped_instances} low-frequency instances")

            nr_sentences = corpus.num_sentences()
            nr_rows, nr_instances = con.count_rows_and_instances()
            logging.debug(f" --> created instance database, {nr_rows} rows, {nr_instances} instances, {nr_sentences} sentences")

            # Build keys files, index file and sets file
            index_keys = [DiskIntArrayBuilder(path, max_value = len(corpus.strings(feat)))
                    for (feat, _), path in zip(template, paths['keys'])]
            index_sets = DiskIntArrayBuilder(paths['sets'], max_value = nr_sentences+1)
            # nr_rows is the sum of all set sizes, but the .sets file also includes the set sizes,
            # so in some cases we get an OverflowError.
            # This happens e.g. for bnc-20M when building lemma0: nr_rows = 16616400 < 2^24 < 16777216 = nr_rows+nr_sets
            # What we need is max_value = nr_rows + nr_sets; this is a simple hack until we have better solution:
            index_index = DiskIntArrayBuilder(paths['index'], max_value = nr_rows*2)

            nr_keys = nr_elements = 0
            current = None
            set_start = set_size = -1
            # Dummy sentence to account for null pointers:
            index_sets.append(0)
            nr_elements += 1
            for row in progress_bar(con.row_iterator(), "Creating index", total=nr_rows):
                key = row[:-1]
                sent = row[-1]
                if current != key:
                    if set_start >= 0:
                        assert set_size > 0
                        # Now the set is full, and we can write the size of the set at its beginning
                        index_sets[set_start] = set_size
                    for builder, k in zip(index_keys, key):
                        builder.append(k)
                    # Add a placeholder for the size of the set
                    set_start, set_size = len(index_sets), 0
                    index_sets.append(set_size)
                    index_index.append(set_start)
                    nr_elements += 1
                    current = key
                    nr_keys += 1
                index_sets.append(sent)
                set_size += 1
                nr_elements += 1
            # Write the size of the final set at its beginning
            index_sets[set_start] = set_size
            logging.info(f"Built index for {template}, with {nr_keys} keys, {nr_elements} set elements")

            for k in index_keys: k.close()
            index_sets.close()
            index_index.close()


################################################################################
## Alternative: implemented as a suffix array

class SAIndex(Index):
    dir_suffix = '.sa-indexes'

    corpus : Corpus
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
                corpus.words[tmpl_feat][index[k] + tmpl_delta]
        elif len(self.template) == 2:
            [(tmpl_feat1, tmpl_delta1), (tmpl_feat2, tmpl_delta2)] = list(template)
            self.search_key = lambda k: (
                corpus.words[tmpl_feat1][index[k] + tmpl_delta1],
                corpus.words[tmpl_feat2][index[k] + tmpl_delta2],
            )
        else:
            # The above two are just optimisations of the following generic search key:
            self.search_key = lambda k: tuple(
                corpus.words[feat][index[k] + delta] 
                for feat, delta in template
            )

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
        basepath = corpus.path.with_suffix(SAIndex.dir_suffix)
        return basepath / str(template) / str(template)

    @staticmethod
    def build(corpus:Corpus, template:Template, keep_tmpfiles:bool=False, min_frequency:int=0, use_sqlite:bool=False):
        logging.debug(f"Building index for {template}...")
        index_path = SAIndex.indexpath(corpus, template)
        index_path.parent.mkdir(exist_ok=True)

        maxdelta = template.maxdelta()
        index_size = len(corpus) - maxdelta

        unary_indexes : List[SAIndex] = []
        if min_frequency > 0 and len(template) > 1:
            unary_indexes = [SAIndex(corpus, Template((feat,0))) for (feat, _pos) in template]

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
                if maxdelta == 0:
                    yield from pbar
                else:
                    # Don't generate instances that cross sentence borders
                    sentences = corpus.sentences()
                    sent = next(sentences)
                    start, stop = sent.start, sent.stop-maxdelta
                    for pos in pbar:
                        if pos >= stop:
                            sent = next(sentences)
                            start, stop = sent.start, sent.stop-maxdelta
                        if start <= pos:
                            yield pos

        if use_sqlite:
            dbfile = index_path.parent / 'index.db.tmp'
            with IndexBuilderDB(dbfile, len(template)+1, keep_dbfile=keep_tmpfiles) as con:
                skipped_instances : int = 0
                def generate_db_rows() -> Iterator[Tuple[int, ...]]:
                    nonlocal skipped_instances
                    for pos in generate_positions():
                        instance_values = [corpus.words[feat][pos+i] for (feat, i) in template]
                        if unary_indexes and not all_unary_min_frequency(instance_values):
                            skipped_instances += 1
                        else:
                            yield tuple(value.index for value in instance_values) + (pos,)

                con.insert_rows(generate_db_rows())
                if skipped_instances:
                    logging.debug(f"Skipped {skipped_instances} low-frequency instances")

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
                    for pos in generate_positions():
                        instance_values = [corpus.words[feat][pos+i] for (feat, i) in template]
                        if unary_indexes and not all_unary_min_frequency(instance_values):
                            skipped_instances += 1
                        else:
                            suffix_array.append(pos)
                    if skipped_instances:
                        logging.debug(f"Skipped {skipped_instances} low-frequency instances")
                else:
                    for pos in generate_positions():
                        suffix_array.append(pos)
                nr_rows = len(suffix_array)

            # sort the suffix array
            feature_positions = list(template)
            feat, delta = feature_positions[0]
            assert delta == 0   # delta for the first feature should always be 0
            text1 = corpus.words[feat]
            if len(feature_positions) == 1:
                sortkey = lambda pos: (text1[pos], pos)
            elif len(feature_positions) == 2:
                feat, delta = feature_positions[1]
                text2 = corpus.words[feat]
                sortkey = lambda pos: (text1[pos], text2[pos+delta], pos)
            else:
                # the provious sortkeys above are just optimisations of this generic one:
                sortkey = lambda pos: (tuple(corpus.words[feat][pos+delta] for feat, delta in feature_positions), pos)

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
        nr_instances = self.con.execute(f"select count(*) from (select distinct {','.join(self.columns)} from builder)").fetchone()[0]
        return nr_rows, nr_instances

    def close(self):
        self.con.close()
        if not self.keep_dbfile:
            self.dbfile.unlink()

    def __enter__(self) -> 'IndexBuilderDB':
        return self

    def __exit__(self, exc_type:BaseException, exc_val:BaseException, exc_tb:TracebackType):
        self.close()

