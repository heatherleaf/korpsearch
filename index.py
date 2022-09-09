
import itertools
from pathlib import Path
from typing import Tuple, List, Iterator, Union, Callable
import sys
import logging
import sqlite3

from disk import DiskIntArray, DiskIntArrayBuilder, DiskIntArrayType, InternedString
from corpus import Corpus
from util import progress_bar

try:
    import fast_intersection  # type: ignore
except ModuleNotFoundError:
    print("Module 'fast_intersection' not found.\n"
          "To install, run: 'python setup.py build_ext --inplace'.\n"
          "Using a slow internal implementation instead.\n", 
          file=sys.stderr)


################################################################################
## Templates and instances

class Template:
    def __init__(self, *feature_positions:Tuple[str,int]):
        assert len(feature_positions) > 0
        assert feature_positions[0][-1] == 0
        self._feature_positions : Tuple[Tuple[str,int],...] = feature_positions

    def __bytes__(self) -> bytes:
        return str(self).encode()

    def __str__(self) -> str:
        return '+'.join(feat + str(pos) for feat, pos in self._feature_positions)

    def __iter__(self) -> Iterator[Tuple[str,int]]:
        yield from self._feature_positions

    def __len__(self) -> int:
        return len(self._feature_positions)


class Instance:
    def __init__(self, *values : InternedString):
        self._values : Tuple[InternedString,...] = values

    def values(self) -> Tuple[InternedString,...]:
        return self._values

    def __bytes__(self) -> bytes:
        return b' '.join(map(bytes, self._values))

    def __str__(self) -> str:
        return ' '.join(map(str, self._values))

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
    keys : List[DiskIntArrayType]
    index : DiskIntArrayType
    sets : DiskIntArrayType

    def __init__(self, corpus:Corpus, template:Template):
        self.template = template
        paths = self.indexpaths(corpus, template)
        self.keys = [DiskIntArray(path) for path in paths['keys']]
        self.index = DiskIntArray(paths['index'])
        self.sets = DiskIntArray(paths['sets'])

    def __str__(self) -> str:
        return self.__class__.__name__ + ':' + str(self.template) 

    def __len__(self) -> int:
        return len(self.index)

    def search(self, instance:Instance) -> 'IndexSet':
        set_start : int = self._lookup_instance(instance)
        return IndexSet(self.sets, set_start)

    def _lookup_instance(self, instance:Instance) -> int:
        # binary search
        instance_key : Tuple[int,...] = tuple(s.index for s in instance)
        start : int = 0; end : int = len(self)-1
        while start <= end:
            mid : int = (start + end) // 2
            key : Tuple[int,...] = tuple(keyarray[mid] for keyarray in self.keys)
            if key == instance_key:
                return self.index[mid]
            elif key < instance_key:
                start = mid + 1
            else:
                end = mid - 1
        raise KeyError(f'Instance "{instance}" not found')

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
    def build(corpus:Corpus, template:Template, keep_tmpfiles:bool=False, min_frequency:int=0):
        logging.debug(f"Building index for {template}...")
        paths = Index.indexpaths(corpus, template)
        paths['base'].parent.mkdir(exist_ok=True)

        unary_indexes : List[Index] = []
        if min_frequency > 0 and len(template) > 1:
            unary_indexes = [Index(corpus, Template((feat,0)))
                            for (feat, _pos) in template]

        dbfile : Path = paths['base'].with_suffix('.db.tmp')
        con : sqlite3.Connection = sqlite3.connect(dbfile)

        # Switch off journalling etc to speed up database creation
        con.execute('pragma synchronous = off')
        con.execute('pragma journal_mode = off')

        # Create table - one column per feature, plus one column for the sentence
        features : str = ', '.join(f'feature{i}' for i in range(len(template)))
        feature_types : str = ', '.join(f'feature{i} int' for i in range(len(template)))
        con.execute(f'''
            create table features(
                {feature_types},
                sentence int,
                primary key ({features}, sentence)
            ) without rowid''')

        # Add all features
        def generate_instances(sentence:slice) -> Iterator[Instance]:
            try:
                for k in range(sentence.start, sentence.stop):
                    instance_values = [corpus.words[feat][k+i] for (feat, i) in template]
                    yield Instance(*instance_values)
            except IndexError:
                pass

        skipped_instances : int = 0
        def generate_db_rows() -> Iterator[Tuple[int, ...]]:
            nonlocal skipped_instances
            with progress_bar(corpus.sentences(), "Building database", total=corpus.num_sentences()) as pbar_sentences:
                for n, sentence in enumerate(pbar_sentences, 1):
                    for instance in generate_instances(sentence):
                        if unary_indexes and any(
                                    len(unary.search(Instance(val))) < min_frequency 
                                    for val, unary in zip(instance, unary_indexes)
                                ):
                            skipped_instances += 1
                            continue
                        yield tuple(value.index for value in instance.values()) + (n,)

        places : str = ', '.join('?' for _ in template)
        con.executemany(f'insert or ignore into features values({places}, ?)', generate_db_rows())
        if skipped_instances:
            logging.debug(f"Skipped {skipped_instances} low-frequency instances")

        nr_sentences : int = corpus.num_sentences()
        nr_instances : int = con.execute(f'select count(*) from (select distinct {features} from features)').fetchone()[0]
        nr_rows : int = con.execute(f'select count(*) from features').fetchone()[0]
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
        db_row_iterator = con.execute(f'select * from features order by {features}, sentence')
        for row in progress_bar(db_row_iterator, "Creating index", total=nr_rows):
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

        # Cleanup
        if not keep_tmpfiles:
            dbfile.unlink()


################################################################################
## Index set

IndexSetValuesType = Union[DiskIntArrayType, List[int]]

class IndexSet:
    start : int
    size : int
    values : IndexSetValuesType

    def __init__(self, values:IndexSetValuesType, start:int=0, size:int=-1):
        self.values = values
        self.start = start
        if isinstance(values, list):
            self.size = size if size >= 0 else len(values) - start
        else:
            assert size == -1
            self.size = values[start]
            self.start += 1

    def __len__(self) -> int:
        return self.size

    def __str__(self) -> str:
        MAX = 5
        if len(self) <= MAX:
            return "{" + ", ".join(str(n) for n in self) + "}"
        return f"{{{', '.join(str(n) for n in itertools.islice(self, MAX))}, ... (N={len(self)})}}"

    def __iter__(self) -> Iterator[int]:
        yield from self.values[self.start:self.start+self.size]

    def intersection_update(self, other:'IndexSet'):
        assert not isinstance(self.values, list) \
            and not isinstance(other.values, list)

        self.values = self.intersection(other)
        self.start = 0
        self.size = len(self.values)

    def intersection(self, other:'IndexSet') -> List[int]:
        """Take the intersection of two sorted arrays."""
        arr1, start1, length1 = self.values, self.start, self.size
        arr2, start2, length2 = other.values, other.start, other.size
        try:
            return fast_intersection.intersection(arr1, start1, length1, arr2, start2, length2)
        except NameError:
            pass
        result = []
        k1, k2 = 0, 0
        x1, x2 = arr1[start1], arr2[start2]
        while k1 < length1 and k2 < length2:
            if x1 < x2: 
                k1 += 1
                x1 = arr1[start1 + k1]
            elif x1 > x2:
                k2 += 1
                x2 = arr2[start2 + k2]
            else:
                result.append(x1)
                k1 += 1
                x1 = arr1[start1 + k1]
                k2 += 1
                x2 = arr2[start2 + k2]
        return result

    def filter(self, check:Callable[[int],bool]):
        self.values = [elem for elem in self if check(elem)]
        self.start = 0
        self.size = len(self.values)

    def __contains__(self, elem:int) -> bool:
        values = self.values
        start : int = self.start
        end : int = start + self.size - 1
        while start <= end:
            mid : int = (start + end) // 2
            mid_elem : int = values[mid]
            if mid_elem == elem:
                return True
            elif mid_elem < elem:
                start = mid + 1
            else:
                end = mid - 1
        return False

