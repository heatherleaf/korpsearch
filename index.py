
from typing import Any
from collections.abc import Callable
from pathlib import Path
import json
import logging

from pyroaring import BitMap

from disk import IntArray, IntBytesMap, SymbolRange, SymbolList
from corpus import Corpus
from literals import Template, Instance
from util import add_suffix, progress_bar, binsearch_range


################################################################################
## Inverted sentence index
## Implemented as a sorted array of symbols (interned strings)
## This is a kind of modified suffix array - a "pruned" SA if you like

class Index:
    dir_suffix: str = '.indexes'
    corpus: Corpus
    template: Template
    smallsets: IntArray | None
    bigsets: IntBytesMap | None
    path: Path

    def __init__(self, corpus: Corpus, template: Template) -> None:
        self.corpus = corpus
        self.template = template
        self.path = self.indexpath(corpus, template)
        try:
            self.smallsets = IntArray(self.path)
        except FileNotFoundError:
            self.smallsets = None
        try:
            self.bigsets = IntBytesMap(self.path)
        except FileNotFoundError:
            self.bigsets = None
        if self.smallsets is self.bigsets is None:
            raise FileNotFoundError(f"Index does not exist: {self.path}")

    def __str__(self) -> str:
        return self.__class__.__name__ + ':' + str(self.template)

    def __len__(self) -> int:
        return self.getconfig()['size']

    def __enter__(self) -> 'Index':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def getconfig(self) -> dict[str, Any]:
        try:
            with open(self.getconfigpath(self.path)) as configfile:
                return json.load(configfile)
        except FileNotFoundError:
            assert self.smallsets
            return self.smallsets.getconfig()

    def close(self) -> None:
        if self.smallsets: self.smallsets.close()
        if self.bigsets: self.bigsets.close()

    def search(self, instance: Instance, offset: int = 0) -> BitMap:
        try:
            ranges = self.get_instance_range(instance)
        except ValueError:
            return BitMap()
        result = BitMap()
        for start, end in ranges:
            small = self.lookup_smallset(start, end)
            big = self.lookup_bigset(start, end)
            result |= small | big
        if offset:
            result = result.shift(-offset)
        return result

    def get_instance_range(self, instance: Instance) -> list[tuple[int, int]]:
        raise NotImplementedError("Must be overridden by a subclass")

    def get_search_key(self) -> Callable[[int], int]:
        raise NotImplementedError("Must be overridden by a subclass")

    def lookup_smallset(self, start_key: int, end_key: int) -> BitMap:
        if self.smallsets:
            try:
                search_key = self.get_search_key()
                error = (start_key == end_key)
                start, end = binsearch_range(0, len(self.smallsets)-1, start_key, end_key, search_key, error=error)
                if 0 <= start <= end < len(self.smallsets):
                    return BitMap(self.smallsets.slice(start, end+1))
            except (KeyError, IndexError):
                pass
        return BitMap()

    def lookup_bigset(self, start_key: int, end_key: int) -> BitMap:
        if self.bigsets:
            try:
                if start_key == end_key:
                    bmap = self.bigsets[start_key]
                    return BitMap.deserialize(bmap)
                else:
                    bmaps = self.bigsets.slice(start_key, end_key)
                    logging.debug(f"Found {len(bmaps)} bitmaps between {start_key}..{end_key}")
                    return BitMap.union(*(BitMap.deserialize(bm) for bm in bmaps))
            except (KeyError, IndexError):
                pass
        return BitMap()

    def sanity_check(self) -> None:
        logging.info(f"Checking search index: {self.template}")
        if self.bigsets:
            self.bigsets.sanity_check()
        if self.smallsets:
            prev_pos = -1
            prev_instance = None
            for i in progress_bar(range(len(self.smallsets)), desc="Checking index"):
                pos = self.smallsets.array[i]
                instance = self.template.instantiate(self.corpus, pos)
                assert instance is not None
                if prev_instance is not None:
                    assert prev_instance <= instance, f"Index position {i}: {prev_instance} > {instance}"  # type: ignore
                    if prev_instance == instance:
                        assert prev_pos < pos, f"Index position {i}: {prev_instance} == {instance} but {prev_pos} >= {pos}"
                prev_pos = pos
                prev_instance = instance


    @staticmethod
    def getconfigpath(path: Path) -> Path:
        return add_suffix(path, '.cfg')

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


class UnaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 1, f"UnaryIndex templates must have length 1: {template}"
        super().__init__(corpus, template)

    def get_instance_range(self, instance: Instance) -> list[tuple[int, int]]:
        assert len(instance) == 1, f"UnaryIndex instance must have length 1: {instance}"
        (value,) = instance
        return [
            v if isinstance(v, SymbolRange) else (v, v)
            for v in (value.symbols if isinstance(value, SymbolList) else [value])
        ]

    def get_search_key(self) -> Callable[[int], int]:
        assert self.smallsets
        tmpl = self.template.template[0]
        features = self.corpus.tokens[tmpl.feature]
        offset = tmpl.offset
        index = self.smallsets.array
        def search_key(k: int) -> int:
            return features[index[k] + offset]
        return search_key


class BinaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
        super().__init__(corpus, template)

    def get_instance_range(self, instance: Instance) -> list[tuple[int, int]]:
        assert len(instance) == 2, f"BinaryIndex instance must have length 2: {instance}"
        (left, right) = instance
        if isinstance(left, tuple):
            raise ValueError("BinaryIndex cannot have a range as left value")
        if isinstance(left, list) and isinstance(right, list):
            raise ValueError("BinaryIndex cannot have lists for both left and right values")
        return [
            (l32+r0, l32+r1)
            for l in (left.symbols if isinstance(left, SymbolList) else [left])
            for l32 in [l << 32]
            for r in (right.symbols if isinstance(right, SymbolList) else [right])
            for (r0, r1) in [(r, r) if isinstance(r, int) else r]
        ]

    def get_search_key(self) -> Callable[[int], int]:
        assert self.smallsets
        tmpl1, tmpl2 = self.template.template
        offset1, offset2 = tmpl1.offset, tmpl2.offset
        features1 = self.corpus.tokens[tmpl1.feature]
        features2 = self.corpus.tokens[tmpl2.feature]
        index = self.smallsets.array
        def search_key(k: int) -> int:
            key1 = features1[index[k] + offset1]
            key2 = features2[index[k] + offset2]
            return (key1 << 32) + key2
        return search_key

