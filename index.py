
from typing import Optional, Any, NewType
from collections.abc import Iterator, Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from pyroaring import BitMap

from disk import Symbol, SymbolRange, IntArray, IntBytesMap
from corpus import Corpus
from util import add_suffix, progress_bar, binsearch_range, check_feature, Feature, FValue


################################################################################
## Literals, templates and instances

Instance = NewType('Instance', Sequence[Symbol|SymbolRange])


@dataclass(frozen=True, order=True)
class KnownLiteral:
    negative: bool
    offset: int
    feature: Feature
    value: Symbol | SymbolRange
    corpus: Corpus = field(compare=False)

    def __post_init__(self) -> None:
        check_feature(self.feature)

    def __str__(self) -> str:
        if isinstance(self.value, tuple):
            value0 = self.corpus.lookup_symbol(self.feature, self.value[0]).decode()
            value1 = self.corpus.lookup_symbol(self.feature, self.value[1]).decode()
            return f"{self.feature.decode()}:{self.offset}{'#' if self.negative else '='}{value0}-{value1}"
        else:
            value = self.corpus.lookup_symbol(self.feature, self.value).decode()
            return f"{self.feature.decode()}:{self.offset}{'#' if self.negative else '='}{value}"

    def is_prefix(self) -> bool:
        return isinstance(self.value, tuple)

    # unoptimized version for use with Query.check_position
    def check_position(self, corpus: Corpus, pos: int) -> bool:
        value = corpus.tokens[self.feature][pos + self.offset]
        if isinstance(self.value, tuple):
            return self.negative != (self.value[0] <= value <= self.value[1])
        else:
            return self.negative != (self.value == value)

    def test(self, pos: int) -> bool:
        value = self.corpus.tokens[self.feature][pos + self.offset]
        return (value == self.value) != self.negative

    @staticmethod
    def parse(corpus: Corpus, litstr: str) -> 'KnownLiteral':
        try:
            featstr, rest = litstr.split(':')
            feature = Feature(featstr.lower().encode())
            check_feature(feature)
            try:
                offset, valstr = rest.split('=')
                negative = False
            except ValueError:
                offset, valstr = rest.split('#')
                negative = True
            value = FValue(valstr.encode())
            symbol = corpus.get_symbol(feature, value)
            return KnownLiteral(negative, int(offset), feature, symbol, corpus)
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed literal: {litstr}")


@dataclass(frozen=True, order=True)
class DisjunctiveGroup:
    negative: bool
    offset: int
    literals: tuple[KnownLiteral, ...]

    def __str__(self) -> str:
        return ("|".join(map(str, self.literals)) if len(self.literals) < 10 else
                "|".join(map(str, self.literals[:3])) + "|...|" + "|".join(map(str, self.literals[-3:])))

    # unoptimized version for use with Query.check_position
    def check_position(self, corpus: Corpus, pos: int) -> bool:
        return any(lit.check_position(corpus, pos) for lit in self.literals)

    def is_prefix(self) -> bool:
        return any(lit.is_prefix() for lit in self.literals)

    @staticmethod
    def create(literals: tuple[KnownLiteral, ...]) -> 'DisjunctiveGroup':
        negative = any(lit.negative for lit in literals)
        offset = min(lit.offset for lit in literals)
        return DisjunctiveGroup(negative, offset, literals)


@dataclass(frozen=True, order=True)
class TemplateLiteral:
    offset: int
    feature: Feature

    def __post_init__(self) -> None:
        check_feature(self.feature)

    def __str__(self) -> str:
        return f"{self.feature.decode()}:{self.offset}"

    @staticmethod
    def parse(litstr: str) -> 'TemplateLiteral':
        try:
            featstr, offset = litstr.split(':')
            feature = Feature(featstr.lower().encode())
            check_feature(feature)
            return TemplateLiteral(int(offset), feature)
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed template literal: {litstr}")


@dataclass(frozen=True, order=True, init=False)
class Template:
    size: int  # Having 'size' first means shorter templates are ordered before longer
    template: tuple[TemplateLiteral,...]
    literals: tuple[KnownLiteral,...] = ()

    def __init__(self, template: Sequence[TemplateLiteral], literals: Collection[KnownLiteral] = []) -> None:
        # We need to use __setattr__ because the class is frozen:
        object.__setattr__(self, 'template', tuple(template))
        object.__setattr__(self, 'literals', tuple(sorted(set(literals))))
        object.__setattr__(self, 'size', len(self.template))
        try:
            assert self.template == tuple(sorted(set(self.template))), f"Unsorted template"
            assert self.literals == tuple(sorted(self.literals)),      f"Duplicate literal(s)"
            assert len(self.template) > 0,                             f"Empty template"
            assert self.min_offset() == 0,                             f"Minimum offset must be 0"
            assert all(lit.negative for lit in self.literals),         f"Positive template literal(s)"
        except AssertionError:
            raise ValueError(f"Invalid template: {self}")

    def offsets(self) -> set[int]:
        return {lit.offset for lit in self.template + self.literals}

    def min_offset(self) -> int:
        return min(self.offsets())

    def max_offset(self) -> int:
        return max(self.offsets())

    def __str__(self) -> str:
        return '+'.join(map(str, self.template + self.literals))

    def querystr(self) -> str:
        tokens: list[str] = []
        for offset in range(self.min_offset(), self.max_offset()+1):
            token = ','.join('?' + lit.feature.decode() for lit in self.template if lit.offset == offset)
            literal = ','.join(
                f'{lit.feature.decode()}{"≠" if lit.negative else "="}"{val}"'
                for lit in self.literals if lit.offset == offset
                for lval in [lit.value[0] if isinstance(lit.value, tuple) else lit.value]
                for val in [lit.corpus.lookup_symbol(lit.feature, lval).decode()]
            )
            if literal:
                tokens.append(token + '|' + literal)
            else:
                tokens.append(token)
        return ''.join('[' + tok + ']' for tok in tokens)

    def __iter__(self) -> Iterator[TemplateLiteral]:
        return iter(self.template)

    def __len__(self) -> int:
        return self.size

    def instantiate(self, corpus: Corpus, pos: int) -> Optional['Instance']:
        if not all(lit.test(pos) for lit in self.literals):
            return None
        return Instance(tuple(corpus.tokens[tmpl.feature][pos + tmpl.offset] for tmpl in self.template))

    @staticmethod
    def parse(corpus: Corpus, template_str: str) -> 'Template':
        try:
            literals: list[KnownLiteral] = []
            template: list[TemplateLiteral] = []
            for litstr in template_str.split('+'):
                try:
                    literals.append(KnownLiteral.parse(corpus, litstr))
                except ValueError:
                    template.append(TemplateLiteral.parse(litstr))
            return Template(template, literals)
        except (ValueError, AssertionError):
            raise ValueError(
                "Ill-formed template - it should be on the form pos:0 or word:0+pos:2: " + template_str
            )


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
            start, end = self.get_instance_range(instance)
        except ValueError:
            return BitMap()
        small = self.lookup_smallset(start, end)
        big = self.lookup_bigset(start, end)
        result = small | big
        if offset:
            result = result.shift(-offset)
        logging.debug(f"Search: {len(small)} smallsets + {len(big)} bigsets = {len(result)}")
        return result

    def get_instance_range(self, instance: Instance) -> tuple[int, int]:
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

    def get_instance_range(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 1, f"UnaryIndex instance must have length 1: {instance}"
        (value,) = instance
        return value if isinstance(value, tuple) else (value, value)

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

    def get_instance_range(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 2, f"BinaryIndex instance must have length 2: {instance}"
        (left, right) = instance
        if isinstance(left, (tuple, list)):
            raise ValueError("BinaryIndex cannot have a range as left value")
        left <<= 32
        if isinstance(right, tuple):
            return (left + right[0], left + right[1])
        else:
            return (left + right, left + right)

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

