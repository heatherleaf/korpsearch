
from typing import Optional, Any, NewType
from collections.abc import Iterator, Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import logging

from disk import InternedString, InternedRange, DiskIntArray
from corpus import Corpus
from indexset import IndexSet
from util import progress_bar, binsearch_range, check_feature, Feature, FValue


################################################################################
## Literals, templates and instances

Instance = NewType('Instance', tuple[InternedRange, ...])

def mkInstance(strs: Sequence[InternedString | InternedRange]) -> Instance:
    return Instance(tuple(s if isinstance(s, tuple) else (s, s) for s in strs))


@dataclass(frozen=True, order=True)
class KnownLiteral:
    negative: bool
    offset: int
    feature: Feature
    value: InternedString
    value2: InternedString
    corpus: Corpus = field(compare=False)

    def __post_init__(self) -> None:
        check_feature(self.feature)

    def __str__(self) -> str:
        value = self.corpus.lookup_value(self.feature, self.value).decode()
        value2 = self.corpus.lookup_value(self.feature, self.value2).decode()
        if self.is_prefix():
            return f"{self.feature.decode()}:{self.offset}{'#' if self.negative else '='}{value}-{value2}"
        else:
            return f"{self.feature.decode()}:{self.offset}{'#' if self.negative else '='}{value}"

    def is_prefix(self) -> bool:
        return self.value != self.value2

    # unoptimized version for use with Query.check_position
    def check_position(self, corpus: Corpus, pos: int) -> bool:
        value = corpus.tokens[self.feature][pos + self.offset]
        return (value >= self.value and value <= self.value2) != self.negative

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
            interned_value = corpus.intern(feature, value)
            return KnownLiteral(negative, int(offset), feature, interned_value, interned_value, corpus)
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed literal: {litstr}")


@dataclass(frozen=True, order=True)
class DisjunctiveGroup:
    negative: bool
    offset: int
    literals: tuple[KnownLiteral, ...]

    def __str__(self) -> str:
        return "|".join(map(str, self.literals))

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
            literal = ','.join(f'{lit.feature.decode()}{"≠" if lit.negative else "="}"{val}"'
                               for lit in self.literals if lit.offset == offset
                               for val in [lit.corpus.lookup_value(lit.feature, lit.value).decode()])
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
        return mkInstance(tuple(corpus.tokens[tmpl.feature][pos + tmpl.offset] for tmpl in self.template))

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

    def getconfig(self) -> dict[str, Any]:
        return self.index.config

    def close(self) -> None:
        self.index.close()

    def search(self, instance: Instance, offset: int = 0) -> IndexSet:
        set_start, set_end = self.lookup_instance(instance)
        set_size = set_end - set_start + 1
        iset = IndexSet(self.index, path=self.path, start=set_start, size=set_size, offset=offset)
        return iset

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        raise NotImplementedError("Must be overridden by a subclass")

    def sanity_check(self) -> None:
        logging.info(f"Checking search index: {self.template}")
        prev_pos = -1
        prev_instance = None
        for i in progress_bar(range(len(self.index)), desc="Checking index"):
            pos = self.index.array[i]
            instance = self.template.instantiate(self.corpus, pos)
            assert instance is not None
            if prev_instance is not None:
                assert prev_instance <= instance, f"Index position {i}: {prev_instance} > {instance}"
                if prev_instance == instance:
                    assert prev_pos < pos, f"Index position {i}: {prev_instance} == {instance} but {prev_pos} >= {pos}"
            prev_pos = pos
            prev_instance = instance


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

    def search_key(self) -> Callable[[int], InternedString]:
        tmpl = self.template.template[0]
        features = self.corpus.tokens[tmpl.feature]
        offset = tmpl.offset
        index = self.index.array
        return lambda k: features[index[k] + offset]

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 1, f"UnaryIndex instance must have length 1: {instance}"
        ((value1, value2),) = instance
        error = (value1 == value2)
        return binsearch_range(0, len(self)-1, value1, value2, self.search_key(), error=error)


class BinaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
        super().__init__(corpus, template)

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 2, f"BinaryIndex instance must have length 2: {instance}"
        ((left1, left2), (right1, right2)) = instance
        if left1 != left2:
            raise KeyError("BinaryIndex cannot have a range as left value")
        tmpl1, tmpl2 = self.template.template
        offset1, offset2 = tmpl1.offset, tmpl2.offset
        features1 = self.corpus.tokens[tmpl1.feature]
        features2 = self.corpus.tokens[tmpl2.feature]
        index = self.index.array
        def search_key(k: int) -> InternedRange:
            return (features1[index[k] + offset1], features2[index[k] + offset2])
        error = (right1 == right2)
        return binsearch_range(0, len(self)-1, (left1, right1), (left2, right2), search_key, error=error)

