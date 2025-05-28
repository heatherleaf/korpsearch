
from typing import Optional, Any, NewType
from collections.abc import Iterator, Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from mmap import mmap
import logging

from pyroaring import BitMap

from disk import Symbol, IntArray, SymbolArray, SymbolCollection
from corpus import Corpus
from util import binsearch, check_feature, Feature, FValue, add_suffix


################################################################################
## Literals, templates and instances

Instance = NewType('Instance', tuple[Symbol, ...])


@dataclass(frozen=True, order=True)
class KnownLiteral:
    negative: bool
    offset: int
    feature: Feature
    value: Symbol
    corpus: Corpus = field(compare=False)

    def __post_init__(self) -> None:
        check_feature(self.feature)

    def __str__(self) -> str:
        value = self.corpus.lookup_symbol(self.feature, self.value).decode()
        return f"{self.feature.decode()}:{self.offset}{'#' if self.negative else '='}{value}"

    # unoptimized version for use with Query.check_position
    def check_position(self, corpus: Corpus, pos: int) -> bool:
        value = corpus.tokens[self.feature][pos + self.offset]
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
            interned_value = corpus.get_symbol(feature, value)
            return KnownLiteral(negative, int(offset), feature, interned_value, corpus)
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
                               for val in [lit.corpus.lookup_symbol(lit.feature, lit.value).decode()])
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
        if not all(lit.check_position(corpus, pos) for lit in self.literals):
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
## Implemented as a sorted array of interned strings
## This is a kind of modified suffix array - a "pruned" SA if you like

class Index:
    dir_suffix: str = '.indexes'
    corpus: Corpus
    template: Template
    index: IntArray
    bitmaps: mmap
    path: Path

    def __init__(self, corpus: Corpus, template: Template) -> None:
        self.corpus = corpus
        self.template = template
        self.path = self.indexpath(corpus, template)
        self.index = IntArray(self.path)
        with open(self.path, 'r+b') as file:
            self.bitmaps = mmap(file.fileno(), 0)

    def __str__(self) -> str:
        return self.__class__.__name__ + ':' + str(self.template)

    def __len__(self) -> int:
        return len(self.index)

    def __enter__(self) -> 'Index':
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # def getconfig(self) -> dict[str, Any]:
    #     return self.index.config

    def close(self) -> None:
        self.index.close()

    def search(self, instance: Instance, offset: int = 0) -> BitMap:
        bitmap = self.lookup_instance(instance)
        bitmap.shift(offset)
        return bitmap

    def lookup_instance(self, instance: Instance) -> BitMap:
        raise NotImplementedError("Must be overridden by a subclass")

    def sanity_check(self) -> None:
        logging.info("Checking not implemented yet")
        # logging.info(f"Checking search index: {self.template}")
        # prev_pos = -1
        # prev_instance = None
        # for i in progress_bar(range(len(self.index)), desc="Checking index"):
        #     pos = self.index.array[i]
        #     instance = self.template.instantiate(self.corpus, pos)
        #     assert instance is not None
        #     if prev_instance is not None:
        #         assert prev_instance <= instance, f"Index position {i}: {prev_instance} > {instance}"
        #         if prev_instance == instance:
        #             assert prev_pos < pos, f"Index position {i}: {prev_instance} == {instance} but {prev_pos} >= {pos}"
        #     prev_pos = pos
        #     prev_instance = instance


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

    def lookup_instance(self, instance: Instance) -> BitMap:
        assert len(instance) == 1, f"UnaryIndex instance must have length 1: {instance}"
        (value,) = instance
        i = binsearch(0, len(self)-1, value, lambda k: self.index.array[k], error=True)
        bitmap_str = self.bitmaps.from_index(i)
        print(instance, value, i, bitmap_str)
        return BitMap.deserialize(bitmap_str)


class BinaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
        super().__init__(corpus, template)

    def lookup_instance(self, instance: Instance) -> BitMap:
        assert len(instance) == 2, f"BinaryIndex instance must have length 2: {instance}"
        (left, right) = instance
        tmpl1, tmpl2 = self.template.template
        offset1, offset2 = tmpl1.offset, tmpl2.offset
        features1 = self.corpus.tokens[tmpl1.feature]
        features2 = self.corpus.tokens[tmpl2.feature]
        index = self.index.raw()
        def search_key(k: int) -> tuple[Symbol, Symbol]:
            return (features1[index[k] + offset1], features2[index[k] + offset2])
        i = binsearch(0, len(self)-1, (left, right), search_key, error=True)
        bitmap_str = self.index.interned_bytes(self.index[i])
        return BitMap.deserialize(bitmap_str)

