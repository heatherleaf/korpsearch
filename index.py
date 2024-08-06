
from typing import Optional, Any, NewType
from collections.abc import Iterator, Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

from disk import InternedString, DiskIntArray
from corpus import Corpus
from indexset import IndexSet
from util import progress_bar, binsearch_range, check_feature, Feature, FValue


################################################################################
## Literals, templates and instances

Instance = NewType('Instance', tuple[InternedString, ...])


@dataclass(frozen=True, order=True)
class Literal:
    negative: bool
    offset: int
    feature: Feature
    value: InternedString

    def __post_init__(self) -> None:
        check_feature(self.feature)

    def __str__(self) -> str:
        return f"{self.feature.decode()}:{self.offset}{'#' if self.negative else '='}{self.value}"

    def test(self, corpus: Corpus, pos: int) -> bool:
        value = corpus.tokens[self.feature][pos + self.offset]
        return (value == self.value) != self.negative

    @staticmethod
    def parse(corpus: Corpus, litstr: str) -> 'Literal':
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
            return Literal(negative, int(offset), feature, corpus.intern(feature, value))
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed literal: {litstr}")


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
    literals: tuple[Literal,...] = ()

    def __init__(self, template: Sequence[TemplateLiteral], literals: Collection[Literal] = []) -> None:
        # We need to use __setattr__ because the class is frozen:
        object.__setattr__(self, 'template', tuple(template))
        object.__setattr__(self, 'literals', tuple(sorted(set(literals))))
        object.__setattr__(self, 'size', len(self.template))
        try:
            assert self.template == tuple(sorted(set(self.template))), f"Unsorted template"
            assert self.literals == tuple(sorted(self.literals)),      f"Duplicate literal(s)"
            assert len(self.template) > 0,                             f"Empty template"
            assert min(t.offset for t in self.template) == 0,          f"Minimum offset must be 0"
            if self.literals:
                assert all(lit.negative for lit in self.literals),     f"Positive template literal(s)"
                assert all(0 <= lit.offset <= self.maxdelta() 
                           for lit in self.literals),                  f"Literal offset must be within 0...{self.maxdelta()}"
        except AssertionError:
            raise ValueError(f"Invalid template: {self}")

    def maxdelta(self) -> int:
        return max(t.offset for t in self.template)

    def __str__(self) -> str:
        return '+'.join(map(str, self.template + self.literals))

    def querystr(self) -> str:
        offsets = [lit.offset for lit in self.template] + [lit.offset for lit in self.literals]
        min_offset = min(offsets)
        max_offset = max(offsets)
        tokens: list[str] = []
        for offset in range(min_offset, max_offset+1):
            tok = ','.join('?' + l.feature.decode() for l in self.template if l.offset == offset)
            lit = ','.join(f'{l.feature.decode()}{"≠" if l.negative else "="}"{l.value}"' 
                           for l in self.literals if l.offset == offset)
            if lit:
                tokens.append(tok + '|' + lit)
            else:
                tokens.append(tok)
        return ''.join('[' + tok + ']' for tok in tokens)

    def __iter__(self) -> Iterator[TemplateLiteral]:
        return iter(self.template)

    def __len__(self) -> int:
        return self.size

    def instantiate(self, corpus: Corpus, pos: int) -> Optional['Instance']:
        if not all(lit.test(corpus, pos) for lit in self.literals):
            return None
        return Instance(tuple(corpus.tokens[tmpl.feature][pos + tmpl.offset] for tmpl in self.template))

    @staticmethod
    def parse(corpus: Corpus, template_str: str) -> 'Template':
        try:
            literals: list[Literal] = []
            template: list[TemplateLiteral] = []
            for litstr in template_str.split('+'):
                try:
                    literals.append(Literal.parse(corpus, litstr))
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
        return binsearch_range(0, len(self)-1, instance[0], self.search_key())


class BinaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
        super().__init__(corpus, template)

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 2, f"BinaryIndex instance must have length 2: {instance}"
        tmpl1, tmpl2 = self.template.template
        offset1, offset2 = tmpl1.offset, tmpl2.offset
        features1 = self.corpus.tokens[tmpl1.feature]
        features2 = self.corpus.tokens[tmpl2.feature]
        index = self.index.array
        def search_key(k: int) -> Instance:
            return Instance((features1[index[k] + offset1], features2[index[k] + offset2]))
        return binsearch_range(0, len(self)-1, instance, search_key)

