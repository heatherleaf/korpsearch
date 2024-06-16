
from typing import NamedTuple, Optional, Any
from collections.abc import Iterator, Callable, Collection, Sequence
from functools import total_ordering
from pathlib import Path
import logging

from disk import InternedString, DiskIntArray
from corpus import Corpus
from indexset import IndexSet
from util import progress_bar, binsearch_range


################################################################################
## Literals, templates and instances

class Literal(NamedTuple):
    negative: bool
    offset: int
    feature: bytes
    value: InternedString

    def __str__(self) -> str:
        return f"{self.feature.decode()}:{self.offset}{'#' if self.negative else '='}{self.value}"

    def test(self, corpus: Corpus, pos: int) -> bool:
        value = corpus.tokens[self.feature][pos + self.offset]
        return (value == self.value) != self.negative

    @staticmethod
    def parse(corpus: Corpus, litstr: str) -> 'Literal':
        try:
            featstr, rest = litstr.split(':')
            assert featstr.replace('_','').isalnum()
            feature = featstr.lower().encode()
            try:
                offset, value = rest.split('=')
                return Literal(False, int(offset), feature.lower(), corpus.intern(feature, value.encode()))
            except ValueError:
                offset, value = rest.split('#')
                return Literal(True, int(offset), feature.lower(), corpus.intern(feature, value.encode()))
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed literal: {litstr}")


class TemplateLiteral(NamedTuple):
    offset: int
    feature: bytes

    def __str__(self) -> str:
        return f"{self.feature.decode()}:{self.offset}"

    @staticmethod
    def parse(litstr: str) -> 'TemplateLiteral':
        try:
            featstr, offset = litstr.split(':')
            assert featstr.replace('_','').isalnum()
            feature = featstr.lower().encode()
            return TemplateLiteral(int(offset), feature)
        except (ValueError, AssertionError):
            raise ValueError(f"Ill-formed template literal: {litstr}")


@total_ordering
class Template:
    template: tuple[TemplateLiteral,...]
    literals: tuple[Literal,...]

    def __init__(self, template: Sequence[TemplateLiteral], literals: Collection[Literal] = []) -> None:
        self.template = tuple(template)
        self.literals = tuple(sorted(set(literals)))
        try:
            assert self.template == tuple(sorted(set(self.template))), f"Unsorted template"
            assert self.literals == tuple(sorted(literals)),           f"Duplicate literal(s)"
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
        return len(self.template)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Template) and \
            (self.template, self.literals) == (other.template, other.literals)

    def __lt__(self, other: 'Template') -> bool:
        return (len(self), self.template, self.literals) < (len(other), other.template, other.literals)

    def __hash__(self) -> int:
        return hash((self.template, self.literals))

    def instantiate(self, corpus: Corpus, pos: int) -> Optional['Instance']:
        if not all(lit.test(corpus, pos) for lit in self.literals):
            return None
        return Instance([corpus.tokens[tmpl.feature][pos + tmpl.offset] for tmpl in self.template])

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


class Instance(tuple[InternedString,...]):
    def __str__(self) -> str:
        return '+'.join(map(str, self))


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

    def search_key(self) -> Callable[[int], int]:
        tmpl = self.template.template[0]
        features = self.corpus.tokens[tmpl.feature].raw()
        offset = tmpl.offset
        index = self.index.array
        return lambda k: features[index[k] + offset]

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 1, f"UnaryIndex instance must have length 1: {instance}"
        return binsearch_range(0, len(self)-1, instance[0].index, self.search_key())


class BinaryIndex(Index):
    def __init__(self, corpus: Corpus, template: Template) -> None:
        assert len(template) == 2, f"BinaryIndex templates must have length 2: {template}"
        super().__init__(corpus, template)

    def lookup_instance(self, instance: Instance) -> tuple[int, int]:
        assert len(instance) == 2, f"BinaryIndex instance must have length 2: {instance}"
        tmpl1, tmpl2 = self.template.template
        offset1, offset2 = tmpl1.offset, tmpl2.offset
        features1 = self.corpus.tokens[tmpl1.feature].raw()
        features2 = self.corpus.tokens[tmpl2.feature].raw()
        index = self.index.array
        def search_key(k: int) -> tuple[int, int]:
            return (features1[index[k] + offset1], features2[index[k] + offset2])
        val1, val2 = instance
        return binsearch_range(0, len(self)-1, (val1.index, val2.index), search_key)

