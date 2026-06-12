
from typing import NewType
from collections.abc import Iterable, Collection, Sequence
from dataclasses import dataclass, field

from disk import Symbols, Symbol, SymbolRange
from corpus import Corpus
from util import check_feature, Feature, FValue


################################################################################
## Literals, templates and instances

Instance = NewType('Instance', Sequence[Symbols])


@dataclass(frozen=True, order=True)
class KnownLiteral:
    negative: bool
    offset: int
    feature: Feature
    value: Symbols
    corpus: Corpus = field(compare=False)

    def __post_init__(self) -> None:
        check_feature(self.feature)

    def __str__(self) -> str:
        prefix = f"{self.feature.decode()}:{self.offset}{'#' if self.negative else '='}"
        def lookup(sym: Symbol) -> str:
            return self.corpus.lookup_symbol(self.feature, sym).decode()
        if isinstance(self.value, Symbol):
            return prefix + lookup(self.value)
        elif isinstance(self.value, SymbolRange):
            return prefix + "-".join(map(lookup, self.value))
        else:
            syms = [lookup(s) for s in self.value.symbols]
            if len(syms) >= 5:
                syms[2:len(syms)-1] = ["..."]
            return f"{prefix}{{{','.join(syms)}}}#{len(self.value.symbols)}"

    def is_prefix(self) -> bool:
        return isinstance(self.value, SymbolRange)

    # unoptimized version for use with Query.check_position
    def check_position(self, corpus: Corpus, pos: int) -> bool:
        value = corpus.tokens[self.feature][pos + self.offset]
        if isinstance(self.value, Symbol):
            return self.negative != (self.value == value)
        elif isinstance(self.value, SymbolRange):
            return self.negative != (self.value[0] <= value <= self.value[1])
        else: # isinstance(self.value, SymbolList)
            return self.negative != (value in self.value.symbols)

    def test(self, pos: int) -> bool:
        return self.check_position(self.corpus, pos)

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
        object.__setattr__(self, 'literals', tuple(literals))
        object.__setattr__(self, 'size', len(self.template))
        try:
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
                if isinstance(lit.value, int)
                for val in [lit.corpus.lookup_symbol(lit.feature, lit.value).decode()]
            )
            if literal:
                tokens.append(token + '|' + literal)
            else:
                tokens.append(token)
        return ''.join('[' + tok + ']' for tok in tokens)

    def __iter__(self) -> Iterable[TemplateLiteral]:
        return iter(self.template)

    def __len__(self) -> int:
        return self.size

    def instantiate(self, corpus: Corpus, pos: int) -> 'Instance | None':
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

