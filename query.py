
import re
import itertools
from collections.abc import Iterator, Sequence

from index import KnownLiteral, TemplateLiteral, Template, Instance, Index
from corpus import Corpus
from util import Feature, FValue, SENTENCE, START
from disk import InternedString


################################################################################
## Queries

class Query:
    query_regex = re.compile(r'^ (\[ ( [\w_]+   !?  = " [^"]+ " )* \])+ $', re.X)
    token_regex = re.compile(r'       ([\w_]+) (!?) = "([^"]+)"          ', re.X)

    corpus: Corpus
    literals: list[KnownLiteral]
    features: set[Feature]
    featured_query: dict[Feature, list[tuple[bool, int, InternedString]]]
    template: Template

    def __init__(self, corpus: Corpus, literals: Sequence[KnownLiteral]) -> None:
        self.corpus = corpus
        self.literals = sorted(set(literals))
        self.features = {lit.feature for lit in self.literals}

        # We cannot handle non-atomic querues with only negative literals
        # -A & -B == -(A v B), since we cannot handle union (yet)
        if len(self) > 1 and self.is_negative():
            raise ValueError(f"Cannot handle non-atomic queries with no positive literals: {self}")

        # This is a variant of self.query, optimised for checking a query at a corpus position:
        self.featured_query = {f: [] for f in self.features}
        for lit in self.literals:
            self.featured_query[lit.feature].append(
                (lit.negative, lit.offset, lit.value)
            )

        # We precompute the associated query template. It raises a ValueError if it's not valid.
        if self.is_negative():
            self.template = Template(
                [TemplateLiteral(lit.offset-self.offset(), lit.feature) for lit in self.negative_literals()],
            )
        else:
            self.template = Template(
                [TemplateLiteral(lit.offset-self.offset(), lit.feature) for lit in self.positive_literals()],
                [KnownLiteral(True, lit.offset-self.offset(), lit.feature, lit.value, corpus) for lit in self.negative_literals()],
            )


    def __str__(self) -> str:
        return '[' + ']&['.join(map(str, self.literals)) + ']'

    def __repr__(self) -> str:
        return f"Query({self.literals})"

    def __len__(self) -> int:
        return len(self.literals)

    def offset(self) -> int:
        if self.is_negative():
            return min(lit.offset for lit in self.negative_literals())
        else:
            return min(lit.offset for lit in self.positive_literals())

    def min_offset(self) -> int:
        return min(lit.offset for lit in self.literals)

    def max_offset(self) -> int:
        return max(lit.offset for lit in self.literals)

    def is_negative(self) -> bool:
        return not self.positive_literals()

    def positive_literals(self) -> list[KnownLiteral]:
        return [lit for lit in self.literals if not lit.negative]

    def negative_literals(self) -> list[KnownLiteral]:
        return [lit for lit in self.literals if lit.negative]

    def instance(self) -> Instance:
        if self.is_negative():
            return Instance(tuple(lit.value for lit in self.negative_literals()))
        else:
            return Instance(tuple(lit.value for lit in self.positive_literals()))

    def index(self) -> Index:
        return Index.get(self.corpus, self.template)

    def subqueries(self) -> Iterator['Query']:
        # Subqueries are generated in decreasing order of complexity
        for n in reversed(range(len(self))):
            for literals in itertools.combinations(self.literals, n+1):
                try:
                    yield Query(self.corpus, literals)
                except ValueError:
                    pass

    def subsumed_by(self, others: list['Query']) -> bool:
        other_literals = {lit for other_query in others for lit in other_query.literals}
        return set(self.literals).issubset(other_literals)

    def check_sentence(self, sent: int) -> bool:
        positions = self.corpus.sentence_positions(sent)
        min_offset = self.min_offset()
        max_offset = self.max_offset()
        return any(
            self.check_position(pos)
            for pos in range(positions.start - min_offset, positions.stop - max_offset)
        )

    def check_position(self, pos: int) -> bool:
        # return all(
        #     (self.corpus.tokens[lit.feature][pos + lit.offset] == lit.value) != lit.negative
        #     for lit in self.literals
        # )
        # This is an optimised (but less readable) version of the code above:
        for feature, values in self.featured_query.items():
            lookup = self.corpus.tokens[feature]
            if any((lookup[pos+offset] == value) == negative for negative, offset, value in values):
                return False
        return True

    @staticmethod
    def parse(corpus: Corpus, querystr: str, no_sentence_breaks: bool = False) -> 'Query':
        querystr = querystr.replace(' ', '')
        if not Query.query_regex.match(querystr):
            raise ValueError(f"Error in query: {querystr!r}")
        tokens = querystr.split('][')
        query: list[KnownLiteral] = []
        for offset, token in enumerate(tokens):
            for match in Query.token_regex.finditer(token):
                featstr, negated, valstr = match.groups()
                feature = Feature(featstr.lower().encode())
                value = FValue(valstr.encode())
                negative = (negated == '!')
                query.append(KnownLiteral(negative, offset, feature, corpus.intern(feature, value), corpus))
        if not no_sentence_breaks:
            svalue = corpus.intern(SENTENCE, START)
            for offset in range(1, len(tokens)):
                query.append(KnownLiteral(True, offset, SENTENCE, svalue, corpus))
        return Query(corpus, query)
