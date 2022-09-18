
import re
import itertools
from typing import List, Dict, Tuple, Set, Iterator, Sequence

from index import Literal, TemplateLiteral, Template, Instance, Index
from corpus import Corpus
from disk import InternedString


################################################################################
## Queries

class Query:
    query_regex = re.compile(r'^ (\[ ( [\w_]+   !?  = " [^"]+ " )* \])+ $', re.X)
    token_regex = re.compile(r'       ([\w_]+) (!?) = "([^"]+)"          ', re.X)

    corpus : Corpus
    literals : List[Literal]
    features : Set[str]
    featured_query : Dict[str, List[Tuple[bool, int, InternedString]]]
    # no_sentence_breaks : bool

    def __init__(self, corpus:Corpus, literals:Sequence[Literal], no_sentence_breaks=True):
        self.corpus = corpus
        self.literals = sorted(set(literals))
        self.features = {lit.feature for lit in self.literals}
        # self.no_sentence_breaks = no_sentence_breaks

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

    def __str__(self) -> str:
        return '[' + ']&['.join(map(str, self.literals)) + ']'

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

    def positive_literals(self) -> List[Literal]:
        return [lit for lit in self.literals if not lit.negative]

    def negative_literals(self) -> List[Literal]:
        return [lit for lit in self.literals if lit.negative]

    def template(self) -> Template:
        if self.is_negative():
            return Template(
                [TemplateLiteral(lit.offset-self.offset(), lit.feature) for lit in self.negative_literals()],
            )
        else:
            return Template(
                [TemplateLiteral(lit.offset-self.offset(), lit.feature) for lit in self.positive_literals()],
                [Literal(True, lit.offset-self.offset(), lit.feature, lit.value) for lit in self.negative_literals()],
            )

    def instance(self) -> Instance:
        if self.is_negative():
            return Instance([lit.value for lit in self.negative_literals()])
        else:
            return Instance([lit.value for lit in self.positive_literals()])

    def index(self) -> Index:
        return Index(self.corpus, self.template())

    def subqueries(self) -> Iterator['Query']:
        # Subqueries are generated in decreasing order of complexity
        for n in reversed(range(len(self))):
            for literals in itertools.combinations(self.literals, n+1):
                try:
                    yield Query(self.corpus, literals)
                except ValueError:
                    pass

    def subsumed_by(self, others:List['Query']) -> bool:
        other_literals = {lit for other_query in others for lit in other_query.literals}
        return set(self.literals).issubset(other_literals)

    def check_sentence(self, sent:int) -> bool:
        positions = self.corpus.sentence_positions(sent)
        min_offset = self.min_offset()
        max_offset = self.max_offset()
        return any(
            self.check_position(pos)
            for pos in range(positions.start - min_offset, positions.stop - max_offset)
        )

    def check_position(self, pos:int) -> bool:
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
    def parse(corpus:Corpus, querystr:str, no_sentence_breaks:bool=False) -> 'Query':
        querystr = querystr.replace(' ', '')
        if not Query.query_regex.match(querystr):
            raise ValueError(f"Error in query: {querystr!r}")
        tokens = querystr.split('][')
        query : List[Literal] = []
        for offset, token in enumerate(tokens):
            for match in Query.token_regex.finditer(token):
                feature, negated, value = match.groups()
                feature = feature.lower()
                negative = (negated == '!')
                value = corpus.intern(feature, value.encode())
                query.append(Literal(negative, offset, feature, value))
        if not no_sentence_breaks:
            sfeature = corpus.sentence_feature
            svalue = corpus.intern(sfeature, corpus.sentence_start_value)
            for offset in range(1, len(tokens)):
                query.append(Literal(True, offset, sfeature, svalue))
        return Query(corpus, query, no_sentence_breaks)
