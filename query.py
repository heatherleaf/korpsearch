
import re
import itertools
from typing import List, Dict, Tuple, Set, Iterator, Sequence, Union

from index import Literal, TemplateLiteral, Template, Instance, Index, DisjunctiveGroup
from corpus import Corpus
from disk import InternedString


################################################################################
## Queries

QueryElement = Union[Literal, DisjunctiveGroup]

class Query:
    query_regex = re.compile(r'^ (\[ (\|? [\w_]+   !?  = " [^"]+ " )* \])+ $', re.X)
    token_regex = re.compile(r'  (\|?)(   [\w_]+) (!?) = "([^"]+)"          ', re.X)

    corpus : Corpus
    literals : List[QueryElement]
    features : Set[str]
    # featured_query : Dict[str, List[Tuple[bool, int, InternedString]]]
    template : Template

    def __init__(self, corpus:Corpus, literals:Sequence[QueryElement]):
        self.corpus = corpus
        self.literals = sorted(set(literals), key=compare)
        self.features = {lit.feature for lit in self.literals if isinstance(lit, Literal)}
        for lit in literals:
            if isinstance(lit, DisjunctiveGroup):
                self.features.update(lit.features)

        # We cannot handle non-atomic querues with only negative literals
        # -A & -B == -(A v B), since we cannot handle union (yet)
        if len(self) > 1 and self.is_negative():
            raise ValueError(f"Cannot handle non-atomic queries with no positive literals: {self}")

        # This is a variant of self.query, optimised for checking a query at a corpus position:
        # self.featured_query = {f: [] for f in self.features}
        # for lit in self.literals:
        #     self.featured_query[lit.feature].append(
        #         (lit.negative, lit.offset, lit.first_value, lit.last_value)
        #     )

        # We precompute the associated query template. It raises a ValueError if it's not valid.
        if self.contains_disjunction():
            self.template = None
        else:
            if self.is_negative():
                self.template = Template(
                    [TemplateLiteral(lit.offset-self.offset(), lit.feature) for lit in self.negative_literals()],
                )
            else:
                self.template = Template(
                    [TemplateLiteral(lit.offset-self.offset(), lit.feature) for lit in self.positive_literals()],
                    [Literal(True, lit.offset-self.offset(), lit.feature, lit.first_value, lit.last_value) for lit in self.negative_literals()],
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

    def positive_literals(self) -> List[QueryElement]:
        return [lit for lit in self.literals if not lit.negative]

    def negative_literals(self) -> List[QueryElement]:
        return [lit for lit in self.literals if lit.negative]

    def instance(self) -> Instance:
        if self.is_negative():
            return Instance([(lit.first_value, lit.last_value) for lit in self.negative_literals()])
        else:
            return Instance([(lit.first_value, lit.last_value) for lit in self.positive_literals()])

    def index(self) -> Index:
        return Index(self.corpus, self.template)

    def contains_disjunction(self) -> bool:
        return any(isinstance(lit, DisjunctiveGroup) for lit in self.literals)

    def expand(self) -> Iterator[Literal]:
        groups = [group.literals for group in self.literals if isinstance(group, DisjunctiveGroup)]
        singles = [lit for lit in self.literals if isinstance(lit, Literal)]
        for group in itertools.product(*groups):
            yield singles + list(group)

    def subqueries(self) -> Iterator['Query']:
        # Subqueries are generated in decreasing order of complexity
        for n in reversed(range(len(self))):
            for literals in itertools.combinations(self.literals, n+1):
                if (any([literal.is_prefix() for literal in literals]) and len(literals) > 1):
                    continue
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

    # Right now this function is not optimized and can not give false for
    # cases such as [a&b|c&d]
    def check_position(self, pos:int) -> bool:
        # return all(
        #     (self.corpus.tokens[lit.feature][pos + lit.offset] == lit.value) != lit.negative
        #     for lit in self.literals
        # )
        # This is an optimised (but less readable) version of the code above:
        # for feature, values in self.featured_query.items():
        #     lookup = self.corpus.tokens[feature]
        #     if any((lookup[pos+offset] >= first_value and lookup[pos+offset] <= last_value) == negative for negative, offset, first_value, last_value in values):
        #         return False
        # return True
        return all(lit.check_position(self.corpus.tokens, pos) for lit in self.literals)

    @staticmethod
    def parse(corpus:Corpus, querystr:str, no_sentence_breaks:bool=False) -> 'Query':
        querystr = querystr.replace(' ', '')
        if not Query.query_regex.match(querystr):
            raise ValueError(f"Error in query: {querystr!r}")
        tokens = querystr.split('][')
        query : List[QueryElement] = []
        for offset, token in enumerate(tokens):
            query_list : List[List[Literal]] = [[]]
            for match in Query.token_regex.finditer(token):
                or_separator, feature, negated, value = match.groups()
                feature = feature.lower()
                negative = (negated == '!')
                is_prefix = False
                if value.endswith('*'):
                    is_prefix = True
                    value = value.split('*')[0]
                elif value.startswith('*'):
                    is_prefix = True
                    value = value.split('*')[-1][::-1]
                    feature = feature + "_rev"
                first_value, last_value = corpus.intern(feature, value.encode(), is_prefix)
                if or_separator == "|":
                    query_list.append([])
                query_list[-1].append(Literal(negative, offset, feature, first_value, last_value))
            if len(query_list) > 1:
                query.extend([DisjunctiveGroup.create(literals) for literals in itertools.product(*query_list)])
            else:
                query.extend(*query_list)
        if not no_sentence_breaks:
            sfeature = corpus.sentence_feature
            svalue, _ = corpus.intern(sfeature, corpus.sentence_start_value)
            for offset in range(1, len(tokens)):
                query.append(Literal(True, offset, sfeature, svalue, svalue))
        return Query(corpus, query)


def compare(query_element):
    if isinstance(query_element, Literal): return list(query_element)
    elif isinstance(query_element, DisjunctiveGroup):
        return list(itertools.chain(query_element))
    
