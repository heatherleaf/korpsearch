
import re
import itertools
from typing import Literal
from collections.abc import Iterator, Sequence

from index import KnownLiteral, DisjunctiveGroup, TemplateLiteral, Template, Instance, mkInstance, Index
from corpus import Corpus
from util import Feature, FValue, SENTENCE, START


################################################################################
## Queries

QueryElement = KnownLiteral | DisjunctiveGroup

def get_query_literals(query_element: QueryElement) -> tuple[KnownLiteral, ...]:
    if isinstance(query_element, KnownLiteral): 
        return (query_element,)
    else: # isinstance(query_element, DisjunctiveGroup):
        return query_element.literals


class Query:
    query_regex = re.compile(r'^ (\[ (\|? [\w_]+   !?  = " [^"]+ " )* \])+ $', re.X)
    token_regex = re.compile(r'  (\|?)(   [\w_]+) (!?) = "([^"]+)"          ', re.X)

    corpus: Corpus
    literals: list[QueryElement]
    features: set[Feature]
    # featured_query: dict[Feature, list[tuple[bool, int, InternedString, InternedString]]]
    template: Template | None

    def __init__(self, corpus: Corpus, literals: Sequence[QueryElement]) -> None:
        self.corpus = corpus
        self.literals = sorted(set(literals), key=get_query_literals)
        self.features = {
            lit.feature for query_element in self.literals
            for lit in get_query_literals(query_element)
        }

        # We cannot handle non-atomic querues with only negative literals
        # -A & -B == -(A v B), since we cannot handle union (yet)
        if len(self) > 1 and self.is_negative():
            raise ValueError(f"Cannot handle non-atomic queries with no positive literals: {self}")

        # # This is a variant of self.query, optimised for checking a query at a corpus position:
        # self.featured_query = {f: [] for f in self.features}
        # for lit in self.literals:
        #     self.featured_query[lit.feature].append(
        #         (lit.negative, lit.offset, lit.value, lit.value2)
        #     )

        # We precompute the associated query template. It raises a ValueError if it's not valid.
        if self.contains_disjunction():
            self.template = None
        elif self.is_negative():
            self.template = Template(
                [TemplateLiteral(lit.offset-self.min_offset(), lit.feature) for lit in self.negative_literals()],
            )
        else:
            self.template = Template(
                [TemplateLiteral(lit.offset-self.min_offset(), lit.feature) for lit in self.positive_literals()],
                [KnownLiteral(True, lit.offset-self.min_offset(), lit.feature, lit.value, lit.value2, corpus) for lit in self.negative_literals()],
            )


    def __str__(self) -> str:
        return '[' + ']&['.join(map(str, self.literals)) + ']'

    def __repr__(self) -> str:
        return f"Query({self.literals})"

    def __len__(self) -> int:
        return len(self.literals)

    def min_offset(self) -> int:
        return min(lit.offset for lit in self.literals)

    def max_offset(self) -> int:
        return max(lit.offset for lit in self.literals)

    def is_negative(self) -> bool:
        return not self.positive_literals()

    def positive_literals(self) -> list[KnownLiteral]:
        return [lit for lit in self.literals if isinstance(lit, KnownLiteral) if not lit.negative]

    def negative_literals(self) -> list[KnownLiteral]:
        return [lit for lit in self.literals if isinstance(lit, KnownLiteral) if lit.negative]

    def instance(self) -> Instance:
        if self.is_negative():
            return mkInstance([(lit.value, lit.value2) for lit in self.negative_literals()])
        else:
            return mkInstance([(lit.value, lit.value2) for lit in self.positive_literals()])

    def index(self) -> Index:
        assert self.template
        return Index.get(self.corpus, self.template)

    def contains_disjunction(self) -> bool:
        return any(isinstance(lit, DisjunctiveGroup) for lit in self.literals)

    def expand(self) -> Iterator[list[KnownLiteral]]:
        groups = [group.literals for group in self.literals if isinstance(group, DisjunctiveGroup)]
        singles = [lit for lit in self.literals if isinstance(lit, KnownLiteral)]
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

    # Right now this function is not optimized and can not give false for cases such as [a&b|c&d]
    def check_position(self, pos: int) -> bool:
        return all(
            lit.check_position(self.corpus, pos) for lit in self.literals
        )
        # return all(
        #     (self.corpus.tokens[lit.feature][pos + lit.offset] == lit.value) != lit.negative
        #     for lit in self.literals
        # )
        # This is an optimised (but less readable) version of the code above:
        # for feature, values in self.featured_query.items():
        #     lookup = self.corpus.tokens[feature]
        #     if any((lookup[pos+offset] >= value and lookup[pos+offset] <= value2) == negative for (negative, offset, value, value2) in values):
        #         return False
        # return True

    @staticmethod
    def _classify_value(value: str) -> Literal['normal'] | Literal['prefix'] | Literal['suffix'] | Literal['regex']:
        if value.isalnum():
            return 'normal'
        elif value.endswith('.*') and value[:-2].isalnum():
            return 'prefix'
        elif value.startswith('.*') and value[2:].isalnum():
            return 'suffix'
        else:
            return 'regex'

    @staticmethod
    def parse(corpus: Corpus, querystr: str, no_sentence_breaks: bool = False) -> 'Query':
        querystr = querystr.replace(' ', '')
        if not Query.query_regex.match(querystr):
            raise ValueError(f"Error in query: {querystr!r}")
        tokens = querystr.split('][')
        query: list[QueryElement] = []
        for offset, token in enumerate(tokens):
            query_list: list[list[KnownLiteral]] = [[]]
            for match in Query.token_regex.finditer(token):
                or_separator, featstr, negated, valstr = match.groups()
                if or_separator == '|':
                    query_list.append([])
                feature = Feature(featstr.lower().encode())
                negative = (negated == '!')
                value_type = Query._classify_value(valstr)
                match value_type:
                    case 'normal':
                        value = FValue(valstr.encode())
                        interned = corpus.intern(feature, value)
                        query_list[-1].append(KnownLiteral(negative, offset, feature, interned, interned, corpus))
                    case 'prefix':
                        valstr = valstr.split('.*')[0]
                        value = FValue(valstr.encode())
                        interned = corpus.interned_range(feature, value)
                        query_list[-1].append(KnownLiteral(negative, offset, feature, interned[0], interned[1], corpus))
                    case 'suffix':
                        valstr = valstr.split('.*')[-1][::-1]
                        value = FValue(valstr.encode())
                        feature = Feature(feature + b'_rev')
                        interned = corpus.interned_range(feature, value)
                        query_list[-1].append(KnownLiteral(negative, offset, feature, interned[0], interned[1], corpus))
                    case 'regex':
                        regex_matches = corpus.get_matches(feature, valstr)
                        regexed_literals = [KnownLiteral(negative, offset, feature, match, match, corpus) for match in regex_matches]
                        last_group = query_list.pop()
                        query_list.extend(last_group + [lit] for lit in regexed_literals)
            if len(query_list) > 1:
                query.extend(DisjunctiveGroup.create(literals) for literals in itertools.product(*query_list))
            else:
                query.extend(*query_list)
        if not no_sentence_breaks:
            svalue = corpus.intern(SENTENCE, START)
            for offset in range(1, len(tokens)):
                query.append(KnownLiteral(True, offset, SENTENCE, svalue, svalue, corpus))
        return Query(corpus, query)
