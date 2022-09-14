
import re
from typing import List, Dict, Tuple, Set, Iterator

from index import Template, Instance
from corpus import Corpus
from disk import InternedString

################################################################################
## Queries


class Query:
    query_regex = re.compile(r'^ (\[ ( [a-z]+   !?  = " [^"]+ " )* \])+ $', re.X)
    token_regex = re.compile(r'       ([a-z]+) (!?) = "([^"]+)"          ', re.X)

    corpus : Corpus
    query : List[List[Tuple[str, InternedString, bool]]]
    features : Set[str]
    featured_query : Dict[str, List[Tuple[int, InternedString, bool]]]

    def __init__(self, corpus:Corpus, querystr:str):
        self.corpus = corpus
        self.query = [[
                (feat, self.corpus.intern(feat, value), positive)
                for feat, value, positive in token
            ]
            for token in Query.parse(querystr)
        ]
        self.features = {feat for token in self.query for feat, _, _ in token}

        # This is a variant of self.query, optimised for checking a query at a corpus position:
        self.featured_query = {f: [] for f in self.features}
        for offset, tok in enumerate(self.query):
            for feat, value, positive in tok:
                self.featured_query[feat].append((offset, value, positive))

    @staticmethod
    def parse(querystr:str) -> List[List[Tuple[str, bytes, bool]]]:
        querystr = querystr.replace(' ', '')
        if not Query.query_regex.match(querystr):
            raise ValueError(f"Error in query: {querystr!r}")
        tokens = querystr.split('][')
        query = []
        for tok in tokens:
            query.append([])
            for match in Query.token_regex.finditer(tok):
                feat, excl, value = match.groups()
                positive = excl != '!'
                query[-1].append((feat, value.encode(), positive))
        return query

    def __str__(self) -> str:
        return ' '.join(
            '[' + ' '.join(
                feat + ('=' if positive else '!=') + str(val)
                for feat, val, positive in subq
            ) + ']'
            for subq in self.query
        )

    def subqueries(self) -> Iterator[Tuple[Template, Instance, int, bool]]:
        # Pairs of tokens, only positive!
        for offset, tok in enumerate(self.query):
            for feat, value, positive in tok:
                for dist in range(1, len(self.query)-offset):
                    for feat1, value1, positive1 in self.query[offset+dist]:
                        if positive and positive1:
                            yield (
                                Template([(feat, 0), (feat1, dist)]),
                                Instance([value, value1]),
                                offset,
                                positive,
                            )
        # Single tokens: yield subqueries after more complex queries!
        for offset, tok in enumerate(self.query):
            for feat, value, positive in tok:
                yield (
                    Template([(feat, 0)]),
                    Instance([value]),
                    offset,
                    positive,
                )

    def check_sentence(self, sent:int) -> bool:
        positions = self.corpus.lookup_sentence(sent)
        return any(
            self.check_position(pos)
            for pos in range(positions.start, positions.stop - len(self.query) + 1)
        )

    def check_position(self, pos:int) -> bool:
        # return all(
        #     self.corpus.words[feat][pos+i] == val
        #     for i, token in enumerate(self.query)
        #     for feat, val in token
        # )
        # This is an optimised (but less readable) version of the code above:
        for feat, values in self.featured_query.items():
            fsent = self.corpus.tokens[feat]
            if not all((fsent[pos+i] == val) == positive for i, val, positive in values):
                return False
        return True

    @staticmethod
    def is_subquery(
            subtemplate:Template, subinstance:Instance, suboffset:int, subpositive:bool,
            template:Template, instance:Instance, offset:int, positive:bool,
        ):
        if not (subpositive and positive):
            return False
        positions : List[int] = sorted({pos for _, pos in template})
        QuerySet = Set[Tuple[str, int, InternedString]]
        query : QuerySet = {(feat, pos+offset, val) for ((feat, pos), val) in zip(template, instance)}
        for base in positions:
            subquery : QuerySet = {(feat, base+pos+suboffset, val) for ((feat, pos), val) in zip(subtemplate, subinstance)}
            if subquery.issubset(query):
                return True
        return False

