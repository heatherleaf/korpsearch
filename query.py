
import re
from typing import List, Tuple, Set, Iterator

from index import Template, Instance
from corpus import Corpus
from disk import InternedString

################################################################################
## Queries

QUEREGEX = re.compile(r'^(\[ ([a-z]+ !? = "[^"]+")* \])+$', re.X)

class Query:
    corpus : Corpus
    query : List[List[Tuple[str, InternedString, bool]]]
    features : Set[str]
    _featured_query : List[Tuple[str, List[Tuple[int, InternedString]]]]

    def __init__(self, corpus:Corpus, querystr:str):
        self.corpus = corpus
        querystr = querystr.replace(' ', '')
        if not QUEREGEX.match(querystr):
            raise ValueError(f"Error in query: {querystr!r}")
        tokens = querystr.split('][')
        self.query = []
        for tok in tokens:
            self.query.append([])
            parts = re.findall(r'\w+ !? = "[^"]+"', tok, re.X)
            for part in parts:
                feat, value = part.split('=', 1)
                positive = not feat.endswith('!')
                feat = feat.rstrip('!')
                value = value.replace('"', '')
                self.query[-1].append((feat, self.corpus.intern(feat, value.encode()), positive))

        self.features = {feat for tok in self.query for feat, _, _ in tok}

        # This is a variant of self.query, optimised for checking a query at a corpus position:
        self.featured_query = {f: [] for f in self.features}
        for i, tok in enumerate(self.query):
            for feat, val, positive in tok:
                self.featured_query[feat].append((i, val, positive))

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

