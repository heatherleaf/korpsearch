
import re
from typing import List, Tuple, Set, Iterator

from index import Template, Instance
from corpus import Corpus
from disk import InternedString

################################################################################
## Queries

QUEREGEX = re.compile(r'^(\[ ([a-z]+ = "[^"]+")* \])+$', re.X)

class Query:
    corpus : Corpus
    query : List[List[Tuple[str, InternedString]]]
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
            parts = re.findall(r'\w+="[^"]+"', tok)
            for part in parts:
                feat, value = part.split('=', 1)
                value = value.replace('"', '')
                self.query[-1].append((feat, self.corpus.intern(feat, value.encode())))

        self.features = {feat for tok in self.query for feat, _val in tok}

        # This is a variant of self.query, optimised for checking a query at a corpus position:
        featured_query = {f: [] for f in self.features}
        for i, tok in enumerate(self.query):
            for feat, val in tok:
                featured_query[feat].append((i, val))
        self._featured_query = list(featured_query.items())

    def __str__(self) -> str:
        return " ".join("[" + " ".join(f'{feat}="{bytes(val).decode()}"' for feat, val in subq) + "]"
                        for subq in self.query)

    def subqueries(self) -> Iterator[Tuple[Template, Instance, int]]:
        # Pairs of tokens
        for offset, tok in enumerate(self.query):
            for feat, value in tok:
                for dist in range(1, len(self.query)-offset):
                    for feat1, value1 in self.query[offset+dist]:
                        templ = Template((feat, 0), (feat1, dist))
                        yield (templ, Instance(value, value1), offset)
        # Single tokens: yield subqueries after more complex queries!
        for offset, tok in enumerate(self.query):
            for feat, value in tok:
                yield (Template((feat, 0)), Instance(value), offset)

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
        for feat, values in self._featured_query:
            fsent = self.corpus.words[feat]
            if not all(fsent[pos+i] == val for i, val in values):
                return False
        else:
            return True

    @staticmethod
    def is_subquery(subtemplate:Template, subinstance:Instance, template:Template, instance:Instance):
        positions : List[int] = sorted({pos for _, pos in template})
        QuerySet = Set[Tuple[str, int, InternedString]]
        query : QuerySet = {(feat, pos, val) for ((feat, pos), val) in zip(template, instance)}
        for base in positions:
            subquery : QuerySet = {(feat, base+pos, val) for ((feat, pos), val) in zip(subtemplate, subinstance)}
            if subquery.issubset(query):
                return True
        return False

