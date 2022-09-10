
import re
from typing import List, Tuple, Set, Dict, Iterator

from index import Template, Instance
from corpus import Corpus
from disk import InternedString, DiskStringArray

################################################################################
## Queries

QUEREGEX = re.compile(r'^(\[ ([a-z]+ = "[^"]+")* \])+$', re.X)

class Query:
    corpus : Corpus
    query : List[List[Tuple[str, InternedString]]]
    featured_query : List[Tuple[str, List[Tuple[int, InternedString]]]]

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
        features : Set[str] = {feat for tok in self.query for feat, _val in tok}
        featured_query : Dict[str, List[Tuple[int, InternedString]]] = {f: [] for f in features}
        for i, tok in enumerate(self.query):
            for feat, val in tok:
                featured_query[feat].append((i, val))
        self.featured_query = list(featured_query.items())

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

    def features(self) -> Set[str]:
        return {feat for tok in self.query for feat, _val in tok}

    def check_sentence(self, n:int) -> bool:
        sent : slice = self.corpus.lookup_sentence(n)
        words : Dict[str, DiskStringArray] = self.corpus.words
        for k in range(sent.start, sent.stop - len(self.query) + 1):
            for feat, vals in self.featured_query:
                fsent : DiskStringArray = words[feat]
                if not all(fsent[k+i] == v for i, v in vals):
                    break
            else:
                return True
        return False

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

