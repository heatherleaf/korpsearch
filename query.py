
import re
from index import Template, Instance
from corpus import Corpus
from disk import InternedString, DiskStringArray
from typing import List, Tuple, Set, Dict, Iterator

################################################################################
## Queries

QUEREGEX = re.compile(rb'^(\[ ([a-z]+ = "[^"]+")* \])+$', re.X)

class Query:
    corpus : Corpus
    query : List[List[Tuple[bytes, InternedString]]]
    featured_query : List[Tuple[bytes, List[Tuple[int, InternedString]]]]

    def __init__(self, corpus:Corpus, querystr:bytes):
        self.corpus = corpus
        if isinstance(querystr, str):
            querystr = querystr.encode() 
        querystr = querystr.replace(b' ', b'')
        if not QUEREGEX.match(querystr):
            raise ValueError(f"Error in query: {querystr!r}")
        tokens = querystr.split(b'][')
        self.query = []
        for tok in tokens:
            self.query.append([])
            parts = re.findall(rb'\w+="[^"]+"', tok)
            for part in parts:
                feat, value = part.split(b'=', 1)
                value = value.replace(b'"', b'')
                self.query[-1].append((feat, self.corpus.intern(feat, value)))
        features : Set[bytes] = {feat for tok in self.query for feat, _val in tok}
        featured_query : Dict[bytes, List[Tuple[int, InternedString]]] = {f: [] for f in features}
        for i, tok in enumerate(self.query):
            for feat, val in tok:
                featured_query[feat].append((i, val))
        self.featured_query = list(featured_query.items())

    def __str__(self) -> str:
        return " ".join("[" + " ".join(f'{feat.decode()}="{bytes(val).decode()}"' for feat, val in subq) + "]"
                        for subq in self.query)

    def subqueries(self) -> Iterator[Tuple[Template, Instance]]:
        # Pairs of tokens
        for i, tok in enumerate(self.query):
            for feat, value in tok:
                for dist in range(1, len(self.query)-i):
                    for feat1, value1 in self.query[i+dist]:
                        templ = Template((feat, 0), (feat1, dist))
                        yield (templ, Instance(value, value1))
        # Single tokens: yield subqueries after more complex queries!
        for tok in self.query:
            for feat, value in tok:
                yield (Template((feat, 0)), Instance(value))

    def features(self) -> Set[bytes]:
        return {feat for tok in self.query for feat, _val in tok}

    def check_sentence(self, n:int) -> bool:
        sent : slice = self.corpus.lookup_sentence(n)
        words : Dict[bytes, DiskStringArray] = self.corpus.words
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
        QuerySet = Set[Tuple[bytes, int, InternedString]]
        query : QuerySet = {(feat, pos, val) for ((feat, pos), val) in zip(template, instance)}
        for base in positions:
            subquery : QuerySet = {(feat, base+pos, val) for ((feat, pos), val) in zip(subtemplate, subinstance)}
            if subquery.issubset(query):
                return True
        return False

