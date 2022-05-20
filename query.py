
import re
from index import Template, Instance
from util import bytesify

################################################################################
## Queries

QUEREGEX = re.compile(rb'^(\[ ([a-z]+ = "[^"]+")* \])+$', re.X)

class Query:
    def __init__(self, corpus, querystr):
        self._corpus = corpus
        querystr = bytesify(querystr)
        querystr = querystr.replace(b' ', b'')
        if not QUEREGEX.match(querystr):
            raise ValueError(f"Error in query: {querystr}")
        tokens = querystr.split(b'][')
        self.query = []
        for tok in tokens:
            self.query.append([])
            parts = re.findall(rb'\w+="[^"]+"', tok)
            for part in parts:
                feat, value = part.split(b'=', 1)
                value = value.replace(b'"', b'')
                self.query[-1].append((feat, self._corpus.intern(feat, value)))

    def __str__(self):
        return " ".join("[" + " ".join(f'{feat.decode()}="{bytes(val).decode()}"' for feat, val in subq) + "]"
                        for subq in self.query)

    def subqueries(self):
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

    def features(self):
        return {feat for tok in self.query for feat, _val in tok}

    def check_sentence(self, sentence):
        for k in range(len(sentence) - len(self.query) + 1):
            if all(sentence[k+i][feat] == value 
                   for i, token_query in enumerate(self.query)
                   for feat, value in token_query
                   ):
                return True
        return False

    @staticmethod
    def is_subquery(subtemplate, subinstance, template, instance):
        positions = sorted({pos for _, pos in template})
        query = {(feat, pos, val) for ((feat, pos), val) in zip(template, instance)}
        for base in positions:
            subquery = {(feat, base+pos, val) for ((feat, pos), val) in zip(subtemplate, subinstance)}
            if subquery.issubset(query):
                return True
        return False

