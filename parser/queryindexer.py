from typing import Union
from query import Query
from concatenation import ConcatenationQuery
from disjunction import DisjunctionQuery
from conjunction import ConjunctionQuery
from negation import NegationQuery
from range import Range, QueryRange
from variable import DisjunctionVariable, OffsetVariable, UnknownVariable, Variable
from atomic import AtomicQuery, Feature, FValue, FeatureOperator
from spacer import SpacerQuery
from wildcard import WildcardQuery

class QueryIndexer:
    """
    A utility class that can calculate indices for queries.
    """
    
    def __init__(self, query: Query, start: Variable):
        self.query = query
        self.start = start
        self.end = None
    
    def __match__concatenation__(self, query: ConcatenationQuery) -> QueryRange:
        start = self.start
        queries = []
        for q in query.queries:
            inner_range = QueryIndexer.query_range(q, start)
            start = inner_range.end
            queries.append(inner_range)
        self.end = start
        return QueryRange(ConcatenationQuery(queries), Range(self.start, self.end))
    
    def __match__disjunction__(self, query: DisjunctionQuery) -> QueryRange:
        start = self.start
        possible_ends = []
        queries = []
        for q in query.queries:
            inner_range = QueryIndexer.query_range(q, start)
            queries.append(inner_range)
            possible_ends.append(inner_range.end)
        
        return QueryRange(DisjunctionQuery(queries), Range(self.start, DisjunctionVariable(possible_ends)))
    
    def __match__conjunction__(self, query: ConjunctionQuery) -> QueryRange:
        # [word=a] ; (([word=b] ; [word=c]) & ([word=d] ; [word=e] ; [word=f])) ; [word=g]
        start = self.start
        possible_ends = []
        queries = []
        for q in query.queries:
            inner_range = QueryIndexer.query_range(q, start)
            queries.append(inner_range)
            possible_ends.append(inner_range.end)
        
        return QueryRange(ConjunctionQuery(queries), Range(self.start, DisjunctionVariable(possible_ends)))
        
    
    def __match__negation__(self, query: NegationQuery) -> QueryRange:
        inner_query = query.query
        inner_range = QueryIndexer.query_range(inner_query, self.start)
        return QueryRange(NegationQuery(inner_query), Range(inner_range.start, inner_range.end))
    
    def __match__atomic__(self, query: AtomicQuery) -> QueryRange:
        return QueryRange(query, Range(self.start, self.start + 1))
    
    def __match__spacer__(self, query: SpacerQuery) -> QueryRange:
        return QueryRange(query, Range(self.start, self.start + query.length))
    
    def __match__wildcard__(self, query: WildcardQuery) -> QueryRange:
        inner_query = query.query
        inner_range = QueryIndexer.query_range(inner_query, self.start)
        return QueryRange(WildcardQuery(inner_range), Range(self.start, self.start + UnknownVariable('?')))
    
    def __match__query__(self, query: Query) -> QueryRange:
        t: type = type(query)
        if t == ConcatenationQuery:
            return self.__match__concatenation__(query)
        elif t == DisjunctionQuery:
            return self.__match__disjunction__(query)
        elif t == ConjunctionQuery:
            return self.__match__conjunction__(query)
        elif t == NegationQuery:
            return self.__match__negation__(query)
        elif t == AtomicQuery:
            return self.__match__atomic__(query)
        elif t == SpacerQuery:
            return self.__match__spacer__(query)
        elif t == WildcardQuery:
            return self.__match__wildcard__(query)
        else:
            raise TypeError(f"Unknown query type: {t}")
    
    @staticmethod
    def query_range(query: Query, start: Variable) -> QueryRange:
        """
        Calculate the range of the query.
        
        Returns:
            QueryRange: The range of the query.
        """
        indexer = QueryIndexer(query, start)
        return indexer.__match__query__(query)

def str_sequence(start='a'):
    from itertools import product, count
    import string

    alphabet = string.ascii_lowercase
    start_len = len(start)

    def index_from_str(s):
        value = 0
        for c in s:
            value = value * 26 + (ord(c) - ord('a'))
        return value

    skip = index_from_str(start)

    for length in count(start_len):
        for combo in product(alphabet, repeat=length):
            if skip > 0:
                skip -= 1
                continue
            yield ''.join(combo)

class Multimap:
    """
    A class representing a multimap.
    """
    def __init__(self):
        self.map = {}
    
    def add(self, value):
        if value not in self.map:
            self.map[value] = []
        self.map[value].append(value)
    
    def __repr__(self):
        return str(self.map)
    
    def __str__(self):
        return str(self.map)
    
    def __iter__(self):
        return iter(self.map.values())

if __name__ == "__main__":
    test_query = ConcatenationQuery([
        WildcardQuery(AtomicQuery("word", "a", FeatureOperator.EQUALS)),
        DisjunctionQuery([
            AtomicQuery("word", "x", FeatureOperator.EQUALS),
            SpacerQuery(0),
            ConcatenationQuery([
                AtomicQuery("word", "y", FeatureOperator.EQUALS),
                AtomicQuery("word", "z", FeatureOperator.EQUALS)
            ])
        ]),
        DisjunctionQuery([AtomicQuery("word", "b", FeatureOperator.EQUALS), ConcatenationQuery([
            AtomicQuery("word", "c", FeatureOperator.EQUALS),
            AtomicQuery("word", "d", FeatureOperator.EQUALS)
        ])])
    ])
    
    print("Query:", test_query)
    
    start = OffsetVariable(0)
    query_range = QueryIndexer.query_range(test_query, start)
    print("Range:", query_range)
    
    def collect_queries(query: Union[Query, QueryRange]) -> list[QueryRange]:
        queries = []
        if isinstance(query, QueryRange):
            for q in query.components():
                queries.extend(collect_queries(q))
        queries.append(query)
        return queries
    
    variableMap = Multimap()
    for q in collect_queries(query_range):
        q.collect_variables(variableMap)
    
    generator = str_sequence('a')
    
    variableSet = {}
    
    for variables in variableMap:
        name = next(generator)
        for var in variables:
            if var.name is None:
                var.name = name
            variableSet[var.name] = var
            
    print("Variables:", variableSet)
    
    print("Range:", query_range)
