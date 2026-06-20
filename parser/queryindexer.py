import time
from typing import Iterator, Union
from parser import QueryParser
from queryvariants import QueryVariants
from query import Query
from concatenation import ConcatenationQuery
from disjunction import DisjunctionQuery
from conjunction import ConjunctionQuery
from negation import NegationQuery
from range import Range, QueryRange
from variable import ConjunctionVariable, DisjunctionVariable, OffsetVariable, UnknownVariable, Variable
from atomic import AtomicQuery, Feature, FValue, FeatureOperator
from spacer import SpacerQuery
from wildcard import WildcardQuery
import os
import sys

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
        return QueryRange(ConjunctionQuery(queries), Range(self.start, self.end))
    
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
            
        # Remove duplicates possible_ends
        possible_ends = list(set(possible_ends))
        
        return QueryRange(ConjunctionQuery(queries), Range(self.start, ConjunctionVariable(possible_ends) if len(possible_ends) > 1 else possible_ends[0]))
        
    
    def __match__negation__(self, query: NegationQuery) -> QueryRange:
        inner_query = query.query
        inner_range = QueryIndexer.query_range(inner_query, self.start)
        return QueryRange(NegationQuery(inner_range), Range(inner_range.start, inner_range.end))
    
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
