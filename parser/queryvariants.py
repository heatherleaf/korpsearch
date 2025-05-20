from typing import Iterator, Union
from parser import QueryParser
from query import Query
from concatenation import ConcatenationQuery
from disjunction import DisjunctionQuery
from conjunction import ConjunctionQuery
from negation import NegationQuery
from range import Range, QueryRange
from variable import DisjunctionVariable, OffsetVariable, UnknownVariable, Variable
from atomic import AtomicQuery
from spacer import SpacerQuery
from wildcard import WildcardQuery
import os
import sys

class QueryVariants:
    """
    A utility class that can calculate indices for queries.
    """
    
    def __init__(self, query: Query):
        self.query = query
    
    def __match__concatenation__(self, query: ConcatenationQuery) -> Iterator[Query]:
        queries = query.queries
        
        # Base case
        for i in range(len(queries)):
            for v in QueryVariants.variants(queries[i]):
                new_queries = queries[:i] + [v] + queries[i+1:]
                yield ConcatenationQuery(new_queries)
                
        # (A | B) ; C -> (A ; C) | (B ; C)
        if isinstance(queries[0], DisjunctionQuery):
            for a in queries[0].queries:
                for c in queries[1:]:
                    new_queries = [a] + queries[1:]
                    yield DisjunctionQuery(ConcatenationQuery(new_queries))
        
        # A ; (B | C) -> (A ; B) | (A ; C)
        if isinstance(queries[-1], DisjunctionQuery):
            for a in queries[:-1]:
                for b in queries[-1].queries:
                    new_queries = queries[:-1] + [b]
                    yield DisjunctionQuery(ConcatenationQuery(new_queries))
    
    def __match__disjunction__(self, query: DisjunctionQuery) -> Iterator[Query]:
        queries = query.queries
        
        # Base case
        for i in range(len(queries)):
            for v in QueryVariants.variants(queries[i]):
                new_queries = queries[:i] + [v] + queries[i+1:]
                yield DisjunctionQuery(new_queries)
    
    def __match__conjunction__(self, query: ConjunctionQuery) -> Iterator[Query]:
        queries = query.queries
        
        # Base case
        for i in range(len(queries)):
            for v in QueryVariants.variants(queries[i]):
                new_queries = queries[:i] + [v] + queries[i+1:]
                yield ConjunctionQuery(new_queries)
                
        # (A | B) & C -> (A & C) | (B & C)
        if isinstance(queries[0], DisjunctionQuery):
            for a in queries[0].queries:
                for c in queries[1:]:
                    new_queries = [a] + queries[1:]
                    yield DisjunctionQuery(ConjunctionQuery(new_queries))
        
        # A & (B | C) -> (A & B) | (A & C)
        if isinstance(queries[-1], DisjunctionQuery):
            for a in queries[:-1]:
                for b in queries[-1].queries:
                    new_queries = queries[:-1] + [b]
                    yield DisjunctionQuery(ConjunctionQuery(new_queries))
        
    
    def __match__negation__(self, query: NegationQuery) -> Iterator[Query]:
        # Base case
        for v in QueryVariants.variants(query.query):
            yield NegationQuery(v)

        # Double negation
        if isinstance(query.query, NegationQuery):
            for v in QueryVariants.variants(query.query):
                yield v
    
    def __match__atomic__(self, query: AtomicQuery) -> Iterator[Query]:
        yield query
    
    def __match__spacer__(self, query: SpacerQuery) -> Iterator[Query]:
        yield query
    
    def __match__wildcard__(self, query: WildcardQuery) -> Iterator[Query]:
        yield query
    
    def __match__query__(self, query: Query) -> Iterator[Query]:
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
    def variants(query: Query) -> Iterator[Query]:
        """
        Calculate the range of the query.
        
        Returns:
            QueryRange: The range of the query.
        """
        indexer = QueryVariants(query)
        return indexer.__match__query__(query)
