from query import Query
from parser import QueryParser
from queryvariants import QueryVariants
from queryindexer import QueryIndexer
from range import QueryRange
from variable import OffsetVariable

from typing import Iterator, Union
import sys
import time

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


def recursive_variants(query) -> Iterator['Query']:
    seen = set()
    stack = [query]

    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        yield current
        for variant in QueryVariants.variants(current):
            if variant not in seen:
                stack.append(variant)

def simple_stringify(query: Query) -> str:
    """
    Simplifies the query string by removing unnecessary characters.
    """
    return repr(query).replace("[]", "*").replace("[", "").replace("b'word' = b'", "").replace("'", "").replace("]", "").replace(";", "&").replace("*", "[]")

if __name__ == "__main__":
    input_query = '[word="A"] ; ([word="B"] | [word="C"]) ; ([word="D"] | [word="E"] | [word="F"])' #'([word="A"] [word="O"])* (![word="X"] | e | [word="Y"] [word="Z"]) & [word="B"]'
    
    if len(sys.argv) > 1:
        input_query = sys.argv[1]
    
    print("Query:", input_query)
    tokens = list(QueryParser.tokenize(input_query))
    print("Tokens:", tokens)
    postfix_tokens = list(QueryParser.infix_to_postfix(tokens))
    print("Postfix Tokens:", postfix_tokens)
    test_query = QueryParser.to_query(postfix_tokens)
    print("Parsed Query:", test_query)
    
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
    
    start_time = time.time()
    variants = list(recursive_variants(test_query))
    end_time = time.time()
    
    print("Variants:")
    for variant in variants:
        print(simple_stringify(variant))
        #start = OffsetVariable(0)
        #query_range = QueryIndexer.query_range(test_query, start)
        #print(query_range)
        
    print(f"Time taken to compute {len(variants)} variants from {test_query}: {end_time - start_time:.6f} seconds")
