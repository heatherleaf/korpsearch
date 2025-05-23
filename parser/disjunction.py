from typing import Iterator

from query import Query
from dataclasses import dataclass

@dataclass
class DisjunctionQuery(Query):
    """
    A class representing a disjunction of queries. A disjunction is a logical OR operation.
    """
    queries: list[Query]
    
    def __post_init__(self):
        # If any of the children are DisjunctionQuery, flatten them
        flattened_queries = []
        for query in self.queries:
            if isinstance(query, DisjunctionQuery):
                flattened_queries.extend(query.queries)
            else:
                flattened_queries.append(query)
        self.queries = flattened_queries

    def components(self) -> Iterator[Query]:
        yield from self.queries

    def __eq__(self, other: Query) -> bool:
        if not isinstance(other, DisjunctionQuery):
            return False
        return set(self.queries) == set(other.queries)
    
    def __hash__(self) -> int:
        return hash(tuple(self.queries))
    
    def __repr__(self) -> str:
        return f'({" ∨ ".join(repr(q) for q in self.queries)})'
    
    def __str__(self) -> str:
        return f'({" ∨ ".join(str(q) for q in self.queries)})'
    
    def __len__(self) -> int:
        return len(self.queries)
    
