from typing import Iterator

from query import Query
from dataclasses import dataclass

@dataclass
class ConjunctionQuery(Query):
    """
    A class representing a conjunction of queries. A conjunction is a logical AND operation.
    """
    queries: list[Query]
    
    def __post_init__(self):
        # If any of the children are ConjunctionQuery, flatten them
        flattened_queries = []
        for query in self.queries:
            if isinstance(query, ConjunctionQuery):
                flattened_queries.extend(query.queries)
            else:
                flattened_queries.append(query)
        self.queries = flattened_queries

    def components(self) -> Iterator[Query]:
        yield from self.queries

    def __eq__(self, other: Query) -> bool:
        if not isinstance(other, ConjunctionQuery):
            return False
        return self.queries == other.queries
    
    def __hash__(self) -> int:
        return hash(tuple(self.queries))
    
    def __repr__(self) -> str:
        return f'({" ∧ ".join(repr(q) for q in self.queries)})'
    
    def __str__(self) -> str:
        return f'({" ∧ ".join(str(q) for q in self.queries)})'
    
    def __len__(self) -> int:
        return len(self.queries)
    
