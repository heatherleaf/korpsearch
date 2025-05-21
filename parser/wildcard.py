from typing import Iterator
from query import Query

from dataclasses import dataclass

@dataclass
class WildcardQuery(Query):
    """
    A class representing a 'wildcard' query. Any number of tokens may match this query.
    """
    query: Query
    
    def components(self) -> Iterator[Query]:
        yield self.query
        
    def __eq__(self, other: Query) -> bool:
        if not isinstance(other, WildcardQuery):
            return False
        return self.query == other.query
    
    def __hash__(self) -> int:
        return hash(self.query)
    
    def __repr__(self) -> str:
        return f'{repr(self.query)}*'
    
    def __str__(self) -> str:
        return f'{str(self.query)}*'
    
    def __len__(self) -> int:
        return len(self.query)
    
