from typing import Iterator
from query import Query

from dataclasses import dataclass

@dataclass
class SpacerQuery(Query):
    """
    A class representing a query of arbitrary length that matches anything.
    """
    length: int = 0
    
    def __post_init__(self):
        if self.length < 0:
            raise ValueError("Length must be non-negative")
    
    def components(self) -> Iterator[Query]:
        return iter([])

    def __eq__(self, other: Query) -> bool:
        return isinstance(other, SpacerQuery)
    
    def __hash__(self) -> int:
        return hash(SpacerQuery)
    
    def __repr__(self) -> str:
        if (self.length > 0):
            return f"ε({self.length})"
        return "ε"
    
    def __str__(self) -> str:
        return repr(self)
    
    def __len__(self) -> int:
        return 0
