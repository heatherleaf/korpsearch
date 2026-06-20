from dataclasses import dataclass
from typing import Iterator

from variable import OffsetVariable, Variable
from query import Query

@dataclass
class Range:
    """
    A class representing a range of indices.
    
    A range may be a single index, a slice, or a list of indices.
    """
    start: OffsetVariable
    end: OffsetVariable
    
    def __eq__(self, other: 'Range') -> bool:
        if not isinstance(other, Range):
            return False
        return self.start == other.start and self.end == other.end
    
    def collect_variables(self, variables: set[Variable]) -> None:
        """
        Collect all variables in this range and add them to the set.
        """
        self.start.collect_variables(variables)
        self.end.collect_variables(variables)
    
    def __hash__(self) -> int:
        return hash((self.start, self.end))
    
    def __repr__(self) -> str:
        return f'{repr(self.start)}:{repr(self.end)}'
    
    def __str__(self) -> str:
        return f'{str(self.start)}:{str(self.end)}'

@dataclass
class QueryRange(Query):
    """
    A class representing a query with an annotated range.
    """
    query: Query
    range: Range
    
    @property
    def start(self) -> OffsetVariable:
        return self.range.start
    
    @property
    def end(self) -> OffsetVariable:
        return self.range.end
    
    def components(self) -> Iterator[Query]:
        yield from self.query
        
    def collect_variables(self, variables: set[Variable]) -> None:
        """
        Collect all variables in this query and add them to the set.
        """
        self.range.collect_variables(variables)
        
    def __eq__(self, other: Query) -> bool:
        if not isinstance(other, QueryRange):
            return False
        return self.query == other.query and self.range == other.range
    
    def __hash__(self) -> int:
        return hash((self.query, self.range))
    
    def __repr__(self) -> str:
        return f'{repr(self.query)}@{repr(self.range)}'
    
    def __str__(self) -> str:
        return f'{str(self.query)}@{str(self.range)}'
    
    def __len__(self) -> int:
        return len(self.query)
