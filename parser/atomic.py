from enum import Enum
from typing import Iterator, NewType, Optional

from query import Query
from dataclasses import dataclass

Feature = NewType('Feature', str)
FValue = NewType('FValue', str)

class FeatureOperator(Enum):
    EQUALS = "="
    NOT_EQUALS = "!="
    CONTAINS = "contains"
    
@dataclass
class AtomicQuery(Query):
    feat: Feature
    value: FValue
    operator: FeatureOperator
    
    def components(self) -> Iterator[Query]:
        return iter([])
        
    def __len__(self) -> int:
        return 0
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicQuery):
            return False
        return (self.feat == other.feat and
                self.value == other.value and
                self.operator == other.operator)
        
    def __hash__(self) -> int:
        return hash((self.feat, self.value, self.operator))
    
    def __repr__(self) -> str:
        return f'[{self.feat} {self.operator.value} {self.value}]'
    
