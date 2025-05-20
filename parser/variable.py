from dataclasses import dataclass
from typing import Optional, Union

class Variable:
    """
    Abstract base class for all variables.
    """
    name: Optional[str] = None
    
    def collect_variables(self, variables: set['Variable']) -> None:
        """
        Collect all variables in this variable and add them to the set.
        """
        raise NotImplementedError("collect_variables must be implemented in subclasses")
    
@dataclass
class OffsetVariable(Variable):
    """
    A class representing an index variable.
    
    A variable may have a static offset or relative offset to another variable.
    """
    offset: Union[Variable, int]
    relative_to: Optional[Variable] = None
    
    def __post_init__(self):
        if self.relative_to is not None and not isinstance(self.relative_to, Variable):
            raise TypeError("relative_to must be a Variable or None")
    
    def collect_variables(self, variables: set[Variable]) -> None:
        """
        Collect all variables in this variable and add them to the set.
        """
        variables.add(self)
        if self.relative_to is not None:
            self.relative_to.collect_variables(variables)
        if isinstance(self.offset, Variable):
            self.offset.collect_variables(variables)
        
    def __add__(self, other: Union[int, Variable]) -> 'OffsetVariable':
        if other == 0:
            # Adding 0 has no effect, return self
            return self
        elif isinstance(other, int):
            # Add directly to self's offset if it's an int
            #if isinstance(self.offset, int):
            #    return OffsetVariable(offset=self.offset + other, relative_to=self.relative_to)
            #else:
                # self.offset is an IndexVariable, wrap and add the int to the chain
                return OffsetVariable(offset=other, relative_to=self)
        elif isinstance(other, Variable):
            # Wrap other inside self, combining offsets recursively
            return OffsetVariable(offset=other, relative_to=self)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'IndexVariable' and '{type(other).__name__}'")
            
    def __repr__(self) -> str:
        if self.relative_to is not None:
            return f"{self.relative_to}+{self.offset}"
        return str(self.offset)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OffsetVariable):
            return False
        return self.offset == other.offset and self.relative_to == other.relative_to
    
    def __hash__(self) -> int:
        return hash((self.offset, self.relative_to))
    
    def __str__(self) -> str:
        if self.name is not None:
            return f"{self.name}"
        if self.relative_to is not None:
            return f"{self.relative_to}+{self.offset}"
        return str(self.offset)

@dataclass
class DisjunctionVariable(Variable):
    """
    A class representing a disjunction variable.
    
    It may take one of several values.
    """
    values: list[Variable]
    
    def __post_init__(self):
        if not isinstance(self.values, list):
            raise TypeError("values must be a list of Variables")
        for value in self.values:
            if not isinstance(value, Variable):
                raise TypeError("All values must be Variables")
            
        # Remove duplicates
        new_values = []
        for value in self.values:
            if value not in new_values:
                new_values.append(value)
        self.values = new_values
        
    def collect_variables(self, variables: set[Variable]) -> None:
        """
        Collect all variables in this variable and add them to the set.
        """
        variables.add(self)
        for value in self.values:
            value.collect_variables(variables)
        
    def __add__(self, other: Union[int, Variable]) -> OffsetVariable:
        if isinstance(other, int):
            # Add directly to each value in the disjunction
            #return DisjunctionVariable([v + other for v in self.values])
            return OffsetVariable(offset=other, relative_to=self)
        elif isinstance(other, Variable):
            # Wrap other inside each value in the disjunction
            #return DisjunctionVariable([v + other for v in self.values])
            return OffsetVariable(offset=other, relative_to=self)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'DisjunctionVariable' and '{type(other).__name__}'")
    
    def __repr__(self) -> str:
        return f"({' | '.join(str(v) for v in self.values)})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DisjunctionVariable):
            return False
        return self.values == other.values
    
    def __hash__(self) -> int:
        return hash(tuple(self.values))
    
    def __str__(self) -> str:
        if self.name is not None:
            return f"{self.name}"
        return f"({' | '.join(str(v) for v in self.values)})"

@dataclass
class ConjunctionVariable(Variable):
    """
    A class representing a conjunction variable.
    
    It may take one of several values.
    """
    values: list[Variable]
    
    def __post_init__(self):
        if not isinstance(self.values, list):
            raise TypeError("values must be a list of Variables")
        for value in self.values:
            if not isinstance(value, Variable):
                raise TypeError("All values must be Variables")
            
        # Remove duplicates
        new_values = []
        for value in self.values:
            if value not in new_values:
                new_values.append(value)
        self.values = new_values
        
    def collect_variables(self, variables: set[Variable]) -> None:
        """
        Collect all variables in this variable and add them to the set.
        """
        variables.add(self)
        for value in self.values:
            value.collect_variables(variables)
        
    def __add__(self, other: Union[int, Variable]) -> OffsetVariable:
        if isinstance(other, int):
            # Add directly to each value in the disjunction
            #return DisjunctionVariable([v + other for v in self.values])
            return OffsetVariable(offset=other, relative_to=self)
        elif isinstance(other, Variable):
            # Wrap other inside each value in the disjunction
            #return DisjunctionVariable([v + other for v in self.values])
            return OffsetVariable(offset=other, relative_to=self)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'DisjunctionVariable' and '{type(other).__name__}'")
    
    def __repr__(self) -> str:
        return f"max({', '.join(str(v) for v in self.values)})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DisjunctionVariable):
            return False
        return self.values == other.values
    
    def __hash__(self) -> int:
        return hash(tuple(self.values))
    
    def __str__(self) -> str:
        if self.name is not None:
            return f"{self.name}"
        return f"max({', '.join(str(v) for v in self.values)})"

@dataclass
class UnknownVariable(Variable):
    """
    A class representing an unknown variable.
    
    This is a placeholder for variables that are not yet defined.
    """
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        
    def collect_variables(self, variables: set[Variable]) -> None:
        """
        Collect all variables in this variable and add them to the set.
        """
        variables.add(self)
    
    def __add__(self, other: Union[int, Variable]) -> Variable:
        if isinstance(other, int):
            # Add directly to self's offset if it's an int
            return OffsetVariable(offset=other, relative_to=self)
        elif isinstance(other, Variable):
            # Wrap other inside self
            return OffsetVariable(offset=other, relative_to=self)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'UnknownVariable' and '{type(other).__name__}'")
    
    def __repr__(self) -> str:
        return f"{self.name if self.name else '?'}"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnknownVariable):
            return False
        return self.name == other.name
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __str__(self) -> str:
        return repr(self)
    
