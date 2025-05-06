from enum import Enum
from typing import Iterator, NewType
from dataclasses import dataclass, field

from util import Feature, FValue

import sys


class Query:
    def expand(self, i: int) -> Iterator[tuple['Query', int]]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def atomics(self) -> Iterator['AtomicQuery']:
        raise NotImplementedError("Subclasses must implement this method")

class FeatureOperator(Enum):
    EQUALS = "="
    NOT_EQUALS = "!="
    CONTAINS = "contains"

class AtomicIdentity:
    def __init__(self, feat: Feature, value: FValue, operator: FeatureOperator) -> None:
        self.feat = feat
        self.value = value
        self.operator = operator
        
        assert isinstance(feat, bytes), f"Feature must be a bytestring: {feat!r}"
        assert isinstance(value, bytes), f"Value must be a bytestring: {value!r}"
        assert isinstance(operator, FeatureOperator), f"Operator must be a FeatureOperator: {operator!r}"
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicIdentity):
            return False
        return (self.feat == other.feat and
                self.value == other.value and
                self.operator == other.operator)
    
    def __hash__(self) -> int:
        return hash((self.feat, self.value, self.operator))

@dataclass
class Indexed:
    index: int | None = field(default=None, init=False)

@dataclass
class AtomicQuery(Query, Indexed):
    feat: Feature
    value: FValue
    operator: FeatureOperator
    
    def expand(self, i: int) -> Iterator[tuple['Query', int]]:
        copy = AtomicQuery(self.feat, self.value, self.operator)
        copy.index = i
        yield (copy, i)

    def atomics(self) -> Iterator['AtomicQuery']:
        yield self
        
    def identity(self) -> AtomicIdentity:
        return AtomicIdentity(self.feat, self.value, self.operator)
    
    def __len__(self) -> int:
        return 1
    
    def __repr__(self) -> str:
        position_str = f"@{self.index}" if self.index is not None else ""
        return f"[{self.feat.decode()}{self.operator.value}{self.value.decode()}{position_str}]"
    
@dataclass
class ConjunctionQuery(Query):
    fst: Query
    snd: Query
    
    def expand(self, i: int) -> Iterator[tuple['Query', int]]:
        for q1, i1 in self.fst.expand(i):
            for q2, i2 in self.snd.expand(i1):
                yield (ConjunctionQuery(q1, q2), i2)
                
    def atomics(self) -> Iterator['AtomicQuery']:
        for q1 in self.fst.atomics():
            yield q1
        for q2 in self.snd.atomics():
            yield q2
    
    def __len__(self) -> int:
        return len(self.fst) + len(self.snd)
    
    def __repr__(self) -> str:
        return f"({self.fst} & {self.snd})"

@dataclass
class DisjunctionQuery(Query):
    fst: Query
    snd: Query
    
    def expand(self, i: int) -> Iterator[tuple['Query', int]]:
        for q1, i1 in self.fst.expand(i):
            yield (q1, i1)
        for q2, i2 in self.snd.expand(i):
            yield (q2, i2)
            
    def atomics(self) -> Iterator['AtomicQuery']:
        for q1 in self.fst.atomics():
            yield q1
        for q2 in self.snd.atomics():
            yield q2
            
    def __len__(self) -> int:
        return max(len(self.fst), len(self.snd))
    
    def __repr__(self) -> str:
        return f"({self.fst} | {self.snd})"

@dataclass
class NegationQuery(Query):
    query: Query
    
    def expand(self, i: int) -> Iterator[tuple['Query', int]]:
        for q, i1 in self.query.expand(i):
            yield (NegationQuery(q), i1)
            
    def atomics(self) -> Iterator['AtomicQuery']:
        for q in self.query.atomics():
            yield q
            
    def __len__(self) -> int:
        return len(self.query)
            
    def __repr__(self) -> str:
        return f"!{self.query}"

@dataclass
class ConcatenationQuery(Query):
    fst: Query
    snd: Query
    
    def expand(self, i: int) -> Iterator[tuple['Query', int]]:
        for fst_query, fst_index in self.fst.expand(i):
            for snd_query, snd_index in self.snd.expand(fst_index + 1):
                yield (ConjunctionQuery(fst_query, snd_query), snd_index)
                
    def atomics(self) -> Iterator['AtomicQuery']:
        for q1 in self.fst.atomics():
            yield q1
        for q2 in self.snd.atomics():
            yield q2
                
    def __len__(self) -> int:
        return len(self.fst) + len(self.snd)
                
    def __repr__(self) -> str:
        return f"({self.fst} ; {self.snd})"

@dataclass
class WildcardQuery(Query, Indexed):

    def expand(self, i: int) -> Iterator[tuple['Query', int]]:
        copy = WildcardQuery()
        copy.index = i
        yield (copy, i)
        
    def atomics(self) -> Iterator['AtomicQuery']:
        yield self
        
    def __repr__(self) -> str:
        return f"[]@{self.index}"

class QueryOperator(Enum):
    AND = "&"
    OR = "|"
    NOT = "!"
    CONCATENATE = ";"
    OPEN_PARENTHESIS = "("
    CLOSE_PARENTHESIS = ")"
    
def valid_enum_value(value: str, enum_class: Enum) -> bool:
    """
    Check if the given string is a valid enum value.
    """
    return value in enum_class._value2member_map_

class QueryParser:
    atomic_open = "["
    atomic_close = "]"
    precedence = {
        QueryOperator.AND: 3,
        QueryOperator.OR: 1,
        QueryOperator.NOT: 4,
        QueryOperator.CONCATENATE: 2
    }
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def _skip_whitespace(query: str, start: int, end: int) -> int:
        while start < end and query[start].isspace():
            start += 1
        return start
    
    @staticmethod
    def _parse_atomic(query: str, start: int) -> Iterator[QueryOperator | AtomicQuery | int]:
        if query[start] != QueryParser.atomic_open:
            raise ValueError("Invalid feature format")
        
        start += 1  # Skip opening bracket
        end = len(query)
        
        is_wildcard = True
        
        yield QueryOperator.OPEN_PARENTHESIS
        
        while start < end:
            # Skip whitespace
            start = QueryParser._skip_whitespace(query, start, end)
            if start >= end:
                break
            
            if query[start] == QueryParser.atomic_close:
                # Skip closing bracket
                start += 1
                break
            
            is_wildcard = False
            
            feat_start = start
            # While alphanumeric
            while start < end and query[start].isalnum():
                start += 1
            feat = query[feat_start:start]
            
            if not feat:
                raise ValueError("No feature found")
            
            start = QueryParser._skip_whitespace(query, start, end)
            if start >= end:
                raise ValueError("No operator found")
            
            operator_start = start
            # Take until quote or whitespace
            while start < end and query[start] != '"' and not query[start].isspace():
                start += 1
            if start >= end:
                raise ValueError("No operator found")
            
            operator = query[operator_start:start]
            # Check if it's a valid operator
            if not valid_enum_value(operator, FeatureOperator):
                raise ValueError(f"Invalid operator: {operator}")
            
            start = QueryParser._skip_whitespace(query, start, end)
            if start >= end:
                raise ValueError("No value found")
            
            if query[start] != '"':
                raise ValueError("Expected opening quote for value")
            start += 1
            value_start = start
            # Take until closing quote
            while start < end and query[start] != '"':
                start += 1
            if start >= end:
                raise ValueError("No closing quote for value")
            value = query[value_start:start]
            start += 1  # Skip closing quote
            
            yield AtomicQuery(feat.encode(), value.encode(), FeatureOperator(operator))
            
            # Skip whitespace
            start = QueryParser._skip_whitespace(query, start, end)    
            if start >= end:
                break
            
            if query[start] in {QueryOperator.AND.value, QueryOperator.OR.value}:
                yield QueryOperator(query[start])
                start += 1
                # Skip whitespace
                start = QueryParser._skip_whitespace(query, start, end)
                if start >= end:
                    break
            elif query[start] == QueryOperator.CONCATENATE.value:
                raise ValueError("Concatenation operator not allowed in atomic query")
            elif query[start] == QueryParser.atomic_close:
                # Skip closing bracket
                start += 1
                break
            
            else:
                yield QueryOperator.AND
                
        if is_wildcard:
            yield WildcardQuery()
                
        yield QueryOperator.CLOSE_PARENTHESIS
        
        yield start
    
    @staticmethod
    def tokenize(query: str) -> Iterator[QueryOperator | AtomicQuery]:
        """
        Tokenizes the query string into a list of operators and atomic queries.
        """
        start = 0
        end = len(query)
        
        should_concatenate = False
        
        while start < end:
            # Skip whitespace
            start = QueryParser._skip_whitespace(query, start, end)
            if start >= end:
                break
            
            if query[start] == QueryParser.atomic_open:
                if should_concatenate:
                    yield QueryOperator.CONCATENATE
                
                # Find end of atomic query
                generator = QueryParser._parse_atomic(query, start)
                for token in generator:
                    if isinstance(token, int):
                        start = token
                    else:
                        yield token
                should_concatenate = True
            elif valid_enum_value(query[start], QueryOperator):
                should_concatenate = False
                # Yield operator
                yield QueryOperator(query[start])
                start += 1
            else:
                raise ValueError(f"Unexpected character: {query[start]} at position {start}")
    
    @staticmethod
    def infix_to_postfix(tokens: Iterator[QueryOperator | AtomicQuery]) -> Iterator[QueryOperator | AtomicQuery]:
        """
        Converts infix notation to postfix notation using the Shunting Yard algorithm.
        """
        stack = []
        
        for token in tokens:
            if isinstance(token, AtomicQuery) or isinstance(token, WildcardQuery):
                yield token
            elif isinstance(token, QueryOperator):
                if token == QueryOperator.OPEN_PARENTHESIS:
                    stack.append(token)
                elif token == QueryOperator.CLOSE_PARENTHESIS:
                    while stack and stack[-1] != QueryOperator.OPEN_PARENTHESIS:
                        yield stack.pop()
                    if not stack or stack[-1] != QueryOperator.OPEN_PARENTHESIS:
                        raise ValueError("Mismatched parentheses")
                    stack.pop()
                elif token in QueryParser.precedence:
                    while (stack and stack[-1] != QueryOperator.OPEN_PARENTHESIS and
                           QueryParser.precedence[token] <= QueryParser.precedence[stack[-1]]):
                        yield stack.pop()
                    stack.append(token)
                else:
                    raise ValueError(f"Unexpected operator: {token}")
        while stack:
            if stack[-1] == QueryOperator.OPEN_PARENTHESIS:
                raise ValueError("Mismatched parentheses")
            yield stack.pop()
            
    @staticmethod
    def to_query(postfix_tokens: Iterator[QueryOperator | AtomicQuery]) -> Query:
        """
        Converts postfix tokens into a query object.
        """
        stack = []
        
        for token in postfix_tokens:
            if isinstance(token, AtomicQuery) or isinstance(token, WildcardQuery):
                stack.append(token)
            elif isinstance(token, QueryOperator):
                if token == QueryOperator.AND:
                    snd = stack.pop()
                    fst = stack.pop()
                    stack.append(ConjunctionQuery(fst, snd))
                elif token == QueryOperator.OR:
                    snd = stack.pop()
                    fst = stack.pop()
                    stack.append(DisjunctionQuery(fst, snd))
                elif token == QueryOperator.CONCATENATE:
                    snd = stack.pop()
                    fst = stack.pop()
                    stack.append(ConcatenationQuery(fst, snd))
                elif token == QueryOperator.NOT:
                    q = stack.pop()
                    stack.append(NegationQuery(q))
                else:
                    raise ValueError(f"Unexpected operator: {token}")
            else:
                raise ValueError(f"Unexpected token: {token}")
        
        if len(stack) > 1:
            # Concatenate remaining queries
            while len(stack) > 1:
                snd = stack.pop()
                fst = stack.pop()
                stack.append(ConcatenationQuery(fst, snd))
        if len(stack) == 0:
            raise ValueError("Empty query")
        if len(stack) > 1:
            raise ValueError("Too many queries in stack")
        
        return stack[0]
    
    @staticmethod
    def optimize_positions(expansions: list[tuple['Query', int]]) -> Iterator[tuple['Query', int]]:
        """
        For example, turn:
         (([word=big@0] & [word=dog@1]) & [word=is@2])
         (([word=small@0] & [word=dog@1]) & [word=is@2])
         (([word=tiny@0] & [word=dog@1]) & [word=is@2])
         ([word=cat@0] & [word=is@1])
        into:
         (([word=big@-1] & [word=dog@0]) & [word=is@1])
         (([word=small@-1] & [word=dog@0]) & [word=is@1])
         (([word=tiny@-1] & [word=dog@0]) & [word=is@1])
         ([word=cat@0] & [word=is@1])
         
        Because word=is is the same in all of them.
        """
        
        # Find common atomics
        atomics: map[AtomicIdentity, list[AtomicQuery]] = {}
        for query, position in expansions:
            for atomic in query.atomics():
                identity = atomic.identity()
                if identity not in atomics:
                    atomics[identity] = []
                atomics[identity].append(atomic)
        
        biggest_identity = None
        biggest_count = 0
        for identity, atomics_list in atomics.items():
            if len(atomics_list) > biggest_count:
                biggest_identity = identity
                biggest_count = len(atomics_list)
        if biggest_identity is None:
            raise ValueError("No common atomics found")
        
        # Shift positions so that the biggest identity is at 0
        for query, position in expansions:
            biggest_pos = None
            for atomic in query.atomics():
                if atomic.identity() == biggest_identity:
                    biggest_pos = atomic.position
                    break
            if biggest_pos is not None:
                for atomic in query.atomics():
                    if atomic.identity() == biggest_identity:
                        atomic.position = 0
                    else:
                        atomic.position = atomic.position - biggest_pos
            yield (query, position)

# Example usage in the main block
if __name__ == "__main__":
    input_query = '([word="grand" lemma!="la"] | [word="here"]) [] [] ([word="hit" lemma="la"])'
    
    if len(sys.argv) > 1:
        input_query = sys.argv[1]
    
    # Tokenize the example query
    print("Query:", input_query)
    tokens = list(QueryParser.tokenize(input_query))
    print("Tokens:", tokens)
    postfix_tokens = list(QueryParser.infix_to_postfix(tokens))
    print("Postfix Tokens:", postfix_tokens)
    query = QueryParser.to_query(postfix_tokens)
    print("Parsed Query:", query)
    
    # Expand the query into DNF
    expanded_query = query.expand(0)
    print("\nExpanded Query in DNF:")
    for expanded_query, position in expanded_query:
        print(f"Expanded Query: {expanded_query}, Position: {position}")