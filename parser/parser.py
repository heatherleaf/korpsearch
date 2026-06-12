from enum import Enum
from typing import Iterator

from atomic import AtomicQuery, FeatureOperator
from wildcard import WildcardQuery
from query import Query
from concatenation import ConcatenationQuery
from disjunction import DisjunctionQuery
from conjunction import ConjunctionQuery
from negation import NegationQuery
from spacer import SpacerQuery

class QueryOperator(Enum):
    AND = "&"
    OR = "|"
    NOT = "!"
    CONCATENATE = ";"
    OPEN_PARENTHESIS = "("
    CLOSE_PARENTHESIS = ")"
    WILDCARD = "*"

class Associativity(Enum):
    LEFT = "left"
    RIGHT = "right"

def valid_enum_value(value: str, enum_class: Enum) -> bool:
    """
    Check if the given string is a valid enum value.
    """
    return value in enum_class._value2member_map_

class QueryParser:
    atomic_open = "["
    atomic_close = "]"
    empty_spacer = "e"
    
    precedence = {
        QueryOperator.WILDCARD: 5,
        QueryOperator.NOT: 4,
        QueryOperator.AND: 3,
        QueryOperator.CONCATENATE: 2,
        QueryOperator.OR: 1
    }
    
    associativity = {
        QueryOperator.WILDCARD: Associativity.LEFT,
        QueryOperator.NOT: Associativity.RIGHT,
        QueryOperator.AND: Associativity.LEFT,
        QueryOperator.CONCATENATE: Associativity.LEFT,
        QueryOperator.OR: Associativity.LEFT
    }
    
    arity = {
        QueryOperator.WILDCARD: 1,
        QueryOperator.NOT: 1,
        QueryOperator.AND: 2,
        QueryOperator.CONCATENATE: 2,
        QueryOperator.OR: 2
    }

    is_postfix = [
        QueryOperator.WILDCARD
    ]

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
        
        is_single_spacer = True
        
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
            
            is_single_spacer = False
            
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
                
        if is_single_spacer:
            yield SpacerQuery(1)
        
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
            elif query[start] == QueryParser.empty_spacer:
                if should_concatenate:
                    yield QueryOperator.CONCATENATE
                
                should_concatenate = True
                yield SpacerQuery(0)
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
            if isinstance(token, AtomicQuery) or isinstance(token, SpacerQuery):
                yield token
            elif isinstance(token, QueryOperator):
                arity = QueryParser.arity.get(token, 2)

                if token == QueryOperator.OPEN_PARENTHESIS:
                    stack.append(token)
                elif token == QueryOperator.CLOSE_PARENTHESIS:
                    while stack and stack[-1] != QueryOperator.OPEN_PARENTHESIS:
                        yield stack.pop()
                    if not stack:
                        raise ValueError("Mismatched parentheses")
                    stack.pop()
                elif arity == 1 and token in QueryParser.is_postfix:
                    yield token
                elif arity == 1 and token not in QueryParser.is_postfix:
                    stack.append(token)
                elif arity == 2:
                    # Binary operator logic (your existing implementation)
                    while (stack and stack[-1] != QueryOperator.OPEN_PARENTHESIS and
                        ((QueryParser.associativity[token] == Associativity.LEFT and
                            QueryParser.precedence[token] <= QueryParser.precedence[stack[-1]]) or
                            (QueryParser.associativity[token] == Associativity.RIGHT and
                            QueryParser.precedence[token] < QueryParser.precedence[stack[-1]]))):
                        yield stack.pop()
                    stack.append(token)
                else:
                    raise ValueError(f"Unhandled operator: {token}")
                
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
            if isinstance(token, AtomicQuery) or isinstance(token, SpacerQuery):
                stack.append(token)
            elif isinstance(token, QueryOperator):
                if token == QueryOperator.AND:
                    snd = stack.pop()
                    fst = stack.pop()
                    stack.append(ConjunctionQuery([fst, snd]))
                elif token == QueryOperator.OR:
                    snd = stack.pop()
                    fst = stack.pop()
                    stack.append(DisjunctionQuery([fst, snd]))
                elif token == QueryOperator.CONCATENATE:
                    snd = stack.pop()
                    fst = stack.pop()
                    stack.append(ConcatenationQuery([fst, snd]))
                elif token == QueryOperator.NOT:
                    q = stack.pop()
                    stack.append(NegationQuery(q))
                elif token == QueryOperator.WILDCARD:
                    q = stack.pop()
                    stack.append(WildcardQuery(q))
                else:
                    raise ValueError(f"Unexpected operator: {token}")
            else:
                raise ValueError(f"Unexpected token: {token}")
        
        if len(stack) > 1:
            # Concatenate remaining queries
            while len(stack) > 1:
                snd = stack.pop()
                fst = stack.pop()
                stack.append(ConcatenationQuery([fst, snd]))
        if len(stack) == 0:
            raise ValueError("Empty query")
        if len(stack) > 1:
            raise ValueError("Too many queries in stack")
        
        return stack[0]
