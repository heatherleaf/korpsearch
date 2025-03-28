import re
from pyeda.inter import And, Or, Not
from pyeda.boolalg.expr import exprvar

def tokenize(expression):
    """ Tokenizes a Boolean expression into variables, operators, and parentheses. """
    tokens = re.findall(r'[A-Za-z]+|[&|!()]', expression)
    return tokens

def parse_boolean_expr(tokens):
    """ Parses a tokenized Boolean expression and constructs a pyeda expression. """
    def parse_factor():
        """ Parses a single factor: variable, negation, or subexpression. """
        token = tokens.pop(0)
        if token.isalpha():  # Variable (e.g., 'a', 'b')
            return exprvar(token)
        elif token == "!":  # NOT operator
            return Not(parse_factor())
        elif token == "(":  # Subexpression
            expr = parse_or_expr()
            if tokens and tokens[0] == ")":
                tokens.pop(0)  # Consume closing parenthesis
            return expr
        else:
            raise ValueError(f"Unexpected token: {token}")

    def parse_and_expr():
        """ Parses an AND-expression, flattening chains like a & b & c into And(a, b, c). """
        factors = [parse_factor()]
        while tokens and (tokens[0] == "&" or tokens[0] == "("):
            if tokens[0] == "(": # Implicit "&" operator
                factors.append(parse_or_expr())
            else:
                tokens.pop(0)  # Consume "&"
                factors.append(parse_factor())
        return factors[0] if len(factors) == 1 else And(*factors)

    def parse_or_expr():
        """ Parses an OR-expression, flattening chains like a | b | c into Or(a, b, c). """
        terms = [parse_and_expr()]
        while tokens and tokens[0] == "|":
            tokens.pop(0)  # Consume "|"
            terms.append(parse_and_expr())
        return terms[0] if len(terms) == 1 else Or(*terms)

    return parse_or_expr()

def evaluate_boolean_expr(expression):
    """ Tokenizes and parses a Boolean expression into a pyeda expression. """
    tokens = tokenize(expression)
    return parse_boolean_expr(tokens)
