import sympy
from sympy import symbols, sympify, simplify_logic
from sympy.logic.boolalg import to_dnf
import itertools

from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols

import re

def get_symbols(exprs):
    """Find all uppercase alphabetic symbols in the given expressions"""
    import re
    all_vars = set()
    for expr in exprs:
        all_vars.update(re.findall(r'\b[A-Z]\b', expr))
    return symbols(' '.join(sorted(all_vars)))

def parse_all_exprs(source_str, variant_strs):
    all_exprs = [source_str] + variant_strs
    # Replace ! with ~
    all_exprs = [e.replace("!", "~") for e in all_exprs]

    # Extract symbols (A, B, C, etc.)
    symbol_list = get_symbols(all_exprs)
    local_dict = {str(s): s for s in symbol_list}

    # Parse each expression safely
    return [parse_expr(expr, local_dict=local_dict, evaluate=False) for expr in all_exprs]


def get_variables(exprs):
    """Get the set of all Boolean variables across all expressions."""
    vars_set = set()
    for expr in exprs:
        vars_set |= expr.free_symbols
    return sorted(vars_set, key=lambda s: s.name)

def expr_to_model_set(expr, all_vars):
    """Convert a Boolean expression to the set of satisfying truth assignments."""
    expr = to_dnf(expr, simplify=True)
    models = set()

    if expr == False:
        return models  # No models satisfy this
    elif expr == True:
        all_assignments = list(itertools.product([False, True], repeat=len(all_vars)))
        for vals in all_assignments:
            models.add(tuple(zip(all_vars, vals)))
        return models

    if isinstance(expr, sympy.Or):
        clauses = expr.args
    else:
        clauses = [expr]

    for clause in clauses:
        # Extract the literals in the clause
        if isinstance(clause, sympy.And):
            literals = clause.args
        else:
            literals = [clause]

        assignment = {}
        for lit in literals:
            if isinstance(lit, sympy.Symbol):
                assignment[lit] = True
            elif isinstance(lit, sympy.Not) and isinstance(lit.args[0], sympy.Symbol):
                assignment[lit.args[0]] = False
            else:
                raise ValueError(f"Unsupported literal format: {lit}")

        # Fill in unspecified vars with both True/False to cover all models
        unspecified = [v for v in all_vars if v not in assignment]
        for vals in itertools.product([False, True], repeat=len(unspecified)):
            full_assignment = assignment.copy()
            for var, val in zip(unspecified, vals):
                full_assignment[var] = val
            models.add(tuple(sorted(full_assignment.items(), key=lambda p: p[0].name)))
    
    return models

def check_equivalence(source_str, variant_strs):
    # Replace '!' with '~' (logical NOT)
    all_exprs_str = [source_str] + variant_strs
    all_exprs_str = [expr.replace("!", "~") for expr in all_exprs_str]

    # Find all uppercase variable names (A, B, ..., Z)
    all_vars = set()
    for expr in all_exprs_str:
        all_vars.update(re.findall(r'\b[A-Z]\b', expr))
    sym_vars = symbols(' '.join(sorted(all_vars)))
    local_dict = {str(s): s for s in sym_vars}

    # Parse all expressions
    parsed_exprs = [parse_expr(expr, local_dict=local_dict, evaluate=False) for expr in all_exprs_str]

    # Simplify the first expression as the base
    base = simplify_logic(parsed_exprs[0], form='dnf')

    # Check if all variants match the base
    for i, expr in enumerate(parsed_exprs[1:], start=1):
        simplified = simplify_logic(expr, form='dnf')
        if simplified != base:
            print(f"❌ Variant {i} does not match the source.")
            print(f"    Expected: {base}")
            print(f"    Got     : {simplified}")
        else:
            print(f"✅ Variant {i} matches the source.")

if __name__ == "__main__":
    # Example usage
    source = "((A & (B | C)) & ((D | E) | F))"
    variants = [
            " ((A & (B | C)) & ((D | E) | F)) ",
            " ((((A & B) | (A & C)) & (D | E)) | (((A & B) | (A & C)) & F)) ",
            " (((((A & B) | (A & C)) & D) | (((A & B) | (A & C)) & E)) | (((A & B) & F) | ((A & C) & F))) ",
            " (((((A & B) & D) | ((A & C) & D)) | (((A & B) & E) | ((A & C) & E))) | (((A & B) & F) | ((A & C) & F))) ",
            " (((((A & B) & D) | ((A & C) & D)) | (((A & B) | (A & C)) & E)) | (((A & B) & F) | ((A & C) & F))) ",
            " (((((A & B) | (A & C)) & D) | (((A & B) & E) | ((A & C) & E))) | (((A & B) & F) | ((A & C) & F))) ",
            " (((((A & B) | (A & C)) & D) | (((A & B) | (A & C)) & E)) | (((A & B) | (A & C)) & F)) ",
            " (((((A & B) & D) | ((A & C) & D)) | (((A & B) & E) | ((A & C) & E))) | (((A & B) | (A & C)) & F)) ",
            " (((((A & B) & D) | ((A & C) & D)) | (((A & B) | (A & C)) & E)) | (((A & B) | (A & C)) & F)) ",
            " (((((A & B) | (A & C)) & D) | (((A & B) & E) | ((A & C) & E))) | (((A & B) | (A & C)) & F)) ",
            " ((((A & B) & (D | E)) | ((A & C) & (D | E))) | (((A & B) & F) | ((A & C) & F))) ",
            " (((((A & B) & D) | ((A & B) & E)) | (((A & C) & D) | ((A & C) & E))) | (((A & B) & F) | ((A & C) & F))) ",
            " (((((A & B) & D) | ((A & B) & E)) | ((A & C) & (D | E))) | (((A & B) & F) | ((A & C) & F))) ",
            " ((((A & B) & (D | E)) | (((A & C) & D) | ((A & C) & E))) | (((A & B) & F) | ((A & C) & F))) ",
            " ((((A & B) & (D | E)) | ((A & C) & (D | E))) | (((A & B) | (A & C)) & F)) ",
            " (((((A & B) & D) | ((A & B) & E)) | (((A & C) & D) | ((A & C) & E))) | (((A & B) | (A & C)) & F)) ",
            " (((((A & B) & D) | ((A & B) & E)) | ((A & C) & (D | E))) | (((A & B) | (A & C)) & F)) ",
            " ((((A & B) & (D | E)) | (((A & C) & D) | ((A & C) & E))) | (((A & B) | (A & C)) & F)) ",
            " ((((A & B) | (A & C)) & (D | E)) | (((A & B) & F) | ((A & C) & F))) ",
            " (((A & (B | C)) & (D | E)) | ((A & (B | C)) & F)) ",
            " (((((A & B) | (A & C)) & D) | (((A & B) | (A & C)) & E)) | ((A & (B | C)) & F)) ",
            " (((((A & B) & D) | ((A & C) & D)) | (((A & B) & E) | ((A & C) & E))) | ((A & (B | C)) & F)) ",
            " (((((A & B) & D) | ((A & C) & D)) | (((A & B) | (A & C)) & E)) | ((A & (B | C)) & F)) ",
            " (((((A & B) | (A & C)) & D) | (((A & B) & E) | ((A & C) & E))) | ((A & (B | C)) & F)) ",
            " ((((A & (B | C)) & D) | ((A & (B | C)) & E)) | (((A & B) | (A & C)) & F)) ",
            " (((((A & B) | (A & C)) & D) | ((A & (B | C)) & E)) | (((A & B) & F) | ((A & C) & F))) ",
            " (((((A & B) & D) | ((A & C) & D)) | ((A & (B | C)) & E)) | (((A & B) & F) | ((A & C) & F))) ",
            " (((((A & B) | (A & C)) & D) | ((A & (B | C)) & E)) | (((A & B) | (A & C)) & F)) ",
            " (((((A & B) & D) | ((A & C) & D)) | ((A & (B | C)) & E)) | (((A & B) | (A & C)) & F)) ",
            " ((((A & (B | C)) & D) | (((A & B) | (A & C)) & E)) | (((A & B) & F) | ((A & C) & F))) ",
            " ((((A & (B | C)) & D) | (((A & B) & E) | ((A & C) & E))) | (((A & B) & F) | ((A & C) & F))) ",
            " ((((A & (B | C)) & D) | (((A & B) | (A & C)) & E)) | (((A & B) | (A & C)) & F)) ",
            " ((((A & (B | C)) & D) | (((A & B) & E) | ((A & C) & E))) | (((A & B) | (A & C)) & F)) ",
            " ((((A & (B | C)) & D) | ((A & (B | C)) & E)) | (((A & B) & F) | ((A & C) & F))) ",
            " ((((A & (B | C)) & D) | ((A & (B | C)) & E)) | ((A & (B | C)) & F)) ",
            " (((((A & B) | (A & C)) & D) | ((A & (B | C)) & E)) | ((A & (B | C)) & F)) ",
            " (((((A & B) & D) | ((A & C) & D)) | ((A & (B | C)) & E)) | ((A & (B | C)) & F)) ",
            " ((((A & (B | C)) & D) | (((A & B) | (A & C)) & E)) | ((A & (B | C)) & F)) ",
            " ((((A & (B | C)) & D) | (((A & B) & E) | ((A & C) & E))) | ((A & (B | C)) & F)) ",
            " ((((A & B) | (A & C)) & (D | E)) | ((A & (B | C)) & F)) ",
            " ((((A & B) & (D | E)) | ((A & C) & (D | E))) | ((A & (B | C)) & F)) ",
            " (((((A & B) & D) | ((A & B) & E)) | (((A & C) & D) | ((A & C) & E))) | ((A & (B | C)) & F)) ",
            " (((((A & B) & D) | ((A & B) & E)) | ((A & C) & (D | E))) | ((A & (B | C)) & F)) ",
            " ((((A & B) & (D | E)) | (((A & C) & D) | ((A & C) & E))) | ((A & (B | C)) & F)) ",
            " (((A & (B | C)) & (D | E)) | (((A & B) | (A & C)) & F)) ",
            " (((A & (B | C)) & (D | E)) | (((A & B) & F) | ((A & C) & F))) ",
            " (((A & B) | (A & C)) & ((D | E) | F)) ",
            " (((A & B) & ((D | E) | F)) | ((A & C) & ((D | E) | F))) ",
            " ((((A & B) & (D | E)) | ((A & B) & F)) | (((A & C) & (D | E)) | ((A & C) & F))) ",
            " (((((A & B) & D) | ((A & B) & E)) | ((A & B) & F)) | ((((A & C) & D) | ((A & C) & E)) | ((A & C) & F))) ",
            " (((((A & B) & D) | ((A & B) & E)) | ((A & B) & F)) | (((A & C) & (D | E)) | ((A & C) & F))) ",
            " ((((A & B) & (D | E)) | ((A & B) & F)) | ((((A & C) & D) | ((A & C) & E)) | ((A & C) & F))) ",
            " ((((A & B) & (D | E)) | ((A & B) & F)) | ((A & C) & ((D | E) | F))) ",
            " (((((A & B) & D) | ((A & B) & E)) | ((A & B) & F)) | ((A & C) & ((D | E) | F))) ",
            " (((A & B) & ((D | E) | F)) | (((A & C) & (D | E)) | ((A & C) & F))) ",
            " (((A & B) & ((D | E) | F)) | ((((A & C) & D) | ((A & C) & E)) | ((A & C) & F))) ",
    ]

    check_equivalence(source, variants)
