
import re
import itertools
from typing import Literal
from collections.abc import Iterator, Sequence
from string import ascii_lowercase

from pyeda.inter import expr, Expression, And, Or
from pyeda.boolalg.expr import exprvar, Complement

from index import KnownLiteral, DisjunctiveGroup, TemplateLiteral, Template, Instance, mkInstance, Index
from corpus import Corpus
from util import Feature, FValue, SENTENCE, START

################################################################################
## Queries

QueryElement = KnownLiteral | DisjunctiveGroup

def get_query_literals(query_element: QueryElement) -> tuple[KnownLiteral, ...]:
    if isinstance(query_element, KnownLiteral):
        return (query_element,)
    else: # isinstance(query_element, DisjunctiveGroup):
        return query_element.literals


class Query:
    _token_re_string = r' ([|&]?) \s* ([\w_]+) \s* (!?) (?:=|contains) \s* "\s*([^"]+?)\s*" '
    token_regex = re.compile(rf'               {_token_re_string}                 ', re.X)
    query_regex = re.compile(rf'       \[ \s* ({_token_re_string} \s*)* \]        ', re.X)
    fullq_regex = re.compile(rf'^ \s* (\[ \s* ({_token_re_string} \s*)* \] \s*)+ $', re.X)

    corpus: Corpus
    literals: list[QueryElement]
    features: set[Feature]
    # featured_query: dict[Feature, list[tuple[bool, int, InternedString, InternedString]]]
    template: Template | None

    def __init__(self, corpus: Corpus, literals: Sequence[QueryElement]) -> None:
        self.corpus = corpus
        self.literals = sorted(set(literals), key=get_query_literals)
        self.features = {
            lit.feature for query_element in self.literals
            for lit in get_query_literals(query_element)
        }

        # We cannot handle non-atomic querues with only negative literals
        # -A & -B == -(A v B), since we cannot handle union (yet)
        #if len(self) > 1 and self.is_negative():
        #    raise ValueError(f"Cannot handle non-atomic queries with no positive literals: {self}")

        # # This is a variant of self.query, optimised for checking a query at a corpus position:
        # self.featured_query = {f: [] for f in self.features}
        # for lit in self.literals:
        #     self.featured_query[lit.feature].append(
        #         (lit.negative, lit.offset, lit.value, lit.value2)
        #     )

        # We precompute the associated query template. It raises a ValueError if it's not valid.
        try:
            if self.contains_disjunction():
                self.template = None
            elif self.is_negative():
                self.template = Template(
                    [TemplateLiteral(lit.offset-self.min_offset(), lit.feature) for lit in self.negative_literals()],
                )
            else:
                self.template = Template(
                    [TemplateLiteral(lit.offset-self.min_offset(), lit.feature) for lit in self.positive_literals()],
                    [KnownLiteral(True, lit.offset-self.min_offset(), lit.feature, lit.value, lit.value2, corpus) for lit in self.negative_literals()],
                )
        except Exception as e:
            raise ValueError(f"Invalid query: {self} ({e})") from e


    def __str__(self) -> str:
        return '[' + ']&['.join(map(str, self.literals)) + ']'

    def __repr__(self) -> str:
        return f"Query({self.literals})"

    def __len__(self) -> int:
        return len(self.literals)

    def min_offset(self) -> int:
        return min(lit.offset for lit in self.literals)

    def max_offset(self) -> int:
        return max(lit.offset for lit in self.literals)

    def is_negative(self) -> bool:
        return not self.positive_literals()

    def positive_literals(self) -> list[KnownLiteral]:
        return [lit for lit in self.literals if isinstance(lit, KnownLiteral) if not lit.negative]

    def negative_literals(self) -> list[KnownLiteral]:
        return [lit for lit in self.literals if isinstance(lit, KnownLiteral) if lit.negative]

    def instance(self) -> Instance:
        if self.is_negative():
            return mkInstance([(lit.value, lit.value2) for lit in self.negative_literals()])
        else:
            return mkInstance([(lit.value, lit.value2) for lit in self.positive_literals()])

    def index(self) -> Index:
        assert self.template
        return Index.get(self.corpus, self.template)

    def contains_disjunction(self) -> bool:
        return any(isinstance(lit, DisjunctiveGroup) for lit in self.literals)

    def contains_prefix(self) -> bool:
        return any(lit.is_prefix() for lit in self.literals)

    def expand(self) -> Iterator['Query']:
        groups = [group.literals for group in self.literals if isinstance(group, DisjunctiveGroup)]
        singles = [lit for lit in self.literals if isinstance(lit, KnownLiteral)]

        if not singles:
            for group in groups:
                yield Query(self.corpus, group)
            return

        for group in itertools.product(*groups):
            yield Query(self.corpus, singles + list(group))

    def subqueries(self) -> Iterator['Query']:
        # Subqueries are generated in decreasing order of complexity
        for n in reversed(range(len(self))):
            for literals in itertools.combinations(self.literals, n+1):
                try:
                    yield Query(self.corpus, literals)
                except ValueError:
                    pass

    def subsumed_by(self, others: list['Query']) -> bool:
        other_literals = {lit for other_query in others for lit in other_query.literals}
        return set(self.literals).issubset(other_literals)

    def check_sentence(self, sent: int) -> bool:
        positions = self.corpus.sentence_positions(sent)
        min_offset = self.min_offset()
        max_offset = self.max_offset()
        return any(
            self.check_position(pos)
            for pos in range(positions.start - min_offset, positions.stop - max_offset)
        )

    # Right now this function is not optimized and can not give false for cases such as [a&b|c&d]
    def check_position(self, pos: int) -> bool:
        return all(
            lit.check_position(self.corpus, pos) for lit in self.literals
        )
        # return all(
        #     (self.corpus.tokens[lit.feature][pos + lit.offset] == lit.value) != lit.negative
        #     for lit in self.literals
        # )
        # This is an optimised (but less readable) version of the code above:
        # for feature, values in self.featured_query.items():
        #     lookup = self.corpus.tokens[feature]
        #     if any((lookup[pos+offset] >= value and lookup[pos+offset] <= value2) == negative for (negative, offset, value, value2) in values):
        #         return False
        # return True

    @staticmethod
    def _classify_value(value: str) -> Literal['normal'] | Literal['prefix'] | Literal['suffix'] | Literal['regex']:
        if value.isalnum():
            return 'normal'
        elif value.endswith('.*') and value[:-2].isalnum():
            return 'prefix'
        elif value.startswith('.*') and value[2:].isalnum():
            return 'suffix'
        else:
            return 'regex'
        
    @staticmethod
    def _expand_expression_parts(expression: str) -> list[list[str] | str]:
        """
        Expands: "([pos="DT"] [pos="JJ"] | [pos="PN"]) [word="katt"]"
        to: [["[pos="DT"]", "[pos="JJ"]", "[word="katt"]"], "|", ["[pos="PN"]", "[word="katt"]"]]
        
        The seperator may be either "|" or "&"
        """
        
    @staticmethod
    def _evalute_literal(corpus: Corpus, offset: int, featstr: str, negated: str, valstr: str) -> list[KnownLiteral]:
        feature = Feature(featstr.lower().encode())
        negative = (negated == '!')
        value_type = Query._classify_value(valstr)
        match value_type:
            case 'normal':
                value = FValue(valstr.encode())
                interned_string = corpus.intern(feature, value)
                return [KnownLiteral(negative, offset, feature, interned_string, interned_string, corpus)]
            case 'prefix':
                valstr = valstr.split('.*')[0]
                value = FValue(valstr.encode())
                interned_range = corpus.interned_range(feature, value)
                return [KnownLiteral(negative, offset, feature, interned_range[0], interned_range[1], corpus)]
            case 'suffix':
                valstr = valstr.split('.*')[-1][::-1]
                value = FValue(valstr.encode())
                feature = Feature(feature + b'_rev')
                interned_range = corpus.interned_range(feature, value)
                return [KnownLiteral(negative, offset, feature, interned_range[0], interned_range[1], corpus)]
            case 'regex':
                regex_matches = corpus.get_matches(feature, valstr)
                return [KnownLiteral(negative, offset, feature, match, match, corpus) for match in regex_matches]
        
        raise ValueError(f"Unknown value type: {value_type!r}")
    
    @staticmethod
    def _tokenize_expression(expression: str) -> list[str]:
        """
        Tokenize an expression like this: ([pos="DT" word#"stor"] & [pos="JJ" | pos="DT"] | [pos="PN"]) [word="katt"]
        into a list of strings like this: ['(', '[pos="DT" word#"stor"]', '&', '[pos="JJ" | pos="DT"]', '|', '[pos="PN"]', ')', '[word="katt"]']
        """
        return re.findall(r'\[.*?\]|\(|\)|&|\|', expression)
    
    @staticmethod
    def _variable_name_generator():
        length = 1
        while True:
            for name in (''.join(chars) for chars in itertools.product(ascii_lowercase, repeat=length)):
                yield name
            length += 1
        
    @staticmethod
    def _distribute_expression(expr: Expression) -> Expression:
        """
        Recursively transforms an expression by distributing over Or nodes in And expressions.
        When distributing a factor, it renames the factor to include an index (e.g. d -> d1, d2, ...).
        It also sorts the factors in each And alphabetically by variable name.
        """
        # Base case: if literal, return as is.
        if expr.depth == 0:
            return expr

        # For And nodes: check if any child is an Or.
        if expr.NAME == 'And':
            # Look for the first Or child.
            for idx, subexpr in enumerate(expr.xs):
                if subexpr.depth > 0 and subexpr.NAME == 'Or':
                    # We found an Or child.
                    # Let 'others' be the factors that are not the Or child.
                    others = expr.xs[:idx] + expr.xs[idx+1:]
                    or_child = subexpr

                    new_terms = []
                    # For each disjunct in the Or, assign a unique index.
                    for i, disj in enumerate(or_child.xs, start=1):
                        # Rename each factor in 'others': if it is a literal, add the index.
                        new_factors = []
                        for factor in others:
                            if factor.depth == 0:
                                # For a literal like 'd', create a new variable 'd1', 'd2', etc.
                                base = str(factor)
                                new_factors.append(exprvar(f"{base}_{i}"))
                            else:
                                # Otherwise, leave it (or process it recursively if needed).
                                new_factors.append(Query._distribute_expression(factor))
                        
                        # Process the disjunct recursively in case it contains nested expressions.
                        processed_disj = Query._distribute_expression(disj)
                        
                        # If the disjunct is an And, flatten it.
                        if processed_disj.depth > 0 and processed_disj.NAME == 'And':
                            new_factors.extend(processed_disj.xs)
                        else:
                            new_factors.append(processed_disj)
                        
                        # Sort factors alphabetically by their string name.
                        new_factors_sorted = sorted(new_factors, key=lambda x: str(x))
                        
                        # Build the new And term.
                        new_terms.append(Query._distribute_expression(And(*new_factors_sorted)))
                    
                    # Return the Or of all the newly built And terms.
                    return Or(*new_terms)
            
            # If no Or child was found in this And, process all children recursively.
            return And(*(Query._distribute_expression(arg) for arg in expr.xs))
        
        # For Or nodes, simply process each child.
        if expr.NAME == 'Or':
            return Or(*(Query._distribute_expression(arg) for arg in expr.xs))
        
        # Fallback: return the expression unchanged.
        return expr

    @staticmethod
    def parse(corpus: Corpus, querystr: str, no_sentence_breaks: bool = False) -> 'Query':
        variable_names = Query._variable_name_generator()
        
        tokens = Query._tokenize_expression(querystr)
        # Split tokens like '[pos="DT" word#"stor"]' into ['(', '[pos="DT"]', '[word#"stor"]', ')']
        token_variables: map[str, KnownLiteral] = {}
        expressionString = ""
        offset = 0
        for _, token in enumerate(tokens):
            if token in ['(', ')', '&', '|']:
                expressionString += token
            else:
                if len(expressionString) > 0 and expressionString[-1] not in ['(', '&', '|']:
                    expressionString += '&'
                expressionString += '('
                for match in Query.token_regex.finditer(token):
                    separator, featstr, negated, valstr = match.groups()

                    negative = (negated == '!')

                    if len(expressionString) > 0 and expressionString[-1] not in ['(', '&', '|']:
                        separator = '&'
                    
                    if separator:
                        expressionString += separator
                    
                    if negative:
                        expressionString += "~"
                        
                    literals = Query._evalute_literal(corpus, offset, featstr, negated, valstr)
                    
                    if len(literals) > 1:
                        expressionString += '('
                    
                    for index, literal in enumerate(literals):
                        if index > 0:
                            expressionString += '|'
                        
                        # a + number of variables
                        variable = next(variable_names)
                        
                        # Add variable to list of variables
                        token_variables[variable] = literal
                        
                        expressionString += variable
                        
                    if len(literals) > 1:
                        expressionString += ')'
                offset += 1
                expressionString += ')'
                
        offset -= 1
        
        epr = expr(expressionString)

        expanded = Query._distribute_expression(epr)
        # And(d, Or(c, And(a, b))) ->
        # Or(And(c, d1), And(a, b, d2)) ->
        # And

        # Convert expression to CNF
        expanded = expanded.to_dnf()

        variables: dict[str, KnownLiteral] = {}
        if expanded.depth == 0:
            is_compliment = isinstance(expanded, Complement)
            term_name = expanded.inputs[0].name if is_compliment else expanded.name
            variable_name = term_name.split('_')[0]
            variables[term_name] = token_variables[variable_name].alter_offset(0).alter_negation(is_compliment)
        else:
            for index, term in enumerate(expanded.xs):
                if term.depth == 0:
                    variable_name = term.name[0]
                    variables[term.name] = token_variables[variable_name].alter_offset(index)
                else:
                    for subindex, subterm in enumerate(term.xs):
                        is_compliment = isinstance(subterm, Complement)
                        term_name = subterm.inputs[0].name if is_compliment else subterm.name
                        variable_name = term_name.split('_')[0]
                        variables[term_name] = token_variables[variable_name].alter_offset(subindex).alter_negation(is_compliment)

        cnf = expanded

        # Assert that the two expressions are equal
        #assert expanded.equivalent(cnf)  # -> True

        # Move singler variables into the or expressions
        # And(d, Or(a, b), Or(a, c)) -> 
        
        # Convert to function
        
        # Print the type of the expression
        print(type(cnf))
        
        query: list[QueryElement] = []
        
        try:
            if hasattr(cnf, 'xs'):
                for term in cnf.xs:
                    if isinstance(term, Complement):
                        query.append(variables[term.inputs[0].name])
                    if hasattr(term, 'xs'):                                   
                        literals = tuple(variables[value.inputs[0].name if isinstance(value, Complement) else value.name] for value in term.xs)
                        query.append(DisjunctiveGroup.create(literals))
                    else:
                        query.append(variables[term.name])
            else:
                query.append(variables[cnf.name])
        except Exception as e:
            raise ValueError(f"Error in query: {querystr!r}") from e
        
        #if not no_sentence_breaks:
        #    svalue = corpus.intern(SENTENCE, START)
        #    for soffset in range(1, offset):
        #        query.append(KnownLiteral(True, soffset, SENTENCE, svalue, svalue, corpus))
        
        if not query:
            raise ValueError(f"Found no matching query literals")
        
        assert not all(lit.negative for lit in query), "Cannot handle queries with only negative literals"
        
        return Query(corpus, query)

    @staticmethod
    def parse_old(corpus: Corpus, querystr: str, no_sentence_breaks: bool = False) -> 'Query':
        if not Query.fullq_regex.match(querystr):
            raise ValueError(f"Error in query: {querystr!r}")
        tokens = [tok.group() for tok in Query.query_regex.finditer(querystr)]
        query: list[QueryElement] = []
        for offset, token in enumerate(tokens):
            query_list: list[list[KnownLiteral]] = [[]]
            for match in Query.token_regex.finditer(token):
                separator, featstr, negated, valstr = match.groups()
                if separator == '|' and query_list[-1]:
                    query_list.append([])
                feature = Feature(featstr.lower().encode())
                negative = (negated == '!')
                value_type = Query._classify_value(valstr)
                match value_type:
                    case 'normal':
                        value = FValue(valstr.encode())
                        interned_string = corpus.intern(feature, value)
                        query_list[-1].append(KnownLiteral(negative, offset, feature, interned_string, interned_string, corpus))
                    case 'prefix':
                        valstr = valstr.split('.*')[0]
                        value = FValue(valstr.encode())
                        interned_range = corpus.interned_range(feature, value)
                        query_list[-1].append(KnownLiteral(negative, offset, feature, interned_range[0], interned_range[1], corpus))
                    case 'suffix':
                        valstr = valstr.split('.*')[-1][::-1]
                        value = FValue(valstr.encode())
                        feature = Feature(feature + b'_rev')
                        interned_range = corpus.interned_range(feature, value)
                        query_list[-1].append(KnownLiteral(negative, offset, feature, interned_range[0], interned_range[1], corpus))
                    case 'regex':
                        regex_matches = corpus.get_matches(feature, valstr)
                        regexed_literals = [KnownLiteral(negative, offset, feature, match, match, corpus) for match in regex_matches]
                        last_group = query_list.pop()
                        query_list.extend(last_group + [lit] for lit in regexed_literals)
            if len(query_list) > 1:
                query.extend(DisjunctiveGroup.create(literals) for literals in itertools.product(*query_list))
            elif query_list:
                query.extend(*query_list)
        if not no_sentence_breaks:
            svalue = corpus.intern(SENTENCE, START)
            for offset in range(1, len(tokens)):
                query.append(KnownLiteral(True, offset, SENTENCE, svalue, svalue, corpus))
        if not query:
            raise ValueError(f"Found no matching query literals")
        assert not all(lit.negative for lit in query), "Cannot handle queries with only negative literals"
        return Query(corpus, query)
