
import re
import itertools
from typing import Literal
from collections.abc import Iterator, Sequence
from string import ascii_lowercase

from pyeda.inter import Expression, And, Or
from pyeda.boolalg.expr import exprvar, Complement

from expressions import evaluate_boolean_expr

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
        
        # TODO: I believe this can be removed now?
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

        if not groups:
            yield self
            return

        for group in groups:
            yield Query(self.corpus, singles + list(group))
        return

        # TODO: Look into why this was nessessary
        #for group in itertools.product(*groups):
        #    yield Query(self.corpus, singles + list(group))

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
        Recursively transforms an expression by distributing over Or nodes in And expressions, turning
        it into DNF.
        
        When distributing a factor they are renamed to include an index (e.g. d -> d_1, d_2, ...).
        This is to avoid some simplifications, as the position of the factor is important.
        """
        # Literal.
        if expr.depth == 0:
            return expr

        # And nodes.
        if expr.NAME == 'And':
            
            # Look for the first Or child.
            for idx, subexpr in enumerate(expr.xs):
                if subexpr.depth > 0 and subexpr.NAME == 'Or':
                    
                    # Extract all other children.
                    others = expr.xs[:idx] + expr.xs[idx+1:]
                    or_child = subexpr

                    new_terms = []
                    
                    # For each disjunct in the Or, assign a unique index.
                    for i, disj in enumerate(or_child.xs, start=1):
                        
                        # Produce new factors for each other child.
                        new_factors = []
                        
                        for factor in others:
                            # Literal.
                            if factor.depth == 0:
                                
                                # Rename the factor to include the index.
                                base = str(factor).replace('~', '')
                                variable = exprvar(f'{base}_{i}')
                                
                                if isinstance(factor, Complement):
                                    new_factors.append(~variable)
                                else:
                                    new_factors.append(variable)
                            else:
                                # Otherwise process it recursively.
                                new_factors.append(Query._distribute_expression(factor))
                        
                        # Process the disjunct recursively in case it contains nested expressions.
                        processed_disj = Query._distribute_expression(disj)
                        
                        # If the disjunct is an And, flatten it.
                        if processed_disj.depth > 0 and processed_disj.NAME == 'And':
                            new_factors.extend(processed_disj.xs)
                        else:
                            new_factors.append(processed_disj)
                        
                        # Sort factors alphabetically by their string name.
                        # (not sure if this is necessary, as it's also done in a later step)
                        new_factors_sorted = sorted(new_factors, key=lambda x: str(x).replace('~', ''))
                        
                        conjunction = And(*new_factors_sorted)
                        
                        # Build the new And term.
                        new_terms.append(Query._distribute_expression(conjunction))
                    
                    # Return the Or of all the newly built And terms.
                    return Or(*new_terms)
            
            # If no Or child was found in this And, process all children recursively.
            return And(*(Query._distribute_expression(arg) for arg in expr.xs))
        
        # Or nodes.
        if expr.NAME == 'Or':
            return Or(*(Query._distribute_expression(arg) for arg in expr.xs))
        
        # Add xor in the future?
        
        assert False, f"Unknown expression type: {expr!r}"

    @staticmethod
    def parse(corpus: Corpus, querystr: str, no_sentence_breaks: bool = False) -> 'Query':
        # Setup generator for variable names.
        variable_names = Query._variable_name_generator()
        
        # Tokenize the expression.
        tokens = Query._tokenize_expression(querystr)
        
        # Variables to keep track of the variables and their group index.
        token_variables: map[str, KnownLiteral] = {}    # a -> [word="cat"], b -> [word="dog"], ...
        literal_group: map[str, int] = {}               # a -> 0, b -> 1, ...
        
        # String to build the expression.
        expressionString = ""
        
        offset = 0 # Remove?
        
        for group_index, token in enumerate(tokens):
            # A group may be a single literal, or multiple in the case of [word!="cat" word!="dog"] for example.
            # A group represents 'where' in the query it is located, to keep track of the order of the literals.
            
            # Symbols.
            if token in ['(', ')', '&', '|']:
                expressionString += token
            else:
                # If the previous character was not a symbol, add an implicit AND.
                if len(expressionString) > 0 and expressionString[-1] not in ['(', '&', '|']:
                    expressionString += '&'
                
                expressionString += '('
                
                matches = list(Query.token_regex.finditer(token))
                
                for match in matches:
                    separator, featstr, negated, valstr = match.groups()

                    negative = (negated == '!')

                    # If the previous character was not a symbol, add an implicit AND.
                    if separator:
                        expressionString += separator
                    elif len(expressionString) > 0 and expressionString[-1] not in ['(', '&', '|']:
                        separator = '&'
                    
                    if negative:
                        expressionString += "!"
                    
                    # Evaluate the literal.
                    literals = Query._evalute_literal(corpus, offset, featstr, False, valstr)
                    
                    if len(literals) > 1:
                        expressionString += '('
                    
                    for index, literal in enumerate(literals):
                        if index > 0:
                            expressionString += '|'
                        
                        # Get the next variable name.
                        variable = next(variable_names)
                        
                        # Add variable to list of variables
                        token_variables[variable] = literal
                        literal_group[variable] = group_index
                        
                        expressionString += variable
                        
                    if len(literals) > 1:
                        expressionString += ')'
                
                # Handle the wildcard case.
                if len(matches) == 0:
                    variable = next(variable_names)
                    token_variables[variable] = None
                    literal_group[variable] = group_index
                    expressionString += variable
                
                offset += 1
                expressionString += ')'
                
        offset -= 1
        
        # Evaluate the expression to a pyeda expression.
        # Can likely get rid of pyeda and use custom classes, as this doesn't use much of anything from pyeda.
        epr = evaluate_boolean_expr(expressionString)

        # Distribute the expression to DNF.
        expanded = Query._distribute_expression(epr)
        
        # The variables to use in the query, with the correct offsets and negations.
        variables: dict[str, KnownLiteral] = {}

        # TODO: Simplify this part. There is likely some cases where the ordering is not correct.
        
        lowest_index = 1
        
        # Keep track of the lengths of the statements, to know where to add sentence breaks.
        statement_lengths = {}
        
        # A single literal.
        if expanded.depth == 0:
            is_compliment = isinstance(expanded, Complement)
            term_name = expanded.inputs[0].name if is_compliment else expanded.name
            variable_name = term_name.split('_')[0]
            variables[term_name] = token_variables[variable_name].alter_offset(0).alter_negation(is_compliment)
            lowest_index = 1
        else:
            lowest_index = 1
            
            previous_group_index = None
            index_to_use = 0
            
            # Sort the variables by where they appear in the expression, this is important.
            outer_xs = sorted(expanded.xs, key=lambda x: str(x).replace('~', ''))
            
            for index, term in enumerate(outer_xs):
                if term.depth == 0:
                    is_compliment = isinstance(term, Complement)
                    term_name = term.inputs[0].name if is_compliment else term.name
                    variable_name = term_name.split('_')[0]
                    
                    group_index = literal_group[variable_name]
                    
                    if previous_group_index is not None and group_index != previous_group_index:
                        index_to_use += 1
                    
                    previous_group_index = group_index
                    
                    if token_variables[variable_name] is not None:
                        variables[term_name] = token_variables[variable_name].alter_offset(index_to_use).alter_negation(is_compliment)
                    else:
                        variables[term_name] = None
                else:
                    previous_group_index = None
                    index_to_use = 0
                    
                    # Sort the variables by where they appear in the expression, this is important.
                    xs = sorted(term.xs, key=lambda x: str(x).replace('~', ''))
                    
                    for subterm in xs:
                        is_compliment = isinstance(subterm, Complement)
                        term_name = subterm.inputs[0].name if is_compliment else subterm.name
                        variable_name = term_name.split('_')[0]
                        
                        group_index = literal_group[variable_name]
                        
                        if previous_group_index is not None and group_index != previous_group_index:
                            index_to_use += 1
                        previous_group_index = group_index
                        
                        if token_variables[variable_name] is not None:
                            variables[term_name] = token_variables[variable_name].alter_offset(index_to_use).alter_negation(is_compliment)
                        else:
                            variables[term_name] = None
                        
                    inner_length = index_to_use
                    if inner_length < lowest_index:
                        lowest_index = inner_length
                    statement_lengths[index] = inner_length
                    
        if len(statement_lengths) == 0:
            statement_lengths[0] = index_to_use

        cnf = expanded

        # Convert the expression to a query.
        query: list[QueryElement] = []
        
        has_disjunction = False
        only_top_level_negation = True
        
        try:
            if hasattr(cnf, 'xs'):
                index = 0
                for term in cnf.xs:
                    if isinstance(term, Complement):
                        query.append(variables[term.inputs[0].name])
                    elif hasattr(term, 'xs'):                                   
                        literals = list(variables[value.inputs[0].name if isinstance(value, Complement) else value.name] for value in term.xs)
                        
                        # Filter out None values, the wildcard cases.
                        literals = [lit for lit in literals if lit is not None]
                        
                        # Add any extra sentence breaks if this statement is longer than the shortest.
                        statement_length = statement_lengths[index]
                        if statement_length > lowest_index and not no_sentence_breaks:
                            svalue = corpus.intern(SENTENCE, START)
                            for soffset in range(lowest_index + 1, statement_length):
                                query.append(KnownLiteral(True, soffset, SENTENCE, svalue, svalue, corpus))
                        
                        # Check if all literals are negative.
                        if any(not lit.negative for lit in literals):
                            only_top_level_negation = False
                        
                        query.append(DisjunctiveGroup.create(tuple(literals)))
                        
                        has_disjunction = True
                    else:
                        if variables[term.name] is not None:
                            query.append(variables[term.name])
                        only_top_level_negation = False
                    index += 1
            else:
                query.append(variables[cnf.name])
        except Exception as e:
            raise ValueError(f"Error in query: {querystr!r}") from e
        
        # Add the sentence breaks.
        if not no_sentence_breaks:
            svalue = corpus.intern(SENTENCE, START)
            for soffset in range(1, min(statement_lengths.values()) + 1):
                query.append(KnownLiteral(True, soffset, SENTENCE, svalue, svalue, corpus))
        
        if not query:
            raise ValueError(f"Found no matching query literals")
        
        # TODO: Is this sufficient?
        assert has_disjunction or not only_top_level_negation, "Cannot handle queries with only negative literals"
        
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
