"""
Generate queries for testing the performance of Korpsearch,
compared to other engines such as Corpus Workbench.

The generated queries should work in the 'wikipedia-sv' corpora,
for example corpus/wikipedia-sv-05M, or the full wikipedia-sv if you have access.
"""

from typing import Iterable, overload

groups = ["noun", "verb", "word"]

postags = {
    "noun": ["DT", "JJ", "NN"],
    "verb": ["VB", "PC"],
}

msdtags = {
    "noun": [("UTR", "NEU"), ("SIN", "PLU")],
    "verb": [("PRS", "PRT"), ("AKT", "SFO")],
}

prefixes = ["av", "in", "pre", "ordi"]  # size in wikipedia-sv: 3100k, 2100k,  400k,  11k
infixes  = ["ng", "gj", "stj", "aper"]  # size in wikipedia-sv: 6500k,  210k,   86k,  27k
suffixes = ["de", "are", "or", "erna"]  # size in wikipedia-sv: 4800k, 1650k, 1000k, 640k

equalities = [("=", "="), ("=", "!="), ("!=", "=")]

empty = "[]"
cstar = ".*"


def all_pairs(elems: Iterable[str]) -> Iterable[tuple[str, str]]:
    elems = list(elems)
    for a in elems:
        for b in elems:
            yield a, b

def all_triples(elems: Iterable[str]) -> Iterable[tuple[str, str, str]]:
    elems = list(elems)
    for a in elems:
        for b in elems:
            for c in elems:
                yield a, b, c

@overload
def get_fraction(pct: float, elems: Iterable[str]) -> Iterable[str]: ...
@overload
def get_fraction(pct: float, elems: Iterable[list[str]]) -> Iterable[list[str]]: ...
def get_fraction(pct: float, elems: Iterable[str|list[str]]) -> Iterable[str|list[str]]:
    ctr = 0.0
    for elem in elems:
        ctr += pct
        if ctr >= 1.0:
            ctr -= 1.0
            yield elem

def case_insensitive(tok: str) -> str:
    return "".join(f"[{c.upper()}{c}]" for c in tok)

def single_token(key: str, eq: str, val: str) -> str:
    return f'[{key}{eq}"{val}"]'


def double_token(key1: str, eq1: str, val1: str, key2: str, eq2: str, val2: str) -> str:
    return f'[{key1}{eq1}"{val1}" & {key2}{eq2}"{val2}"]'


def pos_msd_token(group: str, level: int) -> Iterable[str]:
    msdgroups = msdtags[group]
    # [msd=".*UTR.*"]
    yield from get_fraction(1, (
        single_token("msd", "=", cstar + msd + cstar)
        for mss in msdgroups for msd in mss
    ))
    # [msd=".*NEU.*PLU.*"]
    yield from get_fraction(1, (
        single_token("msd", "=", cstar + msd1 + cstar + msd2 + cstar)
        for mss1, mss2 in zip(msdgroups, msdgroups[1:])
        for msd1, msd2 in zip(mss1, mss2)
    ))
    if level <= 1: return
    # [pos?="VB" & msd=".*PRS.*"]
    yield from get_fraction(1/3, (
        double_token("pos", eq, pos, "msd", "=", cstar + msd + cstar)
        for pos in postags[group] for mss in msdgroups for msd in mss
        for eq in ["=", "!="]
    ))
    # [msd?=".*NEU.*" & msd=".*PLU.*"]
    yield from get_fraction(1, (
        double_token("msd", eq, cstar + msd1 + cstar, "msd", "=", cstar + msd2 + cstar)
        for mss1, mss2 in zip(msdgroups, msdgroups[1:])
        for msd1, msd2 in zip(mss1, mss2)
        for eq in ["=", "!="]
    ))


def pos_token() -> Iterable[str]:
    # [pos="NN"]
    yield from get_fraction(1, (
        single_token("pos", "=", pos)
        for group in postags for pos in postags[group]
    ))


def word_token(level: int) -> Iterable[str]:
    # [word="pre.*"]
    yield from get_fraction(1, (
        single_token("word", "=", pre + cstar)
        for pre in prefixes
    ))
    # [word=".*inf.*"]
    yield from get_fraction(1, (
        single_token("word", "=", cstar + inf + cstar)
        for inf in infixes
    ))
    # [word=".*suf"]
    yield from get_fraction(1, (
        single_token("word", "=", cstar + suf)
        for suf in suffixes
    ))
    if level <= 1: return
    # [word=".*[Ii][Nn][Ff].*"]
    yield from get_fraction(1, (
        single_token("word", "=", cstar + case_insensitive(inf) + cstar)
        for inf in infixes
    ))
    # [word="pre.*suf"]
    yield from get_fraction(1/3, (
        single_token("word", "=", pre + cstar + suf)
        for pre in prefixes for suf in suffixes
    ))
    # [word="pre.*inf.*"]
    yield from get_fraction(1/3, (
        single_token("word", "=", cstar + pre + cstar + inf + cstar)
        for pre in prefixes for inf in infixes
    ))
    if level <= 2: return
    # [word=".*inf.*inf.*"]
    yield from get_fraction(1, (
        single_token("word", "=", cstar + inf1 + cstar + inf2 + cstar)
        for inf1, inf2 in zip(infixes, infixes[1:])
    ))
    # [word="pre.*inf.*suf"]
    yield from get_fraction(1/3, (
        single_token("word", "=", pre + cstar + inf + cstar + suf)
        for pre in prefixes for inf in infixes for suf in suffixes
    ))
    # [word?=".*inf.*" & word=".*inf.*"]
    yield from get_fraction(1, (
        double_token("word", eq, cstar + inf1 + cstar, "word", "=", cstar + inf2 + cstar)
        for inf1, inf2 in zip(infixes, infixes[1:])
        for eq in ["=", "!="]
    ))


def query_token(group: str, level: int) -> Iterable[str]:
    if group == "word":
        yield from word_token(level)
    else:
        yield from pos_msd_token(group, level)


def pos_query() -> Iterable[list[str]]:
    yield from get_fraction(1, (q
        for tok in pos_token()
        for q in (
            [tok],
        )))
    yield from get_fraction(1, (q
        for tok1, tok2 in all_pairs(pos_token())
        for q in (
            [tok1, tok2],
            [tok1, empty, tok2],
            [tok1, empty, empty, tok2],
            [tok1, empty, empty, empty, tok2],
            ## Add these when we can handle repetition:
            # [tok1, empty+"+", tok2],
            # [tok1, empty+"?", tok2],
        )))
    yield from get_fraction(1/3, (q
        for tok1 in pos_token()
        for tok2 in word_token(1)
        for q in (
            [tok1, tok2],
            [tok2, tok1],
            [tok1, empty, tok2],
            [tok2, empty, tok1],
        )))
    yield from get_fraction(1/3, (q
        for tok1, tok2, tok3 in all_triples(pos_token())
        for q in (
            [tok1, tok2, tok3],
            [tok1, empty, tok2, tok3],
            [tok1, tok2, empty, tok3],
            [tok1, empty, tok2, empty, tok3],
            [tok1, empty, tok2, empty, tok3],
            ## Add these when we can handle repetition:
            # [tok1, empty+"+", tok2, tok3],
            # [tok1, empty+"?", tok2, tok3],
            # [tok1, tok2, empty+"+", tok3],
            # [tok1, tok2, empty+"?", tok3],
        )))


def simple_query(group: str) -> Iterable[list[str]]:
    yield from get_fraction(1, (q
        for tok in query_token(group, 3)
        for q in (
            [tok],
        )))
    yield from get_fraction(1/11, (q
        for tok1, tok2 in all_pairs(query_token(group, 2))
        for q in (
            [tok1, tok2],
            [tok1, empty, tok2],
        )))
    yield from get_fraction(1/31, (q
        for tok1, tok2, tok3 in all_triples(query_token(group, 1))
        for q in (
            [tok1, tok2, tok3],
            [tok1, empty, tok2, tok3],
            [tok1, tok2, empty, tok3],
            [tok1, empty, tok2, empty, tok3],
        )))


def metadata():
    cats = ["Politiska_partier_i_Norge"]
    for c in cats:
        yield [f'text.categories contains "{c}"']
    titles = ["partiet", "slutning"]
    for t in titles:
        yield [f'text.title = "{cstar}{t}{cstar}"']
        for c in cats:
            yield [f'text.title = "{cstar}{t}{cstar}" & text.categories contains "{c}"']
    dates = ["2016-12-31", "2018-10-30", "2018-11-01"]
    for d1, d2 in zip(dates, dates[1:]):
        yield [f'text.date < {d1}']
        yield [f'{d1} < text.date < {d2}']
        for t in titles:
            yield [f'text.date < {d1} & text.title = "{cstar}{t}{cstar}"']


def all_queries() -> list[str]:
    queries: list[list[str]] = []
    queries += get_fraction(1, pos_query())
    for g in groups:
        queries += simple_query(g)
    ## Add this when we can handle metadata:
    # queries += get_fraction(1/47, (q + ["::"] + m for q in queries for m in metadata()))
    # import random
    # random.seed(42)
    # random.shuffle(queries)
    return [" ".join(q) for q in queries]


def print_all_queries():
    for q in all_queries():
        print(q)


if __name__ == '__main__':
    print_all_queries()

