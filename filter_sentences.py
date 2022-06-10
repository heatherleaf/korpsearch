
from disk import DiskStringArray, DiskIntArrayType, InternedString
from index import IndexSet
from query import Query
from typing import List, Tuple


def filter_sentences(index_set:IndexSet, query:Query):
    """Filter the index set *in place*"""

    sentences : List[int]
    if isinstance(index_set.values, list):
        assert index_set.start == 0
        assert index_set.size == len(index_set.values)
        sentences = index_set.values
    else:
        sentences = list(index_set)

    query_length : int = len(query.query)
    query_values : List[List[Tuple[int, InternedString]]] = []
    corpus_features : List[DiskStringArray] = []
    for feature, values in query.featured_query:
        query_values.append(values)
        corpus_features.append(query.corpus.words[feature])
    runfilter(sentences, query_length, query_values, corpus_features, query.corpus.sentence_pointers)

    index_set.values = sentences
    index_set.start = 0
    index_set.size = len(sentences)


def runfilter(
        sentences : List[int], 
        query_length : int, 
        query_values : List[List[Tuple[int, InternedString]]], 
        corpus_features : List[DiskStringArray],
        sentence_pointers : DiskIntArrayType,
    ):
    """Filter a list of sentence ids *in place*"""
    n_sentences : int = len(sentence_pointers)
    n_features : int = len(query_values)
    filtered : int = 0
    for sent in sentences:
        token_ptr : int = sentence_pointers[sent]
        stop : int = sentence_pointers[sent+1] if sent+1 < n_sentences else n_sentences
        stop -= query_length
        while token_ptr <= stop:
            for feat in range(n_features):
                if not all(corpus_features[feat][token_ptr+i] == val for i, val in query_values[feat]):
                    break
            else:
                sentences[filtered] = sent
                filtered += 1
                break
            token_ptr += 1
    del sentences[filtered:]

