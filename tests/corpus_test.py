import os
import unittest
from unittest.mock import MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import List

from build_indexes import yield_templates
from corpus import Corpus
from index import Index
from index_builder import build_index
from query import Query
from search import search_corpus


class CorpusTest(unittest.TestCase):
    declared_corpus_features: List[str]
    declared_corpus_data: List[List[str]]

    root_dir: TemporaryDirectory[str]
    corpus_source_file: NamedTemporaryFile
    corpus_ready: bool

    corpus_name: str = 'corpus'

    def setUp(self):
        super().setUp()
        self.args = MagicMock()
        self.args.pivot_selector = 'first'
        self.args.cutoff = None
        self.args.filter = None

        self.declared_corpus_features = []
        self.declared_corpus_data = []
        self.root_dir = TemporaryDirectory()
        self.corpus_source_file = NamedTemporaryFile(suffix='.csv', delete=False)
        self.corpus_ready = False

    def tearDown(self):
        super().tearDown()
        self.root_dir.cleanup()
        os.remove(self.corpus_source_file.name)

    def corpus_features(self, *features: str):
        """
        Define which attributes are present in the corpus.
        """
        self.declared_corpus_features = [feat for feat in features]
        self.declared_corpus_data = [[] for _ in self.declared_corpus_features]

    def corpus_data(self, *data_columns: List[str]) -> int:
        """
        Add data to the corpus.

        Data is added as *f* lists, where *f* is the number of defined features.
        Every list must have the same length.

        Returns the corpus position to which the data was inserted.
        """
        self.assertNotEqual(len(self.declared_corpus_features), 0,
                            msg='Cannot add data to corpus without defined features.')

        number_of_features = len(self.declared_corpus_features)
        number_of_columns = len(data_columns)
        self.assertEqual(number_of_features, number_of_columns,
                         msg=f'Expected {number_of_features} columns of data but {number_of_columns} were provided')

        first_column_length = len(data_columns[0])
        inserted_at_position = len(self.declared_corpus_data[0])
        for index, attribute in enumerate(self.declared_corpus_features):
            row = data_columns[index]
            self.assertEqual(first_column_length, len(row),
                             msg=f'Attribute {attribute} expected {first_column_length} values but got {len(row)}')

            self.declared_corpus_data[index].extend(data_columns[index])
        return inserted_at_position

    def get_corpus(self) -> Corpus:
        return Corpus(Path(self.root_dir.name) / CorpusTest.corpus_name)

    def _prepare_corpus_if_not_prepared(self):
        if self.corpus_ready:
            return

        reversed_features = [f'{feature}_rev' for feature in self.declared_corpus_features]
        self.args.features = self.declared_corpus_features + reversed_features
        self.args.no_sentence_breaks = True
        self.args.max_dist = 0

        with self.corpus_source_file as source_file:
            self._write_source_file(source_file)

            dir_corpus = Path(self.root_dir.name) / (CorpusTest.corpus_name + Corpus.dir_suffix)
            dir_corpus.mkdir(exist_ok=True)
            Corpus.build(dir_corpus, Path(source_file.name))

            dir_index = Path(self.root_dir.name) / (CorpusTest.corpus_name + Index.dir_suffix)
            dir_index.mkdir(exist_ok=True)
            with self.get_corpus() as corpus:
                for template in yield_templates(corpus, self.args):
                    build_index(corpus, template, self.args)
        self.corpus_ready = True

    def _write_source_file(self, file):
        header = '\t'.join(self.declared_corpus_features) + '\n'
        file.write(header.encode())
        for token in range(len(self.declared_corpus_data[0])):
            attributes = range(len(self.declared_corpus_features))
            row = '\t'.join([self.declared_corpus_data[attribute][token] for attribute in attributes]) + '\n'
            file.write(row.encode())
        file.flush()

    def execute_test_query(self, corpus, query):
        query = Query.parse(corpus, query, self.args.no_sentence_breaks)
        return search_corpus(corpus, query, self.args)

    def assert_search_finds(self, query: str, needle: int, offset: int = 0):
        """
        Asserts that the query result set is exactly the provided *needle*.

        If an offset is provided, the result set must be *needle* + *offset*.
        """
        self.assert_search_finds_all(query, needle, offset=offset)

    def assert_search_finds_all(self, query, *needles: int, offset: int = 0):
        """
        Asserts that the query result set is exactly the set of provided *needles*.

        If an offset is provided it is added to every needle that must be included in the result set.
        """
        self._prepare_corpus_if_not_prepared()
        with self.get_corpus() as corpus:
            result = self.execute_test_query(corpus, query)
            self.assertEqual(len(result), len(needles),
                             msg=f'Query should find {len(needles)} result(s) but found {len(result)}.')
            for needle in needles:
                self.assertIn(needle + offset, result,
                              msg=f'Query failed to return expected token at position {needle + offset}.')

    def assert_search_has_attributes(self, query: str, **attributes):
        """
        Asserts that the given key-value-pairs match for every token in the result set.
        """
        self._prepare_corpus_if_not_prepared()
        with self.get_corpus() as corpus:
            result = self.execute_test_query(corpus, query)

            if len(result) == 0:
                self.fail('No results found for query.')

            for position, element in enumerate(result):
                for feature, expected_value in attributes.items():
                    strings = corpus.tokens[feature.encode()]
                    actual_value = strings.interned_string(strings[element])

                    self.assertEqual(expected_value, actual_value,
                                     msg=f'Feature {feature} on result #{position} should be {expected_value} but is {actual_value}.')
