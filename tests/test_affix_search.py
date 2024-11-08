import unittest

from tests.corpus_test import CorpusTest
from unittest import skip

class MyTestCase(CorpusTest):

    def test_search_one_attribute(self):
        self.corpus_features('word')
        self.corpus_data(
            ['aba', 'abba', 'abbba', 'b', 'abbbba']
        )

        self.assert_search_finds(
            '[word="a.*"] [word="b"]',
            2
        )

    def test_search_prefix_regex(self):
        self.corpus_features('word')
        self.corpus_data(
            ['a', 'aa', 'aaa', 'ba', 'b']
        )

        self.assert_search_finds_all(
            '[word=".*a"]',
            0, 1, 2, 3
        )

    def test_search_suffix_regex(self):
        self.corpus_features('word')
        self.corpus_data(
            ['a', 'aa', 'aaa', 'ab', 'b']
        )

        self.assert_search_finds_all(
            '[word="a.*"]',
            0, 1, 2, 3
        )

    def test_search_infix_regex(self):
        self.corpus_features('word')
        self.corpus_data(
            ['aa', 'aaa', 'aba', 'ab', 'b']
        )

        self.assert_search_finds_all(
            '[word="a.*a"]',
            0, 1, 2
        )

    @skip('Matching any value using regex is not yet supported.')
    def test_search_matchall_regex(self):
        self.corpus_features('word')
        self.corpus_data(
            ['a', 'b', 'c', 'd', 'e']
        )

        self.assert_search_finds_all(
            '[word=".*"]',
            0, 1, 2, 3, 4
        )

    def test_query_with_two_regex(self):
        self.corpus_features('word')
        self.corpus_data(
            ['ma', 'de', 'de', 'ax', 'bca', 'da', 'a', 'a', 'exas']
        )

        self.assert_search_finds_all(
            '[word="d.*"] [word="a.*"]',
            2, 5
        )

if __name__ == '__main__':
    unittest.main()
