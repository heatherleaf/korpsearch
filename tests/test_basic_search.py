import unittest

from tests.corpus_test import CorpusTest


class MyTestCase(CorpusTest):

    def test_search_one_attribute(self):
        self.corpus_features('word', 'pos')
        needle = self.corpus_data(
            ['This', 'is', 'a', 'test'],
            ['PRON', 'AUX', 'DET', 'NOUN']
        )

        for offset, word in enumerate(['This', 'is', 'a', 'test']):
            self.assert_search_finds(
                f'[word="{word}"]',
                needle,
                offset=offset
            )

    def test_search_all_attributes(self):
        self.corpus_features('a', 'b', 'c', 'd')
        self.corpus_data(
            ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8'],
            ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'],
            ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
            ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8'],
        )

        self.assert_search_has_attributes(
            '[a="a1"]',
            a='a1',
            b='b1',
            c='c1',
            d='d1'
        )
        self.assert_search_has_attributes(
            '[b="b3"]',
            a='a3',
            b='b3',
            c='c3',
            d='d3'
        )
        self.assert_search_has_attributes(
            '[c="c5"]',
            a='a5',
            b='b5',
            c='c5',
            d='d5'
        )
        self.assert_search_has_attributes(
            '[d="d7"]',
            a='a7',
            b='b7',
            c='c7',
            d='d7'
        )

    def test_search_finds_token_sequence(self):
        self.corpus_features('w')
        self.corpus_data(
            [str(i) for i in range(10)],
        )
        needle = self.corpus_data(
            ['needle' for _ in range(4)]
        )
        self.corpus_data(
            [str(i) for i in range(10)],
        )

        self.assert_search_finds(
            '[w="needle"] [w="needle"] [w="needle"] [w="needle"]',
            needle
        )

if __name__ == '__main__':
    unittest.main()
