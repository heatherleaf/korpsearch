import os
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from corpus_reader import CoNLLReader


class MyTestCase(unittest.TestCase):
    files_to_delete: list[str] = []

    default_text = '''# sent_id = 1
# text = They buy and sell books.
1\tThey\tthey\tPRON\tPRP\tCase=Nom|Number=Plur\t2\tnsubj\t2:nsubj|4:nsubj\t_
2\tbuy\tbuy\tVERB\tVBP\tNumber=Plur|Person=3|Tense=Pres\t0\troot\t0:root\t_
3\tand\tand\tCCONJ\tCC\t_\t4\tcc\t4:cc\t_
4\tsell\tsell\tVERB\tVBP\tNumber=Plur|Person=3|Tense=Pres\t2\tconj\t0:root|2:conj\t_
5\tbooks\tbook\tNOUN\tNNS\tNumber=Plur\t2\tobj\t2:obj|4:obj\tSpaceAfter=No
6\t.\t.\tPUNCT\t.\t_\t2\tpunct\t2:punct\t_
'''

    text_with_empty_nodes = '''# sent_id = 2
1\tnosotros\tnosotros\t_\t_\t_\t_\t_\t_\t_
2\tvamos\tir\t_\t_\t_\t_\t_\t_\t_
3\ta\ta\t_\t_\t_\t_\t_\t_\t_
4\tel\tel\t_\t_\t_\t_\t_\t_\t_
5\tmar\tmar\t_\t_\t_\t_\t_\t_\t_
6\ty\ty\t_\t_\t_\t_\t_\t_\t_
7\tvosotros\tvosotros\t_\t_\t_\t_\t_\t_\t_
7.1\tvais\tir\t_\t_\t_\t_\t_\t_\t_
8\ta\ta\t_\t_\t_\t_\t_\t_\t_
9\tel\tel\t_\t_\t_\t_\t_\t_\t_
10\tparque\tparque\t\t_\t_\t_\t_\t_\t_\t_
'''

    text_with_multiwords = '''# sent_id = 3
1\tnosotros\tnosotros\t_\t_\t_\t_\t_\t_\t_
2\tvamos\tir\t_\t_\t_\t_\t_\t_\t_
3-4\tal\t_\t_\t_\t_\t_\t_\t_\t_
3\ta\ta\t_\t_\t_\t_\t_\t_\t_
4\tel\tel\t_\t_\t_\t_\t_\t_\t_
5\tmar\tmar\t_\t_\t_\t_\t_\t_\t_
6\ty\ty\t_\t_\t_\t_\t_\t_\t_
7\tvosotros\tvosotros\t_\t_\t_\t_\t_\t_\t_
8-9\tal\t_\t_\t_\t_\t_\t_\t_\t_
8\ta\ta\t_\t_\t_\t_\t_\t_\t_
9\tel\tel\t_\t_\t_\t_\t_\t_\t_
10\tparque\tparque\t_\t_\t_\t_\t_\t_\t_
'''

    conllu_plus_text = '''# global.columns = ID FORM UPOS HEAD DEPREL MISC PARSEME:MWE
# source_sent_id = conllu http://hdl.handle.net/11234/1-2837 UD_German-GSD/de_gsd-ud-train.conllu train-s1682
# sent_id = train-s1682
# text = Der CDU-Politiker strebt einen einheitlichen Wohnungsmarkt an, auf dem sich die Preise an der ortsüblichen Vergleichsmiete orientieren.
1\tDer\tDET\t2\tdet\t_\t*
2\tCDU\tPROPN\t4\tcompound\tSpaceAfter=No\t*
3\t-\tPUNCT\t2\tpunct\tSpaceAfter=No\t*
4\tPolitiker\tNOUN\t5\tnsubj\t_\t*
5\tstrebt\tVERB\t0\troot\t_\t2:VPC.full
6\teinen\tDET\t8\tdet\t_\t*
7\teinheitlichen\tADJ\t8\tamod\t_\t*
8\tWohnungsmarkt\tNOUN\t5\tobj\t_\t*
9\tan\tADP\t5\tcompound:prt\tSpaceAfter=No\t2
10\t,\tPUNCT\t5\tpunct\t_\t*
11\tauf\tADP\t12\tcase\t_\t*
12\tdem\tPRON\t20\tobl\t_\t*
13\tsich\tPRON\t20\tobj\t_\t1:IRV
14\tdie\tDET\t15\tdet\t_\t*
15\tPreise\tNOUN\t20\tnsubj\t_\t*
16\tan\tADP\t19\tcase\t_\t*
17\tder\tDET\t19\tdet\t_\t*
18\tortsüblichen\tADJ\t19\tamod\t_\t*
19\tVergleichsmiete\tNOUN\t20\tobl\t_\t*
20\torientieren\tVERB\t8\tacl\tSpaceAfter=No\t1
21\t.\tPUNCT\t5\tpunct\t_\t*
'''

    def reader_for(self, content: str) -> CoNLLReader:
        file = NamedTemporaryFile(suffix='.conllu', delete=False)
        self.files_to_delete.append(file.name)
        with open(file.name, 'w') as f:
            f.write(content)
        return CoNLLReader(Path(file.name), self._testMethodName)

    def tearDown(self) -> None:
        for file in self.files_to_delete:
            os.remove(file)
        self.files_to_delete.clear()

    def assert_correct_sentence_structure(self, reader: CoNLLReader, sentence_lengths: list[int]):
        sentences = list(reader.sentences())
        self.assertEqual(len(sentence_lengths), len(sentences),
                         msg=f'Reader should recognize {len(sentence_lengths)} sentences.')
        for sentence_id, (sentence, expected_length) in enumerate(zip(sentences, sentence_lengths)):
            self.assertEqual(expected_length, len(sentence),
                             msg=f'Reader should recognize {expected_length} tokens in sentence #{sentence_id}.')

    def test_conllu_default_header_columns(self):
        with self.reader_for('') as reader:
            self.assertEqual(reader.header, [str.encode(f) for f in CoNLLReader.DEFAULT_COLUMN_HEADERS])

    def test_conllu_reads_sentence(self):
        with self.reader_for(self.default_text) as reader:
            self.assert_correct_sentence_structure(reader, [6])

    def test_conllu_recognizes_sentence_boundaries(self):
        text = '\n'.join([self.default_text, self.default_text])
        with self.reader_for(text) as reader:
            self.assert_correct_sentence_structure(reader, [6, 6])

    def test_conllu_skips_empty_nodes(self):
        with self.reader_for(self.text_with_empty_nodes) as reader:
            self.assert_correct_sentence_structure(reader, [10])

    def test_conllu_skips_multiwords(self):
        with self.reader_for(self.text_with_multiwords) as reader:
            self.assert_correct_sentence_structure(reader, [10])

    def test_conllup_reads_sentence(self):
        actual_header = 'ID FORM UPOS HEAD DEPREL MISC PARSEME:MWE'
        with self.reader_for(self.conllu_plus_text) as reader:
            self.assertEqual([str.encode(c) for c in actual_header.split(' ')], reader.header,
                             msg=f'Reader should recognize header: {actual_header}')
            self.assert_correct_sentence_structure(reader, [21])


if __name__ == '__main__':
    unittest.main()
