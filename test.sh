
python='pypy3'
# python='python3.10'
# corpus='corpora/bnc-01M'
corpus='corpora/bnc-mini'
templates='sentence:0 word:0 pos:0 word:0+pos:1 pos:0+pos:1'
# query='[pos="ART"][pos="ADJ"][pos="SUBST"]'
query='[pos="ART"][pos="ADJ"][pos="SUBST"][][pos="SUBST"]'

alias train="/usr/bin/time $python build_indexes.py $corpus --clean -c -t $templates -v" # --min-frequency 100"
alias test="/usr/bin/time $python search_corpus.py $corpus '$query' --filter -v"

echo "\n------------- Train: Key indexes ----------------"
train

echo "\n------------- Test: Key indexes -----------------"
test

echo "\n------------- Train: Suffix array ---------------"
train --suffix-array

echo "\n------------- Test: Suffix array ----------------"
test --suffix-array

echo "\n------------- Train: Suffix array (sorting) -----"
train --suffix-array --no-sqlite

echo "\n------------- Test: Suffix array (2) ----------------"
test --suffix-array

echo
