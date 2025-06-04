#!/bin/bash

d=private/bnc
corpus=bnc-$1
shift

t="pos:0 word:0 pos:0+pos:1 word:0+word:1"

wincent="../korpsearch-wincent"
python="/usr/bin/time python"
debug="--debug"
xtra="$debug --no-cache --no-sentence-breaks" # --no-binary"
search="$python search_cmdline.py -d $d $xtra -n 2"

# minfreq=""
minfreq="--min-freq 1000"

if [[ $1 == "build" ]]; then
    echo "-------------------------------------------------------------------------------------------------------------"
    (cd $wincent ; python build_indexes.py -d $d -c $corpus --clean -i --no-reverse --no-sentence-breaks)
    du -sh $wincent/$d/$corpus.corpus/feature*
    python build_indexes.py -d $d -c $corpus --clean -i --no-sentence-breaks
    du -sh $d/$corpus.corpus/feature*
    echo "-------------------------------------------------------------------------------------------------------------"
    (cd $wincent ; $python build_indexes.py -d $d -c $corpus.csv.gz --no-sentence-breaks --force -t $t $minfreq -s cython)
    $python build_indexes.py -d $d -c $corpus.csv.gz --no-sentence-breaks --force -t $t $minfreq -s roaring
    shift
fi

du -sh $wincent/$d/$corpus.indexes/*
du -sh $d/$corpus.indexes/*
echo

for query in "$@" ; do
    echo "------ $query ------"
    (cd $wincent ; $search -c $corpus  -q "$query")
    echo
    $search -c $corpus-cython  -q "$query"
    echo
    $search -c $corpus  -q "$query"
    echo
done
