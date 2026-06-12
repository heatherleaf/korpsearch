
# This script runs a lot of test cases to make sure
# that the number of results are correct.
#
# Arguments:
#  - no argument: run tests
#  - "verbose": run tests with --verbose flag
#  - "debug": run tests with --verbose flag
#  - "build": build the corpus indexes


corpus=wikipedia-sv-05M

build=""
args=""

while [[ $# -gt 0 ]]; do
    case $1 in
        *build*)
            build="build"
            ;;
        -v|*verb*)
            args+=" --verbose"
            ;;
        -d|-vv|*debug*)
            args+=" --debug"
            ;;
        *)
            echo "Usage: $0 (build | verbose | debug)"
            exit
    esac
    shift
done

if [ "$build" = "build" ]; then
    python build_indexes.py --corpus $corpus --clean
    python build_indexes.py --corpus $corpus --sanity-check --corpus-index
    python build_indexes.py --corpus $corpus --sanity-check --features word pos --max-dist 0
    python build_indexes.py --corpus $corpus --sanity-check --templates pos:0+pos:1
    python build_indexes.py --corpus $corpus --sanity-check --templates pos:0+word:1 --sorter tmpfile
    python build_indexes.py --corpus $corpus --sanity-check --templates word:0+word:1 --min-frequency 10_000
    exit
fi

search() {
    gold="$1"
    shift
    out=`python search_cmdline.py --corpus $corpus --no-cache $args --num 0 --query "$@"`
    echo $out
    results=`echo $out | perl -ne 'print $1 if /^(\d+) search results/'`
    if [ "$results" == "$gold" ]
    then echo "--> ok  [$@]"
    else echo "--> MISMATCH:  $results =/= $gold  [$@]"
    fi
    echo
}

# Unary and binary indexes
search 177723 '[pos="DT"]'
search 453053 '[pos="VB"]'
search    131 '[pos="MAD"] [pos="VB"]'
search    131 '[pos="MAD"] [pos="VB"]' --no-binary
search   1638 '[pos="MAD"] [pos="VB"]' --no-sentence-breaks
search  56228 '[pos="DT"] [pos="JJ"] [pos="NN"]'
search  56228 '[pos="DT"] [pos="JJ"] [pos="NN"]' --no-binary

# Min-frequency binary index
search     15 '[word="en"] [word="en"]'
search     14 '[word="en"] [word="häst"]'

# Disjunction and conjunction
search 410190 '[pos="DT" | pos="JJ"]'
search 410190 '[pos="DT|JJ"]'
search     59 '[pos="DT"] [pos="JJ"] [word="hus|huset"]'
search     59 '[pos="DT"] [pos="JJ"] [word="hus" | word="huset"]'
search    321 '[pos="DT|JJ"] [word="hus|huset"]'
search    321 '[pos="DT" | pos="JJ"] [word="hus|huset"]'
search    321 '[pos="DT|JJ"] [word="hus" | word="huset"]'
search    321 '[pos="DT" | pos="JJ"] [word="hus" | word="huset"]'

# Prefix queries
search 211542 '[word="a.*"]'
search 328619 '[word="s.*"]'
search  23489 '[word="sa.*"]'
search   6202 '[word="en"] [word="a.*"]'
search  10779 '[pos="DT"] [word="a.*"]'
search  10779 '[pos="DT"] [word="a.*"]' --no-binary

# Regex queries
search  26542 '[word=".*ap.*"]'
search    245 '[word="a.*.*ap.*"]'
search    488 '[word="[aA].*.*[aA][pP].*"]'
search    698 '[word="en"] [word=".*ap.*"]'
search   1610 '[word="a.*"] [word=".*ap.*"]'
search   3144 '[word="a.*" & word=".*ap.*"]'
search 234940 '[word="a.*" | word=".*ap.*"]'

# Negative queries
search 103718 '[pos="DT"] [pos!="NN"]'
search 854772 '[pos!="DT"] [pos="NN"]'
search 102131 '[pos!="DT"] [pos="JJ"] [pos="NN"]'
search  11589 '[pos="DT"] [pos="JJ"] [pos!="NN"]'
search  26588 '[pos="DT"] [pos!="JJ"] [pos="NN"]'
search 209298 '[word="a.*"] [word!=".*ap.*"]'
search  22978 '[word!="a.*"] [word=".*ap.*"]'
search 208398 '[word="a.*" & word!=".*ap.*"]'
search  23398 '[word!="a.*" & word=".*ap.*"]'
