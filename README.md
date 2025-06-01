
# Korp-search

Searching in very large corpora.

The basic idea is to use [inverted indexes](https://en.wikipedia.org/wiki/Inverted_index) to lookup queries.
The indexes are modifications of [suffix arrays](https://en.wikipedia.org/wiki/Suffix_array), but where we don't use the whole suffix -- so a "pruned" suffix array if you like.
We build two kinds of indexes -- unary (for looking up one single feature), and binary (for looking up a pair of adjacent features).
A complex search query is then translated to a conjunction of simple queries which use unary or binary indexes.
Then we calculate the intersection of all query results.
You can read more about the underlying technology in this paper:

Peter Ljunglöf, Nicholas Smallbone, Mijo Thoresson, and Victor Salomonsson (2024).
[Binary indexes for optimising corpus queries](https://aclanthology.org/2024.konvens-main.17/).
In *KONVENS 2024, 20th Conference on Natural Language Processing*, pages 149–158, Vienna, Austria.

## Dependencies

You need Python version 3.10 or later, but we recommend at least version 3.11 because it is so much faster than 3.10.

We use Roaring Bitmaps for storing the search indexes, and the search results.
Therefore you have to install the following library from PyPi:

- [pyroaring](https://pypi.org/project/pyroaring/)

In addition we recommend the following libraries, even though you should be able to run things without them:

- [tqdm](https://pypi.org/project/tqdm/) for a better progress bar

- [Cython](https://pypi.org/project/cython/) if you want to build indexes faster (see below)

- [FastAPI](https://pypi.org/project/fastapi/) if you want to run the web demo (see below)

## Building inverted indexes

Before you can start searching in a corpus you have to build the corpus index, and then some inverted indexes. If you want help, run:

    python3 build_indexes.py --help

The following builds the basic corpus index for the smallest version of the BNC corpus, in the directory `corpora/bnc-100k.corpus/`, which is a compact and efficient representation of the corpus:

    python3 build_indexes.py --corpus corpora/bnc-100k.csv --corpus-index

Now you can build inverted indexes for the corpus, for the features `word`, `lemma`, and `pos`. All inverted indexes are in the directory `corpora/bnc-100k.indexes/`:

    python3 build_indexes.py --corpus corpora/bnc-100k.csv --features word lemma pos --max-dist 2

`--max-dist` tells how many different binary indexes that will be created:
it's the maximum adjacent distance between the tokens in the query. The default setting is 2.
If you only want unary indexes you can set `--max-dist` to 0,
and if you want more control you can use the `--templates` option instead of `--features`.

The original `.csv` file is not used when searching, so you can remove it if you want
(but then you cannot build any new indexes, so it's probably a bad idea).

### Building indexes faster

Note that it can take quite some time to build indexes for large corpora.
But using [PyPy](https://www.pypy.org/) instead of CPython will be around twice as fast.

    pypy3 build_indexes.py --corpus ...

Most of the time spent when building indexes are for sorting them.
There are several possible sorting implementations you can test with (using the `--sorter` option):

- `roaring` (the default) collects Roaring Bitmaps while reading, and then writes them to disk.
  This is fast but slows down considerably for very large corpora (depending on your computer's internal memory).

- `tmpfile` uses a temporary file which is sorted.
  The main advantage is that it doesn't use up any internal memory, so it is useful for very large corpora.

- `internal` uses Python's builtin sort function, which is faster than using a `tmpfile`.
  However, it has to load the whole corpus in memory so it is not very useful for very large corpora
  (depending on your computer, but can cause problems from 10–100 million tokens).

- `cython` uses a Cython implementation using C's builtin `qsort` function for sorting the index.
  – this is also up to 5–10 times faster than `tmpfile`, but doesn't have any memory problems
  because it uses a temporary file.
  Note that you have to compile the Cython module first, by running `make faster-index-builder`.

## Searching from the command line

When you have to have built the inverted indexes as described above, you can search them.
To do this from the command line, you give the corpus and the query:

    python3 search_cmdline.py --corpus corpora/bnc-100k --query '[pos="ART"] [lemma="small"] [pos="SUBST"]'

All searches are cached (in the directory `cache/`), if you don't want to use the cached result, you can use `--no-cache`.
Use `--print json` for JSON output, `--start 42` to print from match 42, and `--num 14` to print at most 14 matches.
Use the following for more help:

    python3 search_cmdline.py --help

Note that this can take a couple of seconds for some very general searches on large corpora.
If you use the fast intersection (see below) some searches can be up to 10 times faster.

## Using the web demo locally

You can open the file `docs/webdemo.html` in your favourite browser to try out a very basic search interface.
This will use an API backend that is running on the domain `korpsearch.cse.chalmers.se`.

To run the web demo on your local builds you can change the first line in the file `docs/korpsearch.js` to:
```
const API_DOMAIN = "http://127.0.0.1:8000/";
```

To use the web demo locally you need to install [FastAPI](https://pypi.org/project/fastapi/).
Before running the demo you also have to build search indexes for the corpora you want to play with.
Then you can run the following to start the webserver:

    python search_fastapi.py [--corpus-dir DIR]

When the server has started you can open the URL <http://127.0.0.1:8000/docs/webdemo.html> to play around with the web demo.

There are some additional settings which you can see if you run `python search_fastapi.py --help`.
