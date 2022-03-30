
# Korp-search

Searching in very large corpora.

There are currently two variants of the search database:

- using the `shelve` module to store dictionaries of sets of sentences
- using a file-based hash table for the same thing

## Building indexes

The following builds search indexes for the BNC-mini corpus using the features `word`, `lemma`, and `pos`:

```
python korpsearch.py corpora/bnc-mini.csv --build-index --features word lemma pos
```

If you want to use the `shelve` module instead of the internal implementation, you can use `--use-shelf`.

## Searching

To search, you just provide a query as the second argument:

```
python korpsearch.py corpora/bnc-mini.csv '[pos="ART"] [lemma="small"] [pos="SUBST"]' 
```
