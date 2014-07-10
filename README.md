[![Build Status](https://travis-ci.org/trein/simple-search-engine.png?branch=master)](https://travis-ci.org/trein/simple-search-engine)

# Simple Search Engine

### Definitions
- `tf*idf` – term frequency times inverse document frequency – read more [here](http://en.wikipedia.org/wiki/Tf%E2%80%93idf);
- `query` - a search query which can have multiple words / terms;
- `term` – a single word in a query.

### Goal
Given a set of Author and Title data from several books, implement a `tf*idf` index in memory which does the following:
- Read English text data into the index;
- For a given `query`, output the top 10 results ranked by their `tf*idf` scores.

### Data
`title_author.tab.txt` – This file contains most of the titles and authors of a book catalog. Due to its size, it has to be download separately [here](https://www.dropbox.com/s/kxo1c5yoqzcxtly/title_author.tab.txt.zip).

This is a line taken from the tab-delimited title_author.tab.txt.gz: `800 The plays Oscar Wilde`

The columns are `id`, `title`, `author`, so in this case:
`id -> 800 title -> The plays author -> Oscar Wilde`

### Sample Output
Your solution should output something similar to the following, but does not need to be exactly the same:

```
$ python solution.py
Creating index ...
Done creating index, 1284903 docs in index
Enter a query, or hit enter to quit
Alys Eyre Macklin Greuze
score: 32.6460707315, id: 1277695, title: Greuze, author: Alys Eyre Macklin
score: 32.6460707315, id: 570698, title: Greuze, author: Alys Eyre Macklin
score: 32.6460707315, id: 39325, title: Greuze, author: Alys Eyre Macklin
score: 19.9661713056, id: 642628, title: Twenty-nine tales from the French, author: Alys Eyre Macklin
score: 17.9820399437, id: 350719, title: A Plain Statement of Facts, Relative to Sir Eyre Coote: Containing the ..., author: William Bagwell, Sir Eyre
Coote
score: 12.679899426, id: 417681, title: Greuze and his models, author: John Rivers
score: 12.2744343179, id: 1229365, title: Chareles Macklin, author: Edward Abbott Parry
score: 12.2744343179, id: 1006303, title: Charles Macklin, author: Parry, Edward Abbott, Sir
score: 12.2744343179, id: 723442, title: Captain Macklin : his memoirs, author: Richard Harding Davis,Walter Appleton Clark
score: 12.2744343179, id: 539572, title: Charles Macklin, author: Parry, Edward Abbott, Sir
Enter a query, or hit enter to quit
```

### Code
You may write your solution in either Python, Java, C#, C, or C++. If you need to make any assumptions in your code, clearly document them in the comments. On startup, your code should read the given data file, then prompt the user for queries in a loop (reading from stdin), outputting the search results in a reasonable text format.

### Solution

The proposed problems were solved using Python `v2.7.5` the following libraries:

- numpy `v1.6.2`: (matrices operations and other utilities)
- scipy `v0.11.0` (sparse matrix data structures)

The current implementation follows [Google Style Python]
(http://google-styleguide.googlecode.com/svn/trunk/pyguide.html).

#### Contents:
 - `lib/search.py`: Module containing search implementation
 - `lib/book.py`: Module containing search abstraction for the context of books
 - `tests/test_search.py`: Module containing search unit tests
 - `tests/test_book.py`: Module containing books search unit tests
 - `solution.py`: Command line interface for books search

#### Running the application
    $ python solution.py --data "./data/title_author.tab.txt"

#### Running the unit tests
    $ python tests/test_search.py
    $ python tests/test_book.py

#### Comments
The current implementation proposes a general framework for indexing and ranking documents. The classes `SearchEngine`, `Index`, `TfidfRank`, `Indexable` and `IndexableResult` are not limited to the context of books and can be used in other applications.

A simple benchmark was performed to evaluate some results:

##### Parameters
- Machine: Intel i5 dual-core 2.5GHz 8GB RAM
- Dataset: 1284904 documents and 230885 terms in vocabulary

##### Results
- Index building time: 406.88 sec (6.8 min)
- Memory usage: around 3.0GB
- Average time of query search: 0.5 sec

##### Console output
```
2014-03-12 13:18:21,096 INFO [Main] Loading books...
2014-03-12 13:18:21,096 INFO [Book] Loading books from file...
2014-03-12 13:19:07,219 INFO [Search] Start search engine (Indexing | Ranking)...
2014-03-12 13:19:15,145 INFO [Ranking] Vocabulary assembled with terms count 230885
2014-03-12 13:19:15,145 INFO [Ranking] Starting tf computation...
2014-03-12 13:20:15,647 INFO [Ranking] Starting tf-idf computation...
2014-03-12 13:20:16,429 INFO [Ranking] Starting tf-idf norm computation...
2014-03-12 13:20:16,429 INFO [Index] Building index...
2014-03-12 13:20:16,429 INFO [Benchmark] Function = load_books, Time = 406.88 sec
2014-03-12 13:20:16,429 INFO [Main] Done loading books, 1284904 docs in index
Enter a query, or hit enter to quit:
```