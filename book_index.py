#!/usr/bin/python
"""Book search engine.

This module executes a simple search for books given a term-based query. The
query is capture from STDIN and executed on an indexed data structure
containing books' information. The result is displayed on STDOUT and outputs
the top 10 matches ordered by their tf*idf scores.

Example:
    $ python solution.py --data "./data/title_author.tab.txt"

    [Main] Loading books...
    [Book] Loading books from file...
    [Search] Start search engine (Indexing | Ranking)...
    [Ranking] Vocabulary assembled with terms count 230885
    [Ranking] Starting tf computation...
    [Ranking] Starting tf-idf computation...
    [Ranking] Starting tf-idf norm computation...
    [Index] Building index...
    [Benchmark] Function = load_books, Time = 406.88 sec
    [Main] Done loading books, 1284904 docs in index

    Enter a query, or hit enter to quit: Alys Eyre Macklin Greuze
    [Benchmark] Function = search_books, Time = 0.63 sec
    score: 32.6460707315, id: 1277695, title: Greuze, author: Alys Eyre Macklin
    score: 32.6460707315, id: 570698, title: Greuze, author: Alys Eyre Macklin
    ...

"""
import optparse
import logging
from lib import book

DEBUG = False

# Log initialization
log_level = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

CATALOG_FILENAME = "./data/min_title_author.tab.txt" if DEBUG else "./data/title_author.tab.txt"
QUERY_INPUT_MESSAGE = "Enter a query, or hit enter to quit: "


def execute_search(data_location):
    """Capture query from STDIN and display the result on STDOUT.

    The query of terms is executed against an indexed data structure
    containing books' information. If not result is found, an warning message
    will notify the user of such situation.

    Args:
      data_location (str): Location of the data file that will be indexed.

    """
    query = None
    repository = book.BookInventory(data_location)
    logging.info("[Main] Loading books...")

    repository.load_books()
    docs_number = repository.books_count()
    logging.info("[Main] Done loading books, %d docs in index", docs_number)

    while query is not "":
        query = raw_input(QUERY_INPUT_MESSAGE)
        search_results = repository.search_books(query)

        print search_results


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-d', '--data',
                      dest="data",
                      help="Location of the data file that will be indexed",
                      default=CATALOG_FILENAME)

    options, args = parser.parse_args()
    execute_search(options.data)
