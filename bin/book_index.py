#!/usr/bin/python
"""Book search engine.

This module executes a simple search for books given a term-based query. The
query is capture from STDIN and executed on an indexed data structure
containing books' information. The result is displayed on STDOUT and outputs
the top 10 matches ordered by their tf*idf scores.

Example:
    $ python solution.py --data './data/title_author.tab.txt'

    book_index Loading books...
    book Loading books from file...
    search Start search engine (Indexing | Ranking)...
    search Vocabulary assembled with terms count 230885
    search Starting tf computation...
    search Starting tf-idf computation...
    search Starting tf-idf norm computation...
    search Building index...
    search Function = load_books, Time = 406.88 sec
    book_index Done loading books, 1284904 docs in index

    Enter a query, or hit enter to quit: Alys Eyre Macklin Greuze
    util Function = search_books, Time = 0.63 sec
    score: 32.6460707315, id: 1277695, title: Greuze, author: Alys Eyre Macklin
    score: 32.6460707315, id: 570698, title: Greuze, author: Alys Eyre Macklin
    ...

"""
import sys
import optparse
import logging
sys.path.append('lib')
import book

DEBUG = True

# Log initialization
log_level = logging.DEBUG if DEBUG else logging.INFO
log_format = '%(asctime)s - %(levelname)s - %(module)s : %(lineno)d - %(message)s'
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger(__name__)

CATALOG_FILENAME = 'data/min_title_author.tab.txt' if DEBUG else 'data/title_author.tab.txt'


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
    logger.info('Loading books...')

    repository.load_books()
    docs_number = repository.books_count()
    logger.info('Done loading books, %d docs in index', docs_number)

    while query is not '':
        query = raw_input('Enter a query, or hit enter to quit: ')
        search_results = repository.search_books(query)

        print search_results


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-d', '--data',
                      dest='data',
                      help='Location of the data file that will be indexed',
                      default=CATALOG_FILENAME)

    options, args = parser.parse_args()
    execute_search(options.data)
