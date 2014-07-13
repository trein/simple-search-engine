# -*- coding: utf-8 -*-
import re
import unicodedata
import logging
from util import timed
from search import Indexable
from search import SearchEngine


class Book(Indexable):
    """Class encapsulating a specific behavior of indexed books.

    Args:
      iid (int): Identifier of indexable objects.
      title (str): Title of the book.
      author (str): Author of the book.
      metadata (str): Plain text with data to be indexed.

    Attributes:
      title (str): Title of the book.
      author (str): Author of the book.

    """

    def __init__(self, iid, title, author, metadata):
        Indexable.__init__(self, iid, metadata)
        self.title = title
        self.author = author

    def __repr__(self):
        return "id: %s, title: %s, author: %s" % \
               (self.iid, self.title, self.author)


class BookDataPreprocessor(object):
    """Preprocessor for book entries.

    """

    _EXTRA_SPACE_REGEX = re.compile(r'\s+', re.IGNORECASE)
    _SPECIAL_CHAR_REGEX = re.compile(
        # detect punctuation characters
        r'(?P<p>(\.+)|(\?+)|(!+)|(:+)|(;+)|'
        # detect special characters
        r'(\(+)|(\)+)|(\}+)|(\{+)|("+)|(-+)|(\[+)|(\]+)|'
        # detect commas NOT between numbers
        r'(?<!\d)(,+)(?!=\d)|(\$+))')

    def preprocess(self, entry):
        """Preprocess an entry to a sanitized format.

        The preprocess steps applied to the book entry is the following::
          1) All non-accents are removed;
          2) Special characters are replaced by whitespaces (i.e. -, [, etc.);
          3) Punctuation marks are removed;
          4) Additional whitespaces between replaced by only one whitespaces.

        Args:
          entry (str): Book entry in string format to be preprocess.

        Returns:
          str: Sanitized book entry.

        """
        f_entry = entry.lower()
        f_entry = f_entry.replace('\t', '|').strip()
        f_entry = self.strip_accents(unicode(f_entry, "utf-8"))
        f_entry = self._SPECIAL_CHAR_REGEX.sub(' ', f_entry)
        f_entry = self._EXTRA_SPACE_REGEX.sub(' ', f_entry)

        book_desc = f_entry.split('|')

        return book_desc

    def strip_accents(self, text):
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore')


class BookInventory(object):
    """Class representing a inventory of books.

    Args:
      filename (str): File name containing book inventory data.

    Attributes:
      filename (str): File name containing book inventory data.
      indexer (Indexer): Object responsible for indexing book inventory data.

    """

    _BOOK_META_ID_INDEX = 0
    _BOOK_META_TITLE_INDEX = 1
    _BOOK_META_AUTHOR_INDEX = 2
    _NO_RESULTS_MESSAGE = "Sorry, no results."

    def __init__(self, filename):
        self.filename = filename
        self.engine = SearchEngine()

    @timed
    def load_books(self):
        """Load books from a file name.

        This method leverages the iterable behavior of File objects
        that automatically uses buffered IO and memory management handling
        effectively large files.

        """
        logging.info("[Book] Loading books from file...")
        processor = BookDataPreprocessor()
        with open(self.filename) as catalog:
            for entry in catalog:
                book_desc = processor.preprocess(entry)
                metadata = ' '.join(book_desc[self._BOOK_META_TITLE_INDEX:])

                iid = book_desc[self._BOOK_META_ID_INDEX].strip()
                title = book_desc[self._BOOK_META_TITLE_INDEX].strip()
                author = book_desc[self._BOOK_META_AUTHOR_INDEX].strip()

                book = Book(iid, title, author, metadata)
                self.engine.add_object(book)

        self.engine.start()

    @timed
    def search_books(self, query, n_results=10):
        """Search books according to provided query of terms.

        The query is executed against the indexed books, and a list of books
        compatible with the provided terms is return along with their tf-idf
        score.

        Args:
          query (str): Query string with one or more terms.
          n_results (int): Desired number of results.

        Returns:
          list of IndexableResult: List containing books and their respective
            tf-idf scores.

        """
        result = ''
        if len(query) > 0:
            result = self.engine.search(query, n_results)

        if len(result) > 0:
            return "\n".join([str(indexable) for indexable in result])
        return self._NO_RESULTS_MESSAGE

    def books_count(self):
        """Return number of books already in the index.

        Returns:
          int: Number of books indexed.

        """
        return self.engine.count()