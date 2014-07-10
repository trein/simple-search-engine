# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy.sparse as sp
import logging
from collections import defaultdict


current_path = os.path.dirname(os.path.realpath(__file__))
STOP_WORDS_FILENAME = os.path.join(current_path, "stop_words.txt")


class Indexable(object):
    """Class representing an object that can be indexed.

    It is a general abstraction for indexable objects and can be used in
    different contexts.

    Args:
      iid (int): Identifier of indexable objects.
      metadata (str): Plain text with data to be indexed.

    Attributes:
      iid (int): Identifier of indexable objects.
      words_count (dict): Dictionary containing the unique words from
        `metadata` and their frequency.

    """

    def __init__(self, iid, metadata):
        self.iid = iid
        self.words_count = defaultdict(int)

        for word in metadata.split():
            self.words_count[word] += 1

    def __repr__(self):
        return " ".join(self.words_count.keys()[:10])

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def words_generator(self, stop_words):
        """Yield unique words extracted from indexed metadata.

        Args:
          stop_words (list of str): List with words that mus be filtered.

        Yields:
          str: Unique words from indexed metadata.

        """
        for word in self.words_count.keys():
            if word not in stop_words or len(word) > 5:
                yield word

    def count_for_word(self, word):
        """Frequency of a given word from indexed metadata.

        Args:
          word (str): Word whose the frequency will be retrieved.

        Returns:
          int: Number of occurrences of a given word.

        """
        return self.words_count[word] if word in self.words_count else 0


class IndexableResult(object):
    """Class representing a search result with a tf-idf score.

    Args:
      score (float): tf-idf score for the result.
      indexable (Indexable): Indexed object.

    Attributes:
      score (float): tf-idf score for the result.
      indexable (Indexable): Indexed object.

    """

    def __init__(self, score, indexable):
        self.score = score
        self.indexable = indexable

    def __repr__(self):
        return "score: %f, indexable: %s" % (self.score, self.indexable)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and abs(self.score - other.score) < 0.0001
                and self.indexable == other.indexable)

    def __ne__(self, other):
        return not self.__eq__(other)


class TfidfRank(object):
    """Class encapsulating tf-idf ranking logic.

    Tf-idf means stands for term-frequency times inverse document-frequency
    and it is a common term weighting scheme in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    Args:
      smoothing (int, optional): Smoothing parameter for tf-idf computation
        preventing by-zero divisions when a term does not occur in corpus.

      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    Attributes:
      smoothing (int, optional): Smoothing parameter for tf-idf computation
        preventing by-zero divisions when a term does not occur in corpus.

      stop_words (list of str): Stop words that will be filtered during docs
        processing.

      vocabulary (dict): Dictionary containing unique words of the corpus as
        keys and their respective global index used in tf-idf data structures.

      ft_matrix (matrix): Matrix containing the frequency term for each term
        in the corpus respecting the index stored in `vocabulary`.

      ifd_diag_matrix (list): Vector containing the inverse document
        frequency for each term in the corpus. It respects the index stored in
        `vocabulary`.

      tf_idf_matrix (matrix): Matrix containing the ft-idf score for each term
        in the corpus respecting the index stored in `vocabulary`.

    """

    def __init__(self, stop_words, smoothing=1):
        self.smoothing = smoothing
        self.stop_words = stop_words
        self.vocabulary = {}
        self.ft_matrix = []
        self.ifd_diag_matrix = []
        self.tf_idf_matrix = []

    def build_rank(self, objects):
        """Build tf-idf ranking score for terms in the corpus.

        Note:
          The code in this method could have been extracted to other smaller
          methods, improving legibility. This extraction has not been done so
          that its runtime complexity can be computed easily (the runtime
          complexity can be improved).

        Args:
          objects (list of Indexable): List of indexed objects that will be
            considered during tf-idf score computation.

        """
        self.__build_vocabulary(objects)

        n_terms = len(self.vocabulary)
        n_docs = len(objects)
        ft_matrix = sp.lil_matrix((n_docs, n_terms), dtype=np.float16)

        logging.info("[Ranking] Vocabulary assembled with terms count %s", n_terms)

        # compute idf
        logging.info("[Ranking] Starting tf computation...")
        for index, indexable in enumerate(objects):
            for word in indexable.words_generator(self.stop_words):
                word_index_in_vocabulary = self.vocabulary[word]
                doc_word_count = indexable.count_for_word(word)
                ft_matrix[index, word_index_in_vocabulary] = doc_word_count
        self.ft_matrix = ft_matrix.tocsc()

        logging.info("[Ranking] Starting tf-idf computation...")
        # compute idf with smoothing
        df = np.diff(self.ft_matrix.indptr) + self.smoothing
        n_docs_smooth = n_docs + self.smoothing

        # create diagonal matrix to be multiplied with ft
        idf = np.log(float(n_docs_smooth) / df) + 1.0
        self.ifd_diag_matrix = sp.spdiags(idf, diags=0, m=n_terms, n=n_terms)

        # compute tf-idf
        self.tf_idf_matrix = self.ft_matrix * self.ifd_diag_matrix
        self.tf_idf_matrix = self.tf_idf_matrix.tocsr()

        # compute td-idf normalization
        norm = self.tf_idf_matrix.tocsr(copy=True)
        norm.data **= 2
        norm = norm.sum(axis=1)
        n_nzeros = np.where(norm > 0)
        norm[n_nzeros] = 1.0 / np.sqrt(norm[n_nzeros])
        norm = np.array(norm).T[0]
        sp.sparsetools.csr_scale_rows(self.tf_idf_matrix.shape[0],
                                      self.tf_idf_matrix.shape[1],
                                      self.tf_idf_matrix.indptr,
                                      self.tf_idf_matrix.indices,
                                      self.tf_idf_matrix.data, norm)

    def __build_vocabulary(self, objects):
        """Build vocabulary with indexable objects.

        Args:
          objects (list of Indexable): Indexed objects that will be
            considered during ranking.

        """
        vocabulary_index = 0
        for indexable in objects:
          for word in indexable.words_generator(self.stop_words):
              if word not in self.vocabulary:
                  self.vocabulary[word] = vocabulary_index
                  vocabulary_index += 1

    def compute_rank(self, doc_index, terms):
        """Compute tf-idf score of an indexed document.

        Args:
          doc_index (int): Index of the document to be ranked.
          terms (list of str): List of query terms.

        Returns:
          float: tf-idf of document identified by its index.

        """
        score = 0
        for term in terms:
            term_index = self.vocabulary[term]
            score += self.tf_idf_matrix[doc_index, term_index]
        return score


class Index(object):
    """Class responsible for indexing objects.

    Note:
      In case of a indexer object, we dropped the runtime complexity for a
      search by increasing the space complexity. It is the traditional
      trade-off and here we are more interested in a lightening fast
      search then saving some space. This logic may have to be revisited if the
      index become too large.

    Args:
      term_index (dict): Dictionary containing a term as key and a list of all
        the documents that contain that key/term as values
      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    Attributes:
      term_index (dict): Dictionary containing a term as key and a list of all
        the documents that contain that key/term as values
      stop_words (list of str): Stop words that will be filtered during docs
        processing.

    """

    def __init__(self, stop_words):
        self.stop_words = stop_words
        self.term_index = defaultdict(list)

    def build_index(self, objects):
        """Build index the given indexable objects.

        Args:
          objects (list of Indexable): Indexed objects that will be
            considered during search.

        """
        for position, indexable in enumerate(objects):
          for word in indexable.words_generator(self.stop_words):
              # build dictionary where term is the key and an array
              # of the IDs of indexable object containing the term
              self.term_index[word].append(position)

    def search_terms(self, terms):
        """Search for terms in indexed documents.

        Args:
          terms (list of str): List of terms considered during the search.

        Returns:
          list of int: List containing the index of indexed objects that
            contains the query terms.

        """
        docs_indices = []
        for term_index, term in enumerate(terms):

            # keep only docs that contains all terms
            if term not in self.term_index:
                docs_indices = []
                break

            # compute intersection between results
            # there is room for improvements in this part of the code
            docs_with_term = self.term_index[term]
            if term_index == 0:
                docs_indices = docs_with_term
            else:
                docs_indices = set(docs_indices) & set(docs_with_term)
        return list(docs_indices)


class SearchEngine(object):
    """Search engine for objects that can be indexed.

    Attributes:
      objects (list of Indexable): List of objects that can be considered
        during search.
      stop_words (list of str): Stop words that will be filtered during docs
        processing.
      rank (TfidfRank): Object responsible for tf-idf ranking computation.
      index (Index): Object responsible for data indexing.

    """

    def __init__(self):
        self.objects = []
        self.stop_words = self.__load_stop_words()
        self.rank = TfidfRank(self.stop_words)
        self.index = Index(self.stop_words)

    def __load_stop_words(self):
        """Load stop words that will be filtered during docs processing.

        Stop words are words which are filtered out prior to
        processing of natural language data. There is not one definite
        list of stop words but we are using the list in `stop_words.txt` file.

        Returns:
          list str: List of English stop words.

        """
        stop_words = {}
        with open(STOP_WORDS_FILENAME) as stop_words_file:
            for word in stop_words_file:
                stop_words[word.strip()] = True
        return stop_words

    def add_object(self, indexable):
        """Add object to index.

        Args:
          indexable (Indexable): Object to be added to index.

        """
        self.objects.append(indexable)

    def start(self):
        """Perform search engine initialization.

        The current implementation initialize the ranking and indexing of
        added objects. The code below is not very efficient as it iterates over
        all indexed objects twice, but can be improved easily with generators.

        """
        logging.info("[Search] Start search engine (Indexing | Ranking)...")
        self.index.build_index(self.objects)
        self.rank.build_rank(self.objects)

    def search(self, query, n_results=10):
        """Return indexed documents given a query of terms.

        Assumptions:
          1) We assume all terms in the provided query have to be found.
          Otherwise, an empty list will be returned. It is a simple
          assumption that can be easily changed to consider any term.

          2) We do not use positional information of the query term. It is
          not difficult whatsoever to take it into account, but it was just a
          design choice since this requirement was not specified.

        Args:
          query (str): String containing one or more terms.
          n_results (int): Desired number of results.

        Returns:
          list of IndexableResult: List of search results including the indexed
            object and its respective tf-idf score.

        """
        terms = query.lower().split()
        docs_indices = self.index.search_terms(terms)
        search_results = []

        for doc_index in docs_indices:
            indexable = self.objects[doc_index]
            doc_score = self.rank.compute_rank(doc_index, terms)
            result = IndexableResult(doc_score, indexable)
            search_results.append(result)

        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:n_results]

    def count(self):
        """Return number of objects already in the index.

        Returns:
          int: Number of documents indexed.

        """
        return len(self.objects)