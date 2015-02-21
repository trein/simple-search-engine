import unittest
import numpy as np
import sys

sys.path.append('lib')
from search import Index
from search import Indexable
from search import IndexableResult
from search import TfidfRank
from search import SearchEngine


def sample_stop_words():
    return ['a', 'the', 'this', 'is']


class SearchEngineTests(unittest.TestCase):
    """
    Test case for SearchEngine class.
    """

    def setUp(self):
        """
        Setup search engine that will be subjected to the tests.
        """
        self.engine = SearchEngine()

    def test_indexed_doc_count(self):
        """
        Test if the number of indexed object is retrieved correctly.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')
        self.build_sample_index([sample1, sample2, sample3])
        self.assertEqual(self.engine.count(), 3)

    def test_existent_term_search(self):
        """
        Test if search is correctly performed.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')
        self.build_sample_index([sample1, sample2, sample3])

        expected_results = [
            IndexableResult(1.414214, sample1),
            IndexableResult(0.906589, sample2),
            IndexableResult(0.906589, sample3),
        ]

        results = self.engine.search('indexable metadata')
        self.assertListEqual(results, expected_results)

    def test_non_existent_term_search(self):
        """
        Test if search is correctly performed.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')
        self.build_sample_index([sample1, sample2, sample3])

        expected_results = []

        results = self.engine.search('asdasdasdas')
        self.assertListEqual(results, expected_results)

    def test_search_result_limit(self):
        """
        Test if search results can be limited.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')
        self.build_sample_index([sample1, sample2, sample3])

        expected_results = [
            IndexableResult(1.414214, sample1),
        ]

        results = self.engine.search('indexable metadata', 1)
        self.assertListEqual(results, expected_results)

    def build_sample_index(self, objects):
        for indexable in objects:
            self.engine.add_object(indexable)
        self.engine.start()


class IndexTests(unittest.TestCase):
    """
    Test case for Index class.
    """

    def setUp(self):
        """
        Setup index that will be subjected to the tests.
        """
        self.index = Index(sample_stop_words())

    def test_sample_indexing_with_no_exceptions(self):
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')
        self.index.build_index([sample1, sample2, sample3])

    def test_invalid_term_search(self):
        """
        Test if the search returns when the term is not found.
        """
        sample1 = Indexable(1, 'this is an indexable simple metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')

        expected_indices = []

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(['not_valid_term'])
        self.assertItemsEqual(search_results, expected_indices)

    def test_mixed_valid_invalid_term_search(self):
        """
        Test if the search returns when there are valid and invalid terms mixed.
        """
        sample1 = Indexable(1, 'this is an indexable simple metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')

        expected_indices = []

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(['not_valid_term', 'super'])
        self.assertItemsEqual(search_results, expected_indices)

    def test_one_term_search(self):
        """
        Test if the search for one term returns expected results.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable super metadata')

        expected_indices = [1, 2]

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(['super'])
        self.assertItemsEqual(search_results, expected_indices)

    def test_stop_word_search(self):
        """
        Test if stop words are correctly ignored.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable super metadata')

        expected_indices = []

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(['this'])
        self.assertItemsEqual(search_results, expected_indices)

    def test_two_terms_search(self):
        """
        Test if the search for two term returns expected results.
        """
        sample1 = Indexable(1, 'this is an indexable simple metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable super metadata')

        expected_indices = [1, 2]

        self.index.build_index([sample1, sample2, sample3])
        search_results = self.index.search_terms(['indexable', 'super'])
        self.assertItemsEqual(search_results, expected_indices)


class TfidfRankTests(unittest.TestCase):
    """
    Test case for Index class.
    """

    def setUp(self):
        """
        Setup ranker that will be subjected to the tests.
        """
        self.rank = TfidfRank(sample_stop_words())

    def test_sample_ranking_with_no_exceptions(self):
        """
        Test if ranking is built without any exception.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')
        self.rank.build_rank([sample1, sample2, sample3])

    def test_doc_frequency_matrix_with_sample1(self):
        """
        Test if document frequency matrix is correctly built.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')
        self.rank.build_rank([sample1, sample2, sample3])

        expected_vocab_indices = {
            'an': 2, 'super': 3, 'indexable': 1, 'another': 4, 'metadata': 0
        }

        expected_tf = np.array([[1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 0, 0, 1]])

        self.assertEqual(self.rank.vocabulary, expected_vocab_indices)
        np.testing.assert_array_equal(self.rank.ft_matrix.todense(),
                                      expected_tf)

    def test_doc_frequency_matrix_with_sample2(self):
        """
        Test if document frequency matrix is correctly built.
        """
        sample1 = Indexable(1, 'the sky is blue')
        sample2 = Indexable(2, 'the sun is bright')
        self.rank.build_rank([sample1, sample2])

        expected_vocab_indices = {'blue': 0, 'sun': 2, 'bright': 3, 'sky': 1}

        expected_tf = np.array([[1, 1, 0, 0],
                                [0, 0, 1, 1]])

        self.assertEqual(self.rank.vocabulary, expected_vocab_indices)
        np.testing.assert_array_equal(self.rank.ft_matrix.todense(), expected_tf)

    def test_doc_inverse_term_frequency_vector1(self):
        """
        Test if document inverse term frequency vector is correctly built.
        """
        sample1 = Indexable(1, 'this is an indexable metadata')
        sample2 = Indexable(2, 'this is an indexable super metadata')
        sample3 = Indexable(3, 'this is another indexable metadata')
        self.rank.build_rank([sample1, sample2, sample3])

        expected_idf = [1.,  1., 1.28768207, 1.69314718, 1.69314718]
        expected_tf_idf = [[0.52284231, 0.52284231, 0.67325467, 0, 0],
                           [0.39148397, 0.39148397, 0.50410689, 0.66283998, 0],
                           [0.45329466, 0.45329466, 0, 0, 0.76749457]]

        np.testing.assert_almost_equal(
            self.rank.ifd_diag_matrix.diagonal(), expected_idf, 4)

        np.testing.assert_almost_equal(
            self.rank.tf_idf_matrix.todense(), expected_tf_idf, 4)

    def test_doc_inverse_term_frequency_vector2(self):
        """
        Test if document inverse term frequency vector is correctly built.
        """
        sample1 = Indexable(1, 'the sky is blue')
        sample2 = Indexable(2, 'the sun is bright')
        self.rank.build_rank([sample1, sample2])

        expected_idf = [1.40546511, 1.40546511, 1.40546511, 1.40546511]
        expected_tf_idf = [[0.70710678, 0.70710678, 0, 0],
                           [0, 0, 0.70710678, 0.70710678]]

        np.testing.assert_almost_equal(
            self.rank.ifd_diag_matrix.diagonal(), expected_idf, 4)

        np.testing.assert_almost_equal(
            self.rank.tf_idf_matrix.todense(), expected_tf_idf, 4)

    def test_score_computation(self):
        """
        Test if document score is correctly calculated.
        """
        sample1 = Indexable(1, 'the sky is blue')
        self.rank.build_rank([sample1])

        np.testing.assert_almost_equal(
            self.rank.compute_rank(0, ['blue']), 0.707106, 5)
        np.testing.assert_almost_equal(
            self.rank.compute_rank(0, ['sky']), 0.7071067, 5)
        np.testing.assert_almost_equal(
            self.rank.compute_rank(0, ['blue', 'sky']), 1.414213, 5)


if __name__ == '__main__':
    unittest.main()