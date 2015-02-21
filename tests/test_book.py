import unittest
import sys

sys.path.append('lib')
from book import BookInventory


class BookInventoryTests(unittest.TestCase):
    """
    Test case for BookInventory class.
    """

    def setUp(self):
        """
        Setup inventory that will be subjected to the tests.
        """
        self.inventory = BookInventory('./tests/test_title_author.tab.txt')

    def test_inventory_data_loading(self):
        """
        Test inventory loading with data using a light version of data file.
        """
        self.inventory.load_books()

    def test_book_count(self):
        """
        Test if all books in test file are loaded to the index.
        """
        self.inventory.load_books()
        self.assertEqual(self.inventory.books_count(), 10)


if __name__ == '__main__':
    unittest.main()