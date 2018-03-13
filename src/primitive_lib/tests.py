import unittest
from . import *

class TestEnumeration(unittest.TestCase):
    def test_label(self):
        "Tests whether we can turn a primitive into a valid label"
        #primitive_label_from_str("/")

    def test_list(self):
        print(list_primitives())

if __name__ == '__main__':
    unittest.main()