import unittest

import numpy

from playground.sorts.selectionsort import SelectionSort


# SelectionSortTest is a subclass of unittest.TestCase, and it has one method, test_sort, which asserts
# that the sort method of SelectionSort returns a sorted array.
class SelectionSortTest(unittest.TestCase):
    def test_sort(self):
        """
        SelectionSortTest tests that the SelectionSort class properly sorts the array in ascending order.
        """
        array = numpy.array([5, 3, 1, 4, 2])
        numpy.testing.assert_array_equal(
            SelectionSort(array).sort(), numpy.array([1, 2, 3, 4, 5])
        )


if __name__ == "__main__":
    unittest.main()
