import unittest

import numpy

from playground.sorts.bubblesort import BubbleSort


# The BubbleSort class has a sort method that returns a sorted array.
class BubbleSortTest(unittest.TestCase):
    def test_sort(self):
        """
        BubbleSortTest tests that the SelectionSort class properly sorts the array in ascending order.
        """
        array = numpy.array([5, 3, 1, 4, 2])
        numpy.testing.assert_array_equal(
            BubbleSort(array).sort(), numpy.array([1, 2, 3, 4, 5])
        )


if __name__ == "__main__":
    unittest.main()
