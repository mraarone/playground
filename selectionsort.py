import unittest

import numpy


# SelectionSort class
class SelectionSort:
    def __init__(self, array):
        """
        It initializes the array and the size of the array.

        :param array: The array to be sorted
        """
        self.array = array
        self.array_size = len(array)

    def sort(self):
        """
        For each element in the array, find the smallest element to the right of it and swap it with the
        current element
        :return: The sorted array
        """
        for i in range(self.array_size):
            min_index = i
            for j in range(i + 1, self.array_size):
                if self.array[j] < self.array[min_index]:
                    min_index = j
            self.array[i], self.array[min_index] = self.array[min_index], self.array[i]
        return self.array

    def print_array(self):
        """
        It prints the array.
        """
        print(self.array)


# SelectionSortTest is a subclass of unittest.TestCase, and it has one method, test_sort, which asserts
# that the sort method of SelectionSort returns a sorted array.
class SelectionSortTest(unittest.TestCase):
    def test_sort(self):
        """
        It tests that the SelectionSort class properly sorts the array in ascending order.
        """
        array = numpy.array([5, 3, 1, 4, 2])
        self.assertEqual(SelectionSort(array).sort(), [1, 2, 3, 4, 5])


def main():
    """
    It sorts the array in ascending order.
    """
    array = numpy.array([5, 3, 1, 4, 2])
    print(SelectionSort(array).sort())


if __name__ == "__main__":
    main()
