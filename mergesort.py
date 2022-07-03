import unittest

import numpy


# We split the array in half, sort each half, and then merge the two sorted halves
class MergeSort:
    def __init__(self, array):
        self.array = array

    def sort(self):
        """
        We split the array in half, sort each half, and then merge the two sorted halves
        :return: The sorted array.
        """
        if len(self.array) > 1:
            mid = len(self.array) // 2
            left = self.array[:mid]
            right = self.array[mid:]

            left = MergeSort(left).sort()
            right = MergeSort(right).sort()
            return self.merge(left, right)
        return self.array

    def merge(self, left, right):
        """
        It takes two sorted lists and returns a single sorted list by comparing the elements one at a time

        :param left: the left half of the array
        :param right: the right half of the array
        :return: The result of the merge sort.
        """
        result = []

        while len(left) > 0 and len(right) > 0:
            if left[0] < right[0]:
                result.append(left[0])
                left = left[1:]
            else:
                result.append(right[0])
                right = right[1:]
        if len(left) > 0:
            result.extend(left)
        if len(right) > 0:
            result.extend(right)
        return result


# MergeSortTest is a subclass of unittest.TestCase, and it has one method, test_sort, which asserts
# that the sort method of MergeSort returns a sorted array.
class MergeSortTest(unittest.TestCase):
    def test_sort(self):
        """
        It tests that the MergeSort class properly sorts the array in ascending order.
        """
        array = numpy.array([5, 3, 1, 4, 2])
        self.assertEqual(MergeSort(array).sort(), [1, 2, 3, 4, 5])


def main():
    """
    It sorts the array in ascending order.
    """
    array = numpy.array([5, 3, 1, 4, 2])
    print(MergeSort(array).sort())


if __name__ == "__main__":
    main()
