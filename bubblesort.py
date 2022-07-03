import numpy
import unittest

# BubbleSort class that uses numpy arrays
class BubbleSort:
    def __init__(self, array):
        self.array = array
        self.array_size = len(array)

    # Bubble sort algorithm
    def sort(self):
        for _ in range(self.array_size):
            for _ in range(self.array_size - 1):
                if self.array[j] > self.array[j + 1]:
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]
        return self.array

    # Prints the array
    def print_array(self):
        print(self.array)

# The BubbleSort class has a sort method that returns a sorted array.
class BubbleSortTest(unittest.TestCase):
    def test_sort(self):
        array = numpy.array([5, 3, 1, 4, 2])
        self.assertEqual(BubbleSort(array).sort(), [1, 2, 3, 4, 5])

def main():
    # Generate random array
    array = numpy.random.randint(0, 100, 10)

    # Create BubbleSort object
    print(BubbleSort(array).sort())
