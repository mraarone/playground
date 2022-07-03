import unittest

import numpy


# BubbleSort class that uses numpy arrays
class BubbleSort:
    def __init__(self, array):
        self.array = array
        self.array_size = len(array)

    # Bubble sort algorithm
    def sort(self):
        for _ in range(self.array_size):
            for j in range(self.array_size - 1):
                if self.array[j] > self.array[j + 1]:
                    self.array[j], self.array[j + 1] = self.array[j + 1], self.array[j]
        return self.array

    # Prints the array
    def print_array(self):
        print(self.array)


def run():
    print("bubblesort: running...")

    array = numpy.array([5, 3, 1, 4, 2])
    print(BubbleSort(array).sort())

    # # Generate random array
    # array = numpy.random.randint(0, 100, 10)

    # # Create BubbleSort object
    # print(BubbleSort(array).sort())


if __name__ == "__main__":
    run()
