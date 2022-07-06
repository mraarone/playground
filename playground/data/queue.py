import unittest

import numpy

class Queue:
    def __init__(self, values=[]):
        self.items = numpy.array(values)
        self.size = values.size
        
    def enqueue(self, value):
        self.items = numpy.append(self.items, value)
    
    def dequeue(self):
        if self.items.size == 0:
            return None
        else:
            return self.items[0]

    def is_empty(self):
        return self.items.size == 0

    def size(self):
        return self.items.size
    
