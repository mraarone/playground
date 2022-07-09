import unittest

import numpy
from playground.algorithms.xgboost import xgboost_example


# SelectionSortTest is a subclass of unittest.TestCase, and it has one method, test_sort, which asserts
# that the sort method of SelectionSort returns a sorted array.
class XGBoostExampleTest(unittest.TestCase):
    def test_xgboost_getdata(self):
        """
        XGBoost tests that the XGBoostTest class properly gets data.
        """
        raise(NotImplementedError)

    def test_xgboost_example_train(self):
        """
        XGBoost tests that the XGBoostTest class properly trains.
        """
        raise(NotImplementedError)

    def test_xgboost_example_predict(self):
        """
        XGBoost tests that the XGBoostTest class properly predicts.
        """
        raise(NotImplementedError)

if __name__ == "__main__":
    unittest.main()
