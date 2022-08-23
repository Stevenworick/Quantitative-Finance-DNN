from unittest import TestCase
from fifo_doubles_list import FifoDoublesList
from random import random
import numpy as np


# First test case (single value input)
class TestFifoDoublesArray(TestCase):
    def setUp(self) -> None:
        self.fifo_array = FifoDoublesList(10)


class TestFunctionsBasic(TestFifoDoublesArray):
    def test_size(self):
        self.assertEqual(10, self.fifo_array.size())

    def test_put(self):
        self.fifo_array.put(10)

    # Will have an error thrown because there is not enough data in the array.
    def test_get_mean(self):
        self.fifo_array.put(10)
        self.assertEqual(1.0, self.fifo_array.get_mean())


# Second test case (5 values out of 5 input)
class TestFifoDoublesArrayFilled(TestCase):
    def setUp(self) -> None:
        self.fifo_array = FifoDoublesList(5)
        self.fifo_array.put(10)
        self.fifo_array.put(10)
        self.fifo_array.put(10)
        self.fifo_array.put(10)
        self.fifo_array.put(10)


class TestFunctionsFilledArray(TestFifoDoublesArrayFilled):
    def test_size(self):
        self.assertEqual(5, self.fifo_array.size())

    # Will have an error thrown because there is not enough data in the array.
    def test_get_mean(self):
        self.assertEqual(10, self.fifo_array.get_mean())


# Second test case (5 values out of 5 input)
class TestFifoDoublesArrayFilled2(TestCase):
    def setUp(self) -> None:
        self.fifo_array = FifoDoublesList(5)
        self.fifo_array.put(10)
        self.fifo_array.put(10)
        self.fifo_array.put(10)
        self.fifo_array.put(10)
        self.fifo_array.put(10)

        self.fifo_array.put(3)
        self.fifo_array.put(3)
        self.fifo_array.put(3)
        self.fifo_array.put(3)
        self.fifo_array.put(3)


class TestFunctionsRewrittenArray(TestFifoDoublesArrayFilled2):
    def test_size(self):
        self.assertEqual(5, self.fifo_array.size())

    # Will have an error thrown because there is not enough data in the array.
    def test_get_mean(self):
        self.assertEqual(3, self.fifo_array.get_mean())


# Third test case (insert and test float accuracy)
class TestFifoDoublesArrayFilled3(TestCase):
    random_numbers = [random()*10 for i in range(5000)]

    def setUp(self) -> None:
        self.fifo_array = FifoDoublesList(5000)
        for random_number in self.random_numbers:
            self.fifo_array.put(random_number)


class TestMeanFunctionMean(TestFifoDoublesArrayFilled3):
    def test_size(self):
        self.assertEqual(len(self.random_numbers), self.fifo_array.size())

    # Check that the mean is properly calculated on the whole array.
    def test_get_mean(self):
        self.assertAlmostEqual(np.mean(self.random_numbers), self.fifo_array.get_mean(), delta=0.00001)
