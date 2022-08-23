from unittest import TestCase
from fifo_doubles_list import FifoDoublesList


class TestFifoDoublesList(TestCase):

    def test_return_values(self):
        tested_instance = FifoDoublesList(6)
        # Remplir l'instance
        tested_instance.put(1)
        tested_instance.put(2)
        tested_instance.put(3)
        tested_instance.put(4)
        tested_instance.put(5)
        tested_instance.put(6)
        # Tester avec Assert
        self.assertEqual(6, tested_instance.size())
        self.assertEqual([1, 2, 3, 4, 5, 6], tested_instance.return_values())

        # move one case up
        tested_instance.put(7)
        self.assertEqual([2, 3, 4, 5, 6, 7], tested_instance.return_values())