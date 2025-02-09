import unittest
import calculator

class TestCalculator(unittest.TestCase):
    def test_multiply(self):
        self.assertEqual(calculator.multiply(3, 4), 12)
        self.assertEqual(calculator.multiply(3, -4), -12)
        self.assertEqual(calculator.multiply(0, 4), 0)
        self.assertEqual(calculator.multiply(0, -29), 0)
        self.assertEqual(calculator.multiply(100, 2), 200)
        # Example of test case failure
        #self.assertEqual(calculator.multiply(2, 3), 7)

    def test_divide(self):
        self.assertEqual(calculator.divide(10, 2), 5)
        self.assertEqual(calculator.divide(10, -2), -5)
        self.assertEqual(calculator.divide(0, 2), 0)
        self.assertEqual(calculator.divide(0, -29), 0)
        self.assertEqual(calculator.divide(100, 2), 50)
        with self.assertRaises(ValueError):
            calculator.divide(5, 0)
        # Example of test case failure
        #self.assertEqual(calculator.divide(10, 2), 4)

    def test_sqrt(self):
        self.assertEqual(calculator.sqrt(4), 2)
        self.assertEqual(calculator.sqrt(9), 3)
        self.assertEqual(calculator.sqrt(16), 4)
        self.assertEqual(calculator.sqrt(25), 5)
        with self.assertRaises(ValueError):
            calculator.sqrt(-1)
        # Example of test case failure
        #self.assertEqual(calculator.sqrt(4), 3)

if __name__ == '__main__':
    unittest.main()
