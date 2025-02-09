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

''' 
*** Output of Test Case Failures:

"FFF
======================================================================
FAIL: test_divide (__main__.TestCalculator)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test-calculator.py", line 23, in test_divide
    self.assertEqual(calculator.divide(10, 2), 4)
AssertionError: 5.0 != 4

======================================================================
FAIL: test_multiply (__main__.TestCalculator)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test-calculator.py", line 12, in test_multiply
    self.assertEqual(calculator.multiply(2, 3), 7)
AssertionError: 6 != 7

======================================================================
FAIL: test_sqrt (__main__.TestCalculator)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test-calculator.py", line 33, in test_sqrt
    self.assertEqual(calculator.sqrt(4), 3)
AssertionError: 2.0 != 3

----------------------------------------------------------------------
Ran 3 tests in 0.001s"

*** Output of Test Case Success (when failures are commented out):

"...
----------------------------------------------------------------------
Ran 3 tests in 0.000s

OK"

FAILED (failures=3)
'''
