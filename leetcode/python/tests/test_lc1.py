"""
Tests for LeetCode 1: Two Sum
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from problems.lc1 import twoSum


def test_two_sum_example1():
    assert twoSum([2, 7, 11, 15], 9) == [0, 1]


def test_two_sum_example2():
    assert twoSum([3, 2, 4], 6) == [1, 2]


def test_two_sum_example3():
    assert twoSum([3, 3], 6) == [0, 1]
