#!/usr/bin/env python3

# Standard library
import logging as log
import unittest
import sys
from unittest import TestCase

# Third-party
import numpy as np

# Local
from .utils import circle


# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)


class TestCircle(TestCase):
    def setUp(s):
        _ = 0
        s.fld_small = np.array(
            [
                [_, _, _, _, _, _, _, _, _],
                [_, _, _, _, 1, _, _, _, _],
                [_, _, 1, 1, 1, 1, 1, _, _],
                [_, _, 1, 1, 1, 1, 1, _, _],
                [_, 1, 1, 1, 1, 1, 1, 1, _],
                [_, _, 1, 1, 1, 1, 1, _, _],
                [_, _, 1, 1, 1, 1, 1, _, _],
                [_, _, _, _, 1, _, _, _, _],
                [_, _, _, _, _, _, _, _, _],
            ]
        ).T[:, ::-1]
        s.points_small = list(zip(*np.where(s.fld_small == 1)))
        s.shell_small = [
            (4, 7),
            (5, 6),
            (6, 6),
            (6, 5),
            (7, 4),
            (6, 3),
            (6, 2),
            (5, 2),
            (4, 1),
            (3, 2),
            (2, 2),
            (2, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (3, 6),
        ]

    def test_small_points(s):
        res = circle(4, 4, 3)
        s.assertSetEqual(set(res), set(s.points_small))

    def test_small_shell(s):
        shell = []
        _ = circle(4, 4, 3, shell)
        s.assertSetEqual(set(shell), set(s.shell_small))


if __name__ == "__main__":
    unittest.main()
