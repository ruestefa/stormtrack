#!/usr/bin/env python3

# Standard library
import unittest
import sys
from unittest import TestCase

# Third-party
import numpy as np

# First-party
from stormtrack.core.identification import pixels_find_boundaries
from stormtrack.core.typedefs import default_constants

# Local
from ...utils import assertBoundaries


class TestBoundaries_Base(TestCase):
    def assertBoundaries(self, shells1, shells2, holes1, holes2):
        assertBoundaries(self, shells1, shells2)
        assertBoundaries(self, holes1, holes2)


class FindBoundaries_NoHoles(TestBoundaries_Base):
    """Find the inner and outer boundaries of a feature."""

    def setUp(s):

        # No holes
        _, X = 0, 1
        # fmt: off
        s.fld = np.array(
            [ #  0 1 2 3 4  5
                [_,_,_,_,_, _], # 4
                [_,X,X,X,_, _], # 3
                [_,_,X,X,X, _], # 2
                [_,X,X,X,_, _], # 1
                [_,_,_,_,_, _], # 0
            ] #  0 1 2 3 4  5
        ).T[:, ::-1]
        # fmt: on
        s.nx, s.ny = s.fld.shape
        s.pixels = np.dstack(np.where(s.fld > 0))[0].astype(np.int32)

        # 4-connectivity
        # fmt: off
        s.shells4 = [
            np.array(
                [
                    (1, 3), (2, 3), (3, 3), (3, 2), (4, 2), (3, 2), (3, 1), (2, 1),
                    (1, 1), (2, 1), (2, 2), (2, 3), (1, 3),
                ]
            ),
        ]
        # fmt: on
        s.holes4 = []

        # 8-connectivity
        # fmt: off
        s.shells8 = [
            np.array(
                [
                    (1, 3), (2, 3), (3, 3), (4, 2), (3, 1), (2, 1), (1, 1), (2, 2),
                    (1, 3),
                ]
            ),
        ]
        # fmt: on
        s.holes8 = []

    def test_4c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=4)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells4, holes, s.holes4)

    def test_8c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells8, holes, s.holes8)


class FindBoundaries_OneHole_FarInside(TestBoundaries_Base):
    """Find the inner and outer boundaries of a feature."""

    def setUp(s):

        # One hole away from outer boundary
        _, X = 0, 1
        # fmt: off
        s.fld = np.array(
            [ #  0 1 2 3 4  5 6
                [_,X,_,_,_, _,_], # 8
                [_,X,_,_,_, X,_], # 7
                [_,X,X,X,X, X,X], # 6
                [_,X,X,X,X, X,X], # 5

                [_,X,X,_,_, X,X], # 4
                [_,X,X,X,X, X,X], # 3
                [_,X,X,X,X, X,_], # 2
                [_,_,X,X,X, _,_], # 1
                [_,_,_,_,_, _,_], # 0
            ] #  0 1 2 3 4  5 6
        ).T[:, ::-1]
        # fmt: on
        s.nx, s.ny = s.fld.shape
        s.pixels = np.dstack(np.where(s.fld > 0))[0].astype(np.int32)

        # 4-connectivity
        # fmt: off
        s.shells4 = [
            np.array(
                [
                    (1, 8), (1, 7), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (5, 7),
                    (5, 6), (6, 6), (6, 5), (6, 4), (6, 3), (5, 3), (5, 2), (4, 2),
                    (4, 1), (3, 1), (2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (1, 5),
                    (1, 6), (1, 7), (1, 8),
                ]
            ),
        ]
        s.holes4 = [
            np.array(
                [
                    (2, 5), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (4, 3), (3, 3),
                    (2, 3), (2, 4), (2, 5),
                ]
            ),
        ]
        # fmt: on

        # 8-connectivity
        # fmt: off
        s.shells8 = [
            np.array(
                [
                    (1, 8), (1, 7), (2, 6), (3, 6), (4, 6), (5, 7), (6, 6), (6, 5),
                    (6, 4), (6, 3), (5, 2), (4, 1), (3, 1), (2, 1), (1, 2), (1, 3),
                    (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
                ]
            ),
        ]
        s.holes8 = [
            np.array(
                [
                    (3, 5), (4, 5), (5, 4), (4, 3), (3, 3), (2, 4), (3, 5),
                ]
            ),
        ]
        # fmt: on

    def test_4c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=4)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells4, holes, s.holes4)

    def test_8c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells8, holes, s.holes8)


class FindBoundaries_OneHole_NearShell(TestBoundaries_Base):
    """Find the inner and outer boundaries of a feature."""

    def setUp(s):

        # One hole near outer boundary (4-connectivity)
        # No hole at all (8-connectivity)
        _, X = 0, 1
        # fmt: off
        s.fld = np.array(
            [ #  0 1 2 3 4  5 6
                [_,_,_,_,_, _,_], # 5

                [_,_,X,X,X, X,_], # 4
                [_,X,X,_,_, X,_], # 3
                [_,X,X,_,_, X,_], # 2
                [_,_,X,X,_, X,_], # 1
                [_,_,_,_,X, X,_], # 0
            ] #  0 1 2 3 4  5 6
        ).T[:, ::-1]
        # fmt: on
        s.nx, s.ny = s.fld.shape
        s.pixels = np.dstack(np.where(s.fld > 0))[0].astype(np.int32)

        # 4-connectivity
        # fmt: off
        s.shells4 = [
            np.array(
                [
                    (2, 4), (3, 4), (4, 4), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0),
                    (4, 0), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (4, 4), (3, 4),
                    (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), (2, 1), (2, 2), (1, 2),
                    (1, 3), (2, 3), (2, 4),
                ]
            ),
        ]
        # fmt: on
        s.holes4 = []

        # 8-connectivity
        # fmt: off
        s.shells8 = [
            np.array(
                [
                    (2, 4), (3, 4), (4, 4), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0),
                    (4, 0), (3, 1), (2, 1), (1, 2), (1, 3), (2, 4),
                ]
            ),
        ]
        s.holes8 = [
            np.array(
                [
                    (3, 4), (4, 4), (5, 3), (5, 2), (5, 1), (4, 0), (3, 1), (2, 2),
                    (2, 3), (3, 4),
                ]
            ),
        ]
        # fmt: on

    def test_4c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=4)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells4, holes, s.holes4)

    def test_8c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells8, holes, s.holes8)


class FindBoundaries_OneHole_Width1(TestBoundaries_Base):
    """Find the inner and outer boundaries of a feature."""

    def setUp(s):

        # Feature with hole of width 1
        # One shell and one hole
        # Shell identical to hole w/ 4-connectivity
        # Shell a superset of hole w/ 8-connectivity (minus outer edge pixels)
        _, X = 0, 1
        # fmt: off
        s.fld = np.array(
            [ #  0 1 2 3 4  5 6
                [_,_,_,_,_, _,_], # 5

                [_,X,X,X,X, X,_], # 4
                [_,X,_,_,_, X,_], # 3
                [_,X,_,X,X, X,_], # 2
                [_,X,X,X,_, _,_], # 1
                [_,_,_,_,_, _,_], # 0
            ] #  0 1 2 3 4  5 6
        ).T[:, ::-1]
        # fmt: on
        s.nx, s.ny = s.fld.shape
        s.pixels = np.dstack(np.where(s.fld > 0))[0].astype(np.int32)

        # 4-connectivity
        # fmt: off
        s.shells4 = [
            np.array(
                [
                    (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 3), (5, 2), (4, 2),
                    (3, 2), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 4),
                ]
            ),
        ]
        # fmt: on
        s.holes4 = [s.shells4[0].copy()]

        # 8-connectivity
        # fmt: off
        s.shells8 = [
            np.array(
                [
                    (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 3), (5, 2), (4, 2),
                    (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (1, 4),
                ]
            ),
        ]
        s.holes8 = [
            np.array(
                [
                    (2, 4), (3, 4), (4, 4), (5, 3), (4, 2), (3, 2), (2, 1), (1, 2),
                    (1, 3), (2, 4),
                ]
            ),
        ]
        # fmt: on

    def test_4c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=4)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells4, holes, s.holes4)

    def test_8c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells8, holes, s.holes8)


class FindBoundaries_ManyHoles(TestBoundaries_Base):
    """Find the inner and outer boundaries of a feature."""

    def setUp(s):

        # Multiple holes
        _, X = 0, 1
        # fmt: off
        s.fld = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_], # 17
                [_,_,X,X,X, _,_,_,_,_, _,_,_,_,_, _,_,_], # 16
                [_,_,_,X,_, X,_,_,_,_, _,_,_,_,X, _,_,_], # 15

                [_,_,_,X,_, X,X,X,_,_, _,_,_,X,X, X,X,_], # 14
                [_,_,_,X,X, X,X,X,_,X, _,_,_,X,X, X,_,_], # 13
                [_,_,_,_,X, X,X,X,X,X, _,_,X,X,X, X,_,_], # 12
                [_,_,_,X,X, X,_,_,X,X, _,_,X,X,X, _,X,_], # 11
                [_,_,_,X,X, X,_,_,X,X, X,_,X,_,_, _,X,X], # 10

                [_,_,_,_,X, X,_,_,X,X, X,X,X,_,_, _,X,X], #  9
                [_,_,_,_,X, X,X,_,X,X, X,X,_,_,_, _,X,X], #  8
                [_,_,_,_,X, X,X,_,X,X, X,_,_,_,_, X,X,X], #  7
                [_,_,_,X,X, _,X,X,_,X, _,_,_,_,_, X,X,X], #  6
                [_,_,X,X,X, X,X,X,_,X, X,X,X,X,X, X,X,_], #  5

                [X,X,X,X,X, X,X,X,X,X, _,_,X,X,_, _,X,_], #  4
                [_,_,X,X,_, _,_,X,X,_, _,_,_,X,X, _,X,_], #  3
                [_,_,_,X,_, _,_,X,_,_, _,_,_,_,X, X,_,_], #  2
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,X, X,_,_], #  1
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_], #  0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7
        ).T[:, ::-1]
        # fmt: on
        s.nx, s.ny = s.fld.shape
        s.pixels = np.dstack(np.where(s.fld > 0))[0].astype(np.int32)

        # 4-connectivity
        # fmt: off
        s.shells4 = [
            np.array(
                [
                    ( 2, 16), ( 3, 16), ( 4, 16), ( 3, 16), ( 3, 15), ( 3, 14),
                    ( 3, 13), ( 4, 13), ( 5, 13), ( 5, 14), ( 5, 15), ( 5, 14),
                    ( 6, 14), ( 7, 14), ( 7, 13), ( 7, 12), ( 8, 12), ( 9, 12),
                    ( 9, 13), ( 9, 12), ( 9, 11), ( 9, 10), (10, 10), (10,  9),
                    (11,  9), (12,  9), (12, 10), (12, 11), (12, 12), (13, 12),
                    (13, 13), (13, 14), (14, 14), (14, 15), (14, 14), (15, 14),
                    (16, 14), (15, 14), (15, 13), (15, 12), (14, 12), (14, 11),
                    (13, 11), (12, 11), (12, 10), (12,  9), (11,  9), (11,  8),
                    (10,  8), (10,  7), ( 9,  7), ( 9,  6), ( 9,  5), (10,  5),
                    (11,  5), (12,  5), (13,  5), (14,  5), (15,  5), (15,  6),
                    (15,  7), (16,  7), (16,  8), (16,  9), (16, 10), (16, 11),
                    (16, 10), (17, 10), (17,  9), (17,  8), (17,  7), (17,  6),
                    (16,  6), (16,  5), (16,  4), (16,  3), (16,  4), (16,  5),
                    (15,  5), (14,  5), (13,  5), (13,  4), (13,  3), (14,  3),
                    (14,  2), (15,  2), (15,  1), (14,  1), (14,  2), (14,  3),
                    (13,  3), (13,  4), (12,  4), (12,  5), (11,  5), (10,  5),
                    ( 9,  5), ( 9,  4), ( 8,  4), ( 8,  3), ( 7,  3), ( 7,  2),
                    ( 7,  3), ( 7,  4), ( 6,  4), ( 5,  4), ( 4,  4), ( 3,  4),
                    ( 3,  3), ( 3,  2), ( 3,  3), ( 2,  3), ( 2,  4), ( 1,  4),
                    ( 0,  4), ( 1,  4), ( 2,  4), ( 2,  5), ( 3,  5), ( 3,  6),
                    ( 4,  6), ( 4,  7), ( 4,  8), ( 4,  9), ( 4, 10), ( 3, 10),
                    ( 3, 11), ( 4, 11), ( 4, 12), ( 4, 13), ( 3, 13), ( 3, 14),
                    ( 3, 15), ( 3, 16), ( 2, 16),
                ]
            ),
        ]
        s.holes4 = [
            np.array(
                [
                    ( 5, 12), ( 6, 12), ( 7, 12), ( 8, 12), ( 8, 11), ( 8, 10),
                    ( 8,  9), ( 8,  8), ( 8,  7), ( 9,  7), ( 9,  6), ( 9,  5),
                    ( 9,  4), ( 8,  4), ( 7,  4), ( 7,  5), ( 7,  6), ( 6,  6),
                    ( 6,  7), ( 6,  8), ( 5,  8), ( 5,  9), ( 5, 10), ( 5, 11),
                    ( 5, 12),
                ]
            ),
            np.array(
                [
                    ( 4,  7), ( 5,  7), ( 6,  7), ( 6,  6), ( 6,  5), ( 5,  5),
                    ( 4,  5), ( 4,  6), ( 4,  7),
                ]
            ),
        ]
        # fmt: on

        # 8-connectivity
        # fmt: off
        s.shells8 = [
            np.array(
                [
                    ( 2, 16), ( 3, 16), ( 4, 16), ( 5, 15), ( 6, 14), ( 7, 14),
                    ( 7, 13), ( 8, 12), ( 9, 13), ( 9, 12), ( 9, 11), (10, 10),
                    (11,  9), (12, 10), (12, 11), (12, 12), (13, 13), (13, 14),
                    (14, 15), (15, 14), (16, 14), (15, 13), (15, 12), (16, 11),
                    (17, 10), (17,  9), (17,  8), (17,  7), (17,  6), (16,  5),
                    (16,  4), (16,  3), (15,  2), (15,  1), (14,  1), (14,  2),
                    (13,  3), (12,  4), (11,  5), (10,  5), ( 9,  4), ( 8,  3),
                    ( 7,  2), ( 7,  3), ( 6,  4), ( 5,  4), ( 4,  4), ( 3,  3),
                    ( 3,  2), ( 2,  3), ( 1,  4), ( 0,  4), ( 1,  4), ( 2,  5),
                    ( 3,  6), ( 4,  7), ( 4,  8), ( 4,  9), ( 3, 10), ( 3, 11),
                    ( 4, 12), ( 3, 13), ( 3, 14), ( 3, 15), ( 2, 16),
                ]
            ),
        ]
        s.holes8 = [
            np.array(
                [
                    (13, 11), (14, 11), (15, 12), (16, 11), (16, 10), (16,  9),
                    (16,  8), (15,  7), (15,  6), (14,  5), (13,  5), (12,  5),
                    (11,  5), (10,  5), ( 9,  6), (10,  7), (11,  8), (12,  9),
                    (12, 10), (13, 11),
                ]
            ),
            np.array(
                [
                    ( 6, 12), ( 7, 12), ( 8, 11), ( 8, 10), ( 8,  9), ( 8,  8),
                    ( 8,  7), ( 7,  6), ( 6,  7), ( 6,  8), ( 5,  9), ( 5, 10),
                    ( 5, 11), ( 6, 12),
                ]
            ),
            np.array(
                [
                    (14,  5), (15,  5), (16,  4), (16,  3), (15,  2), (14,  3),
                    (13,  4), (14,  5),
                ]
            ),
            np.array(
                [
                    ( 4, 16), ( 5, 15), ( 5, 14), ( 4, 13), ( 3, 14), ( 3, 15),
                    ( 4, 16),
                ]
            ),
            np.array(
                [
                    ( 8,  7), ( 9,  6), ( 9,  5), ( 8,  4), ( 7,  5), ( 7,  6),
                    ( 8,  7),
                ]
            ),
            np.array(
                [
                    ( 5,  7), ( 6,  6), ( 5,  5), ( 4,  6), ( 5,  7),
                ]
            ),
        ]
        # fmt: on

    def test_4c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=4)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells4, holes, s.holes4)

    def test_8c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        # ? holes = sorted(holes, key=lambda i: len(i), reverse=True)
        s.assertBoundaries(shells, s.shells8, holes, s.holes8)


if __name__ == "__main__":
    import logging as log

    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
    log.getLogger().setLevel(log.DEBUG)
    unittest.main()
