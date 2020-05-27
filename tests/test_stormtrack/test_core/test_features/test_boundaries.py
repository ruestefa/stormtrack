#!/usr/bin/env python3

# Standard library
import logging as log
import unittest
import sys
from unittest import TestCase

# Third-party
import numpy as np
import pytest

# First-party
from stormtrack.core.constants import default_constants
from stormtrack.core.identification import pixels_find_boundaries

# Local
from ...utils import assertBoundaries


# SR_DBG <
# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)
# SR_DBG >


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
            [ #  0 1 2 3 4 5 6
                [_,X,_,_,_,_,_], # 8
                [_,X,_,_,_,X,_], # 7
                [_,X,X,X,X,X,X], # 6
                [_,X,X,X,X,X,X], # 5
                [_,X,X,_,_,X,X], # 4
                [_,X,X,X,X,X,X], # 3
                [_,X,X,X,X,X,_], # 2
                [_,_,X,X,X,_,_], # 1
                [_,_,_,_,_,_,_], # 0
            ] #  0 1 2 3 4 5 6
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


class FindBoundaries_NestedShells(TestBoundaries_Base):
    """Find the inner and outer boundaries of a nested feature."""

    def setUp(s):

        # Multiple holes
        _, X = 0, 1
        # fmt: off
        s.fld = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7
                [_,_,_,_,_, _,_,_,_,_, X,X,X,X,_, _,_,_], #  9
                [_,_,_,_,_, _,X,X,X,X, X,X,X,X,X, X,_,_], #  8
                [_,_,_,_,X, X,X,_,_,_, _,_,_,_,X, X,_,_], #  7
                [_,_,_,X,X, _,_,_,_,_, _,_,_,_,_, X,_,_], #  6
                [_,X,X,X,_, _,X,X,X,X, X,X,_,_,X, X,X,_], #  5

                [_,X,X,_,_, X,X,X,X,X, X,_,_,_,X, X,X,_], #  4
                [_,X,X,_,_, _,_,_,_,_, _,_,_,_,X, X,X,_], #  3
                [_,X,X,X,_, _,_,_,_,_, _,_,_,X,X, X,X,_], #  2
                [_,_,X,X,X, X,X,X,X,X, X,X,X,X,X, X,_,_], #  1
                [_,_,_,X,X, X,X,X,X,X, X,X,X,X,X, _,_,_], #  0
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
                    (10,  9), (11,  9), (12,  9), (13,  9), (13,  8), (14,  8),
                    (15,  8), (15,  7), (15,  6), (15,  5), (16,  5), (16,  4),
                    (16,  3), (16,  2), (15,  2), (15,  1), (14,  1), (14,  0),
                    (13,  0), (12,  0), (11,  0), (10,  0), ( 9,  0), ( 8,  0),
                    ( 7,  0), ( 6,  0), ( 5,  0), ( 4,  0), ( 3,  0), ( 3,  1),
                    ( 2,  1), ( 2,  2), ( 1,  2), ( 1,  3), ( 1,  4), ( 1,  5),
                    ( 2,  5), ( 3,  5), ( 3,  6), ( 4,  6), ( 4,  7), ( 5,  7),
                    ( 6,  7), ( 6,  8), ( 7,  8), ( 8,  8), ( 9,  8), (10,  8),
                    (10,  9),
                ]
            ),
            np.array(
                [
                    ( 5,  4), ( 6,  4), ( 6,  5), ( 7,  5), ( 8,  5), ( 9,  5),
                    (10,  5), (11,  5), (10,  5), (10,  4), ( 9,  4), ( 8,  4),
                    ( 7,  4), ( 6,  4), ( 5,  4),
                ]
            ),
        ]
        s.holes4 = [
            np.array(
                [
                    ( 6,  8), ( 6,  7), ( 5,  7), ( 4,  7), ( 4,  6), ( 3,  6),
                    ( 3,  5), ( 2,  5), ( 2,  4), ( 2,  3), ( 2,  2), ( 3,  2),
                    ( 3,  1), ( 4,  1), ( 5,  1), ( 6,  1), ( 7,  1), ( 8,  1),
                    ( 9,  1), (10,  1), (11,  1), (12,  1), (13,  1), (13,  2),
                    (14,  2), (14,  3), (14,  4), (14,  5), (15,  5), (15,  6),
                    (15,  7), (14,  7), (14,  8), (13,  8), (12,  8), (11,  8),
                    (10,  8), ( 9,  8), ( 8,  8), ( 7,  8), ( 6,  8),
                ]
            ),
        ]
        # fmt: on

        # 8-connectivity
        # fmt: off
        s.shells8 = [
            np.array(
                [
                    (10,  9), (11,  9), (12,  9), (13,  9), (14,  8), (15,  8),
                    (15,  7), (15,  6), (16,  5), (16,  4), (16,  3), (16,  2),
                    (15,  1), (14,  0), (13,  0), (12,  0), (11,  0), (10,  0),
                    ( 9,  0), ( 8,  0), ( 7,  0), ( 6,  0), ( 5,  0), ( 4,  0),
                    ( 3,  0), ( 2,  1), ( 1,  2), ( 1,  3), ( 1,  4), ( 1,  5),
                    ( 2,  5), ( 3,  6), ( 4,  7), ( 5,  7), ( 6,  8), ( 7,  8),
                    ( 8,  8), ( 9,  8), (10,  9),
                ]
            ),
            np.array(
                [
                    ( 5,  4), ( 6,  5), ( 7,  5), ( 8,  5), ( 9,  5), (10,  5),
                    (11,  5), (10,  4), ( 9,  4), ( 8,  4), ( 7,  4), ( 6,  4),
                    ( 5,  4),
                ]
            ),
        ]
        s.holes8 = [
            np.array(
                [
                    ( 7,  8), ( 6,  7), ( 5,  7), ( 4,  6), ( 3,  5), ( 2,  4),
                    ( 2,  3), ( 3,  2), ( 4,  1), ( 5,  1), ( 6,  1), ( 7,  1),
                    ( 8,  1), ( 9,  1), (10,  1), (11,  1), (12,  1), (13,  2),
                    (14,  3), (14,  4), (14,  5), (15,  6), (14,  7), (13,  8),
                    (12,  8), (11,  8), (10,  8), ( 9,  8), ( 8,  8), ( 7,  8),
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


# @pytest.mark.skip("WIP")
class FindBoundaries_RealCase(TestBoundaries_Base):
    def setUp(s):

        # The crash that triggered the introduction of this test only occurred
        # for a specific order of the pixels; if they were sorted or shuffled,
        # there was no error. Therefore, the pixels are specified directly in
        # order, not derived from a field as for the previous tests. Also, this
        # triggered the introduction of the `~_shuffle` tests to ensure that the
        # boundaries do not depend of the order of the feature pixels.

        # Define feature pixels (must be in this order, as described above)
        s.nx, s.ny = 100, 100
        # fmt: off
        s.pixels = np.array(
            [
                 (92, 89), (90, 90), (80, 95), (59, 75), (60, 75), (60, 76), (60, 77),
                 (60, 78), (60, 79), (60, 80), (61, 75), (61, 76), (61, 77), (61, 78),
                 (61, 79), (61, 80), (61, 81), (61, 82), (61, 83), (62, 76), (62, 77),
                 (62, 78), (62, 79), (62, 80), (62, 81), (62, 82), (62, 83), (62, 84),
                 (62, 85), (63, 77), (63, 78), (63, 79), (63, 80), (63, 81), (63, 82),
                 (63, 83), (63, 84), (63, 85), (63, 86), (63, 87), (64, 78), (64, 79),
                 (64, 80), (64, 81), (64, 82), (64, 83), (64, 84), (64, 85), (64, 86),
                 (64, 87), (64, 88), (65, 80), (65, 81), (65, 82), (65, 83), (65, 84),
                 (65, 85), (65, 86), (65, 87), (65, 88), (65, 89), (65, 90), (66, 81),
                 (66, 82), (66, 83), (66, 84), (66, 85), (66, 86), (66, 87), (66, 88),
                 (66, 89), (66, 90), (66, 91), (67, 81), (67, 82), (67, 83), (67, 84),
                 (67, 85), (67, 86), (67, 87), (67, 88), (67, 89), (67, 90), (68, 81),
                 (68, 82), (68, 83), (68, 84), (68, 85), (68, 86), (68, 87), (68, 88),
                 (68, 89), (69, 81), (69, 82), (69, 83), (69, 84), (69, 85), (69, 86),
                 (69, 87), (69, 88), (70, 81), (70, 82), (70, 83), (70, 84), (70, 85),
                 (70, 86), (70, 87), (70, 88), (71, 82), (71, 83), (71, 84), (71, 85),
                 (71, 86), (71, 87), (72, 82), (72, 83), (72, 84), (72, 85), (72, 86),
                 (72, 87), (72, 88), (72, 89), (73, 82), (73, 83), (73, 84), (73, 85),
                 (73, 86), (73, 87), (73, 88), (73, 89), (73, 90), (73, 91), (74, 83),
                 (74, 84), (74, 85), (74, 86), (74, 87), (74, 88), (74, 89), (74, 91),
                 (74, 92), (75, 83), (75, 84), (75, 85), (75, 86), (75, 87), (75, 88),
                 (75, 89), (76, 84), (76, 85), (76, 86), (76, 87), (76, 88), (76, 89),
                 (77, 85), (77, 86), (77, 87), (77, 88), (77, 89), (77, 90), (77, 91),
                 (76, 92), (77, 92), (77, 93), (76, 94), (76, 95), (76, 96), (76, 97),
                 (77, 94), (77, 95), (77, 96), (77, 97), (77, 98), (78, 86), (78, 87),
                 (78, 88), (78, 89), (78, 90), (78, 91), (78, 92), (78, 93), (78, 96),
                 (79, 87), (79, 88), (79, 89), (79, 90), (79, 91), (79, 92), (79, 93),
                 (79, 96), (79, 97), (80, 88), (80, 89), (80, 90), (80, 91), (80, 92),
                 (80, 96), (80, 97), (81, 89), (81, 90), (81, 91), (81, 96), (81, 97),
                 (82, 89), (82, 90), (82, 91), (82, 97), (83, 90), (83, 91), (83, 92),
                 (82, 93), (83, 93), (83, 97), (83, 98), (84, 90), (84, 91), (84, 92),
                 (84, 93), (84, 94), (84, 97), (84, 98), (85, 91), (85, 92), (85, 93),
                 (85, 94), (85, 97), (85, 98), (86, 91), (86, 92), (86, 93), (86, 94),
                 (86, 95), (86, 96), (86, 97), (86, 98), (87, 91), (87, 92), (87, 93),
                 (87, 94), (87, 95), (87, 96), (87, 97), (88, 91), (88, 92), (88, 93),
                 (88, 94), (88, 95), (88, 97), (89, 91), (89, 92), (89, 93), (89, 94),
                 (89, 95), (90, 91), (90, 92), (90, 93), (90, 94), (90, 95), (91, 90),
                 (91, 91), (91, 92), (91, 93), (91, 94), (91, 95), (92, 90), (92, 91),
                 (92, 92), (92, 93), (92, 94), (92, 95), (93, 89), (93, 90), (93, 91),
                 (93, 92), (93, 93), (93, 94), (93, 95), (94, 89), (94, 90), (94, 91),
                 (94, 92), (95, 89), (95, 91), (95, 92), (96, 89),
            ],
            np.int32,
        )

        # Independently define the same feature again, but based on a grid array
        # This only serves as a visual aid during debugging/development!
        _, X = 0, 1
        # fmt: off
        arr = np.array(
            [ # 50 1 2 3 4 5 6 7 8 9 60 1 2 3 4 5 6 7 8 9 70 1 2 3 4 5 6 7 8 9 80 1 2 3 4 5 6 7 8 9 90 1 2 3 4 5 6 7 8 9
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 99
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,X,_,_, _,_,_,X,X,X,X,_,_,_, _,_,_,_,_,_,_,_,_,_], # 98
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,X,X,_,X, X,X,X,X,X,X,X,X,X,_, _,_,_,_,_,_,_,_,_,_], # 97
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,X,X,X,X, X,X,_,_,_,_,X,X,_,_, _,_,_,_,_,_,_,_,_,_], # 96
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,X,X,_,_, X,_,_,_,_,_,X,X,X,X, X,X,X,X,_,_,_,_,_,_], # 95
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,X,X,_,_, _,_,_,_,X,X,X,X,X,X, X,X,X,X,_,_,_,_,_,_], # 94
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,X,X,X, _,_,X,X,X,X,X,X,X,X, X,X,X,X,_,_,_,_,_,_], # 93
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,X,_,X,X,X,X, X,_,_,X,X,X,X,X,X,X, X,X,X,X,X,X,_,_,_,_], # 92
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,X,_,_,_, _,_,_,X,X,_,_,X,X,X, X,X,X,X,X,X,X,X,X,X, X,X,X,X,X,X,_,_,_,_], # 91
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,X,X,X,_,_, _,_,_,X,_,_,_,X,X,X, X,X,X,X,X,_,_,_,_,_, X,X,X,X,X,_,_,_,_,_], # 90
              # 50 1 2 3 4 5 6 7 8 9 60 1 2 3 4 5 6 7 8 9 70 1 2 3 4 5 6 7 8 9 80 1 2 3 4 5 6 7 8 9 90 1 2 3 4 5 6 7 8 9
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,X,X,X,X,_, _,_,X,X,X,X,X,X,X,X, X,X,X,_,_,_,_,_,_,_, _,_,X,X,X,X,X,_,_,_], # 89
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,X,X,X,X,X,X, X,_,X,X,X,X,X,X,X,X, X,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 88
                [_,_,_,_,_,_,_,_,_,_, _,_,_,X,X,X,X,X,X,X, X,X,X,X,X,X,X,X,X,X, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 87
                [_,_,_,_,_,_,_,_,_,_, _,_,_,X,X,X,X,X,X,X, X,X,X,X,X,X,X,X,X,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 86
                [_,_,_,_,_,_,_,_,_,_, _,_,X,X,X,X,X,X,X,X, X,X,X,X,X,X,X,X,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 85
                [_,_,_,_,_,_,_,_,_,_, _,_,X,X,X,X,X,X,X,X, X,X,X,X,X,X,X,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 84
                [_,_,_,_,_,_,_,_,_,_, _,X,X,X,X,X,X,X,X,X, X,X,X,X,X,X,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 83
                [_,_,_,_,_,_,_,_,_,_, _,X,X,X,X,X,X,X,X,X, X,X,X,X,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 82
                [_,_,_,_,_,_,_,_,_,_, _,X,X,X,X,X,X,X,X,X, X,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 81
                [_,_,_,_,_,_,_,_,_,_, X,X,X,X,X,X,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 80
              # 50 1 2 3 4 5 6 7 8 9 60 1 2 3 4 5 6 7 8 9 70 1 2 3 4 5 6 7 8 9 80 1 2 3 4 5 6 7 8 9 90 1 2 3 4 5 6 7 8 9
                [_,_,_,_,_,_,_,_,_,_, X,X,X,X,X,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 79
                [_,_,_,_,_,_,_,_,_,_, X,X,X,X,X,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 78
                [_,_,_,_,_,_,_,_,_,_, X,X,X,X,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 77
                [_,_,_,_,_,_,_,_,_,_, X,X,X,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 76
                [_,_,_,_,_,_,_,_,_,X, X,X,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 75
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 74
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 73
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 72
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 71
                [_,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_, _,_,_,_,_,_,_,_,_,_], # 70
            ] # 50 1 2 3 4 5 6 7 8 9 60 1 2 3 4 5 6 7 8 9 70 1 2 3 4 5 6 7 8 9 80 1 2 3 4 5 6 7 8 9 90 1 2 3 4 5 6 7 8 9
        ).T[:, ::-1]
        fld = np.concatenate(
            [np.zeros([100, 70]), np.concatenate([np.zeros([50, 30]), arr], axis=0)],
            axis=1
        )
        # fmt: on

        # Derive the pixels from the grid array above (which only serves as a
        # visual aid); they must be the same as those defined before, only
        # in different order (and, as stated above, order may matter when it
        # comes to triggering the bug that motivated this whole test)
        pixels = np.dstack(np.where(fld > 0))[0].astype(np.int32)
        assert fld.shape == (s.nx, s.ny)
        assert sorted(pixels.tolist()) == sorted(s.pixels.tolist())
        # If no error is raised, then the grid array matches the pixels above

        # 4-connectivity
        # fmt: off
        s.shells4 = [
            np.array(
                [
                    (86, 98),
                    (86, 97),
                    (87, 97),
                    (88, 97),
                    (87, 97),
                    (87, 96),
                    (87, 95),
                    (88, 95),
                    (89, 95),
                    (90, 95),
                    (91, 95),
                    (92, 95),
                    (93, 95),
                    (93, 94),
                    (93, 93),
                    (93, 92),
                    (94, 92),
                    (95, 92),
                    (95, 91),
                    (94, 91),
                    (94, 90),
                    (94, 89),
                    (95, 89),
                    (96, 89),
                    (95, 89),
                    (94, 89),
                    (93, 89),
                    (92, 89),
                    (92, 90),
                    (91, 90),
                    (90, 90),
                    (90, 91),
                    (89, 91),
                    (88, 91),
                    (87, 91),
                    (86, 91),
                    (85, 91),
                    (84, 91),
                    (84, 90),
                    (83, 90),
                    (82, 90),
                    (82, 89),
                    (81, 89),
                    (80, 89),
                    (80, 88),
                    (79, 88),
                    (79, 87),
                    (78, 87),
                    (78, 86),
                    (77, 86),
                    (77, 85),
                    (76, 85),
                    (76, 84),
                    (75, 84),
                    (75, 83),
                    (74, 83),
                    (73, 83),
                    (73, 82),
                    (72, 82),
                    (71, 82),
                    (70, 82),
                    (70, 81),
                    (69, 81),
                    (68, 81),
                    (67, 81),
                    (66, 81),
                    (65, 81),
                    (65, 80),
                    (64, 80),
                    (64, 79),
                    (64, 78),
                    (63, 78),
                    (63, 77),
                    (62, 77),
                    (62, 76),
                    (61, 76),
                    (61, 75),
                    (60, 75),
                    (59, 75),
                    (60, 75),
                    (60, 76),
                    (60, 77),
                    (60, 78),
                    (60, 79),
                    (60, 80),
                    (61, 80),
                    (61, 81),
                    (61, 82),
                    (61, 83),
                    (62, 83),
                    (62, 84),
                    (62, 85),
                    (63, 85),
                    (63, 86),
                    (63, 87),
                    (64, 87),
                    (64, 88),
                    (65, 88),
                    (65, 89),
                    (65, 90),
                    (66, 90),
                    (66, 91),
                    (66, 90),
                    (67, 90),
                    (67, 89),
                    (68, 89),
                    (68, 88),
                    (69, 88),
                    (70, 88),
                    (70, 87),
                    (71, 87),
                    (72, 87),
                    (72, 88),
                    (72, 89),
                    (73, 89),
                    (73, 90),
                    (73, 91),
                    (74, 91),
                    (74, 92),
                    (74, 91),
                    (73, 91),
                    (73, 90),
                    (73, 89),
                    (74, 89),
                    (75, 89),
                    (76, 89),
                    (77, 89),
                    (77, 90),
                    (77, 91),
                    (77, 92),
                    (76, 92),
                    (77, 92),
                    (77, 93),
                    (77, 94),
                    (76, 94),
                    (76, 95),
                    (76, 96),
                    (76, 97),
                    (77, 97),
                    (77, 98),
                    (77, 97),
                    (77, 96),
                    (78, 96),
                    (79, 96),
                    (79, 97),
                    (80, 97),
                    (81, 97),
                    (82, 97),
                    (83, 97),
                    (83, 98),
                    (84, 98),
                    (85, 98),
                    (86, 98),
                ]
            ),
        ]
        s.holes4 = [
            np.array(
                [
                    (86, 97),
                    (86, 96),
                    (86, 95),
                    (86, 94),
                    (85, 94),
                    (84, 94),
                    (84, 93),
                    (83, 93),
                    (82, 93),
                    (83, 93),
                    (83, 92),
                    (83, 91),
                    (82, 91),
                    (81, 91),
                    (80, 91),
                    (80, 92),
                    (79, 92),
                    (79, 93),
                    (78, 93),
                    (77, 93),
                    (77, 94),
                    (77, 95),
                    (77, 96),
                    (78, 96),
                    (79, 96),
                    (80, 96),
                    (80, 95),
                    (80, 96),
                    (81, 96),
                    (81, 97),
                    (82, 97),
                    (83, 97),
                    (84, 97),
                    (85, 97),
                    (86, 97),
                ]
            ),
        ]
        # fmt: on

        # 8-connectivity
        # fmt: off
        s.shells8 = [
            np.array(
                [
                    (86, 98), (87, 97), (88, 97), (87, 96), (88, 95), (89, 95),
                    (90, 95), (91, 95), (92, 95), (93, 95), (93, 94), (93, 93),
                    (94, 92), (95, 92), (95, 91), (94, 90), (95, 89), (96, 89),
                    (95, 89), (94, 89), (93, 89), (92, 89), (91, 90), (90, 90),
                    (89, 91), (88, 91), (87, 91), (86, 91), (85, 91), (84, 90),
                    (83, 90), (82, 89), (81, 89), (80, 88), (79, 87), (78, 86),
                    (77, 85), (76, 84), (75, 83), (74, 83), (73, 82), (72, 82),
                    (71, 82), (70, 81), (69, 81), (68, 81), (67, 81), (66, 81),
                    (65, 80), (64, 79), (64, 78), (63, 77), (62, 76), (61, 75),
                    (60, 75), (59, 75), (60, 76), (60, 77), (60, 78), (60, 79),
                    (60, 80), (61, 81), (61, 82), (61, 83), (62, 84), (62, 85),
                    (63, 86), (63, 87), (64, 88), (65, 89), (65, 90), (66, 91),
                    (67, 90), (68, 89), (69, 88), (70, 88), (71, 87), (72, 88),
                    (72, 89), (73, 90), (73, 91), (74, 92), (74, 91), (73, 90),
                    (74, 89), (75, 89), (76, 89), (77, 90), (77, 91), (76, 92),
                    (77, 93), (76, 94), (76, 95), (76, 96), (76, 97), (77, 98),
                    (77, 97), (78, 96), (79, 97), (80, 97), (81, 97), (82, 97),
                    (83, 98), (84, 98), (85, 98), (86, 98),
                ]
            ),
        ]
        s.holes8 = [
            np.array(
                [
                    (85, 97), (86, 96), (86, 95), (85, 94), (84, 94), (83, 93),
                    (82, 93), (83, 92), (82, 91), (81, 91), (80, 92), (79, 93),
                    (78, 93), (77, 94), (77, 95), (78, 96), (79, 96), (80, 95),
                    (81, 96), (82, 97), (83, 97), (84, 97), (85, 97),
                ]
            ),
        ]
        # fmt: on

    def test_4c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=4)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells4, holes, s.holes4)

    def test_4c_shuffle(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=4)
        pixels = s.pixels.copy()
        shells1, holes1 = pixels_find_boundaries(pixels, constants=const)
        np.random.shuffle(pixels)
        shells2, holes2 = pixels_find_boundaries(pixels, constants=const)
        s.assertBoundaries(shells1, shells2, holes1, holes2)
        np.random.shuffle(s.pixels)

    def test_8c(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        shells, holes = pixels_find_boundaries(s.pixels, constants=const)
        s.assertBoundaries(shells, s.shells8, holes, s.holes8)

    def test_8c_shuffle(s):
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        pixels = s.pixels.copy()
        shells1, holes1 = pixels_find_boundaries(pixels, constants=const)
        np.random.shuffle(pixels)
        shells2, holes2 = pixels_find_boundaries(pixels, constants=const)
        s.assertBoundaries(shells1, shells2, holes1, holes2)
        np.random.shuffle(s.pixels)


if __name__ == "__main__":
    unittest.main()
