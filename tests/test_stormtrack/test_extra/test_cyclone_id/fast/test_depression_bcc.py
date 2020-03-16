#!/usr/bin/env python3

# Standard library
import unittest
from unittest import TestCase

# First-party
import stormtrack.extra.io_misc as io
from stormtrack.extra.cyclone_id.cyclone import Depression
from stormtrack.extra.utilities_misc import Contour

# Local
from ...testing_utilities import ContourSimple
from ...testing_utilities import PointSimple


CONTOUR = ContourSimple


def cont1(coord=False, cont=False):
    """
    Simple contour cluster with 4 closed contours and and 8 BCCs.
    """
    coord_list = [
        # closed contours
        [(0, 3.5), (1, 4.0), (1, 5.0), (0, 5.5), (-1, 5.0), (-1, 4.0), (0, 3.5)],  #  0
        [(0, 2.5), (2, 3.5), (2, 5.5), (0, 6.5), (-2, 5.5), (-2, 3.5), (0, 2.5)],  #  1
        [(0, 1.5), (3, 3.0), (3, 6.0), (0, 7.5), (-3, 6.0), (-3, 3.0), (0, 1.5)],  #  2
        [(0, 0.5), (4, 2.5), (4, 6.5), (0, 8.5), (-4, 6.5), (-4, 2.5), (0, 0.5)],  #  3
        # boundary-crossing contours
        [
            (1, 0.0),
            (5, 2.0),
            (5, 7.0),
            (0, 9.5),
            (-5, 7.0),
            (-5, 2.0),
            (-1, 0.0),
        ],  #  4 20%
        [
            (3, 0.0),
            (6, 1.5),
            (6, 7.5),
            (0, 10.5),
            (-6, 7.5),
            (-6, 1.5),
            (-3, 0.0),
        ],  #  5 33%
        [
            (5, 0.0),
            (7, 1.0),
            (7, 8.0),
            (0, 11.5),
            (-7, 8.0),
            (-7, 1.0),
            (-5, 0.0),
        ],  #  6 43%
        [
            (7, 0.0),
            (8, 0.5),
            (8, 8.5),
            (0, 12.5),
            (-8, 8.5),
            (-8, 0.5),
            (-7, 0.0),
        ],  #  7 50%
        [(9, 0.0), (9, 9.0), (0, 13.5), (-9, 9.0), (-9, 0.0)],  #  8 56%
        [(10, 0.0), (10, 9.5), (0, 14.5), (-10, 9.5), (-10, 0.0)],  #  9 60%
        [(11, 0.0), (11, 10.0), (0, 15.5), (-11, 10.0), (-11, 0.0)],  # 10 64%
        [(12, 0.0), (12, 10.5), (0, 16.5), (-12, 10.5), (-12, 0.0)],  # 11 67%
    ]
    cont_list = [CONTOUR(p, level=i, id=i) for i, p in enumerate(coord_list)]
    if not coord and not cont:
        return coord_list, cont_list
    elif coord:
        return coord_list
    elif cont:
        return cont_list


def cont2(coord=False, cont=False):
    """
    Contour cluster with 2 sub-clusters:
    - 5 shared contours, all BCCs
    - sub-cluster 1: 5 contours, thereof 2 BCCs
    - sub-cluster 2: 4 contours, no BCCs
    -> max. depth: 10
    """
    coord_list = [
        # sub-cluster 1: 5 contours, thereof 2 BCCs
        (0, [(17.0, 3.0), (16.0, 4.0), (15.0, 4.0), (15.0, 3.0), (17.0, 3.0)]),
        (
            1,
            [
                (18.5, 2.0),
                (18.5, 3.0),
                (16.5, 5.0),
                (14.5, 5.0),
                (14.0, 4.5),
                (14.0, 2.0),
                (18.5, 2.0),
            ],
        ),
        (
            2,
            [
                (20.0, 1.0),
                (20.0, 3.0),
                (17.0, 6.0),
                (14.0, 6.0),
                (13.0, 5.0),
                (13.0, 1.0),
                (20.0, 1.0),
            ],
        ),
        (
            3,
            [
                (21.5, 0.5),
                (21.5, 3.0),
                (17.5, 7.0),
                (13.5, 7.0),
                (12.0, 5.5),
                (12.0, 0.5),
            ],
        ),
        (
            4,
            [
                (23.0, 0.5),
                (23.0, 3.0),
                (18.0, 8.0),
                (13.0, 8.0),
                (11.0, 6.0),
                (11.0, 0.5),
            ],
        ),
        # sub-cluster 2: 4 contours, no BCCs
        (1, [(25.0, 9.0), (26.0, 10.0), (25.0, 11.0), (23.0, 11.0), (25.0, 9.0)]),
        (2, [(25.0, 7.0), (28.0, 10.0), (26.0, 12.0), (20.0, 12.0), (25.0, 7.0)]),
        (
            3,
            [
                (25.0, 5.0),
                (30.0, 10.0),
                (27.0, 13.0),
                (19.0, 13.0),
                (18.0, 12.0),
                (25.0, 5.0),
            ],
        ),
        (
            4,
            [
                (25.0, 3.0),
                (32.0, 10.0),
                (28.0, 14.0),
                (18.0, 14.0),
                (16.0, 12.0),
                (25.0, 3.0),
            ],
        ),
        # 5 shared contours, all BCCs
        (
            5,
            [
                (25.5, 0.5),
                (34.0, 9.0),
                (34.0, 11.0),
                (29.0, 16.0),
                (18.0, 16.0),
                (9.0, 7.0),
                (9.0, 0.5),
            ],
        ),
        (
            6,
            [
                (28.5, 0.5),
                (36.0, 8.0),
                (36.0, 11.5),
                (29.5, 18.0),
                (17.0, 18.0),
                (7.0, 8.0),
                (7.0, 0.5),
            ],
        ),
        (
            7,
            [
                (31.5, 0.5),
                (38.0, 7.0),
                (38.0, 12.0),
                (30.0, 20.0),
                (16.0, 20.0),
                (5.0, 9.0),
                (5.0, 0.5),
            ],
        ),
        (
            8,
            [
                (34.5, 0.5),
                (40.0, 6.0),
                (40.0, 12.5),
                (30.5, 22.0),
                (15.0, 22.0),
                (3.0, 10.0),
                (3.0, 0.5),
            ],
        ),
        (
            9,
            [
                (37.5, 0.5),
                (42.0, 5.0),
                (42.0, 13.0),
                (31.0, 24.0),
                (14.0, 24.0),
                (1.0, 11.0),
                (1.0, 0.5),
            ],
        ),
    ]
    cont_list = [CONTOUR(p, level=l, id=i) for i, (l, p) in enumerate(coord_list)]
    if not coord and not cont:
        return coord_list, cont_list
    elif coord:
        return coord_list
    elif cont:
        return cont_list


class TestMethods(TestCase):
    """Test basic BCC-related class properties/methods."""

    def setUp(s):
        s.cont = cont1(cont=True)

    def test_contour_is_boundary_crossing(s):
        """Check that Contour.is_boundary_crossing() works correctly."""
        res = [c.is_boundary_crossing() for c in s.cont]
        sol = 4 * [False] + 8 * [True]
        s.assertCountEqual(res, sol)

    def test_depression_bcc_fraction_single_center(s):
        """Check the method Depression.bcc_fraction for a simple cluster with
        only one center (i.e. a perfectly nested one).
        """
        # cont1: 4 CCs + 8 BCCs = 12 Cs
        cont = cont1(cont=True)
        # res = Depression.create(cont,bcc_frac=1.0)[0].bcc_fraction()
        d = Depression.create(cont, bcc_frac=1.0)[0]
        res = d.bcc_fraction()
        s.assertAlmostEqual(res, 8.0 / 12.0)

    def test_depression_bcc_fraction_double_center(s):
        """Check the method Depression.bcc_fraction for a cluster with
        two centers (i.e. sub-clusters).
        """
        # cont2: 3 CCs + (2+5) BCCs = 10 Cs (deepest sub-cluster only)
        cont = cont2(cont=True)
        res = Depression.create(cont, bcc_frac=1.0)[0].bcc_fraction()
        s.assertAlmostEqual(res, 7.0 / 10.0)

    def test_depression_bcc_fraction_double_center_equal_depth(s):
        """Check the method Depression.bcc_fraction for a cluster with
        two centers, which have equal depth but different number of BCCs.
        The sub-cluster with more BCCs is used for the computation.
        """
        # Remove the innermost contour from the deeper cluster so both
        # have equal depth, but different numbers of BCCs (0/4 and 2/4).
        # The resulting total depth is 4+5=9.
        cont = cont2(cont=True)[1:]
        res = Depression.create(cont, bcc_frac=1.0)[0].bcc_fraction()
        s.assertAlmostEqual(res, 7.0 / 9.0)


class TestSimple(TestCase):
    """Test a perfectly nested contour cluster partly ouside the domain."""

    def setUp(s):
        s.cont = cont1(cont=True)

    def test_depression_bcc_000(s):
        """For bcc-frac=0.0, no BCCs are accepted."""
        dep = Depression.create(s.cont, [], bcc_frac=0.0)
        s.assertEqual(len(dep), 1)
        res = dep[0].contours()
        sol = s.cont[: 4 + 0]
        s.assertCountEqual(res, sol)

    def test_depression_bcc_020(s):
        """For bcc-frac=0.2, only 1 BCC is accepted."""
        dep = Depression.create(s.cont, [], bcc_frac=0.2)
        s.assertEqual(len(dep), 1)
        res = dep[0].contours()
        sol = s.cont[: 4 + 1]
        s.assertCountEqual(res, sol)

    def test_depression_bcc_040(s):
        """For bcc-frac=0.2, only 2 BCCs are accepted."""
        dep = Depression.create(s.cont, [], bcc_frac=0.4)
        s.assertEqual(len(dep), 1)
        res = dep[0].contours()
        sol = s.cont[: 4 + 2]
        s.assertCountEqual(res, sol)

    def test_depression_bcc_050(s):
        """For bcc-frac=0.5, only 4 BCCs are accepted."""
        dep = Depression.create(s.cont, [], bcc_frac=0.5)
        s.assertEqual(len(dep), 1)
        res = dep[0].contours()
        sol = s.cont[: 4 + 4]
        s.assertCountEqual(res, sol)

    def test_depression_bcc_065(s):
        """For bcc-frac=0.65, all but 1 BCCs are accepted."""
        dep = Depression.create(s.cont, [], bcc_frac=0.65)
        s.assertEqual(len(dep), 1)
        res = dep[0].contours()
        sol = s.cont[: 4 + 7]
        s.assertCountEqual(res, sol)

    def test_depression_bcc_070(s):
        """For bcc-frac=0.70, all 8 BCCs are accepted."""
        dep = Depression.create(s.cont, [], bcc_frac=0.7)
        s.assertEqual(len(dep), 1)
        res = dep[0].contours()
        sol = s.cont[: 4 + 8]
        s.assertCountEqual(res, sol)

    def test_depression_bcc_100(s):
        """For bcc-frac=1.0, all 8 BCCs are accepted."""
        dep = Depression.create(s.cont, [], bcc_frac=1.0)
        s.assertEqual(len(dep), 1)
        res = dep[0].contours()
        sol = s.cont[: 4 + 8]
        s.assertCountEqual(res, sol)


class TestSplitting(TestCase):
    """Test a contour cluster with two sub-clusters."""

    # 0 enclosing contour: max_depth=5; BCCs=2; bcc_frac_crit=0.4
    # 1 enclosing contour: max_depth=6; BCCs=2+1=3; bcc_frac_crit=0.5
    # 2 enclosing contours: max_depth=7; BCCs=2+2=4; bcc_frac_crit=0.57

    def setUp(s):
        s.cont = cont2(cont=True)

    def test_bcc_100(s):
        """For bcc_frac=1.0, a single Depression object containing all
        contours is created."""
        depr = Depression.create(s.cont, bcc_frac=1.0)
        s.assertEqual(len(depr), 1)
        res = depr[0].contours()
        s.assertCountEqual(res, s.cont)

    def test_bcc_055(s):
        """For bcc_frac=0.55, a single Depression object containing all
        but the 4 outermost contours is created.
        """
        # For bcc_frac=0.55, 4/5 enclosing contours should be discarded,
        # but the Depression should not yet be split up!
        depr = Depression.create(s.cont, bcc_frac=0.55)
        s.assertEqual(len(depr), 1)
        res = depr[0].contours()
        s.assertCountEqual(res, s.cont[:-4])

    def test_bcc_045(s):
        """For bcc_frac=0.45, the initial Depression object is split up into
        two because all enclosing contours are discarded.
        """
        depr = Depression.create(s.cont, bcc_frac=0.45)
        s.assertEqual(len(depr), 2)
        res = depr[0].contours() + depr[1].contours()
        s.assertSetEqual(set(res), set(s.cont[:-5]))

    def test_bcc_020(s):
        """For bcc_frac=0.2, two Depression objects are created, but neither
        contains any BCCs.
        """
        depr = Depression.create(s.cont, bcc_frac=0.20)
        s.assertEqual(len(depr), 2)
        res = depr[0].contours() + depr[1].contours()
        # This is quite messy...
        sol = s.cont[:3] + s.cont[5:9]
        s.assertSetEqual(set(res), set(sol))


if __name__ == "__main__":
    # io.plot_contours("test_bcc_cont1.png",cont1(cont=True))
    # io.plot_contours("test_bcc_cont2.png",cont2(cont=True))
    unittest.main()
