#!/usr/bin/env python

# Standard library
import logging as log
import random
import unittest
from pprint import pprint as pp
from unittest import TestCase

# Third-party
from numpy import array as arr
from numpy.testing import assert_almost_equal
from shapely.geometry import Point
from shapely.geometry import Polygon

# First-party
import stormtrack.extra.utilities_misc as io
from stormtrack.extra.cyclone_id.cyclone import Cyclone
from stormtrack.extra.cyclone_id.cyclone import Depression
from stormtrack.extra.cyclone_id.cyclone import DoubleCenterCyclone
from stormtrack.extra.cyclone_id.cyclone import SingleCenterCyclone
from stormtrack.extra.cyclone_id.cyclone import TripleCenterCyclone

# Local
from ...testing_utilities import ContourSimple
from ...testing_utilities import PointSimple
from ...testing_utilities import create_nested_circular_contours as cncc


# CONTOUR CLUSTER DEFINITIONS


def plot_cont(fct):
    cont, min = fct()
    outfile = "cycl_{n}.png".format(n=fct.__name__)
    io.plot_contours(outfile, cont, min, labels=lambda x: x.lvl)


def cont1(dx=0.0, dy=0.0, dlvl=0.0):
    """
    A simple perfectly nested cluster (5 contours).
    Contains 1 minimum.
    """
    cx, cy = (3.7 + dx, 5.8 + dy)
    cont, min = cncc(5, (cx - 0.40, cy + 1.80), 0.12, (3 + dlvl, 1))
    return cont, min


def cont2(dx=0.0, dy=0.0, dlvl=0.0):
    """
    A cluster with two subclusters.
    The subclusters have two and three contours, resp., and are enclosed
    by four contours, which amounts to a maximum depth of seven contours.
    """
    cx, cy = (3.7 + dx, 5.8 + dy)
    cont_min = [
        cncc(2, (cx - 0.62, cy - 0.15), 0.20, (2 + dlvl, 1)),
        cncc(3, (cx + 0.42, cy + 0.16), 0.20, (1 + dlvl, 1)),
        cncc(4, (cx + 0.03, cy + 0.05), 0.28, (4 + dlvl, 1), rmin=1.18, no_min=True),
    ]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def cont3(dx=0.0, dy=0.0, dlvl=0.0):
    """
    A cluster with two subclusters.
    The subclusters have three and five contours, resp., and are enclosed
    by two contours, which amounts to a maximum depth of seven contours.
    """
    cx, cy = (3.2 + dx, 4.3 + dy)
    cont_min = [
        cont1(cx - 3.70, cy - 5.80, dlvl - 2),
        cncc(3, (cx + 0.70, cy + 1.45), 0.15, (3 + dlvl, 1)),
        cncc(2, (cx + 0.10, cy + 1.70), 0.15, (6 + dlvl, 1), rmin=1.2, no_min=True),
    ]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def cont4(dx=0.0, dy=0.0, dlvl=0.0):
    """
    A cluster with three subclusters.
    The cluster is based on cont2, but with an additional subcluster and
    many move all-enclosing contours.
    Two subclusters have two contours and the other has three. The are
    enclosed by a total of eight contours, which amounts to a maximum
    depth of eleven contours.
    """
    cx, cy = (3.7 + dx, 5.8 + dy)
    cont_min = [
        cncc(2, (cx - 0.65, cy + 0.25), 0.20, (2 + dlvl, 1)),
        cncc(2, (cx - 0.22, cy - 0.55), 0.20, (2 + dlvl, 1)),
        cncc(3, (cx + 0.48, cy + 0.26), 0.20, (1 + dlvl, 1)),
        cncc(8, (cx + 0.03, cy + 0.05), 0.28, (4 + dlvl, 1), rmin=1.22, no_min=True),
    ]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def contX(dx=0.0, dy=0.0, dlvl=0.0):
    """
    A cluster with two centers.
    The cluster corresponds to cont1, surrounded by three contours with
    lots of space in-between. It is primarily used as a bilding block
    for more complex cluster contructs following below.
    """
    cx, cy = (3.7 + dx, 5.8 + dy)
    cont_min = [
        cont2(cx - 3.70, cy - 5.80, dlvl),
        cncc(3, (cx + 0.00, cy + 1.40), 0.40, (8 + dlvl, 1), rmin=4.00, no_min=True),
    ]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def cont5(dx=0.0, dy=0.0, dlvl=0):
    """
    A cluster with three centers.
    Two of those are near each other (in terms of shared contours), while
    the third is farther away. This cluster comprises a combination of
    clust1 and clust2, surrounded by three contours.
    """
    cx, cy = (3.7 + dx, 5.8 + dy)
    cont_min = [contX(cx - 3.7, cy - 5.8, dlvl), cont1(cx - 3.7, cy - 3.8, dlvl)]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def cont6(dx=0.0, dy=0.0, dlvl=0.0):
    """
    A cluster with four centers.
    This cluster corresponds to cont5, with an additional simple cluster
    inside the three enclosing contours.
    """
    cx, cy = (3.7 + dx, 5.8 + dy)
    cont_min = [contX(cx - 3.70, cy - 5.80), cont3(cx - 3.20, cy - 2.30)]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


# TEST CASES


# CYCLONE CREATION
# - The contour ratio thresholds for double- (DCCs) and triple-center cyclones
# (TCCs) are 0.5 and 0.7, respectively. Furthermore, the parameter nmin_max,
# which is the maximum number of minima in a cyclone, is 3. Both is following
# HC12. The respective arguments to Cyclone.create() are stored in KWARGS.


KWARGS = {"min_depth": 1.0, "nmin_max": 3, "thresh_dcc": 0.5, "thresh_tcc": 0.7}


# SR_TODO: Add a test which checks that too shallow Cyclones are removed
# SR_TODO: when a cluster is split because the shared contour ratio criterion
# SR_TODO: has not been met! Just fixed this, but this hadn't been caught
# SR_TODO: earlier because there was not test!


class TestNoSubclustersOneMinimum(TestCase):
    """One simple cluster with one minimum."""

    def setUp(s):
        """The contour depth of the cluster is 5."""
        s.kwargs = KWARGS
        s.depr = Depression.create(*cont1())[0]
        s.cycl = Cyclone.create(s.depr, **s.kwargs)

    def test_return(s):
        """Return a SCC object."""
        s.assertEqual(len(s.cycl), 1)
        s.assertEqual(type(s.cycl[0]), SingleCenterCyclone)


class TestTwoSubclustersTwoMinimaDeep(TestCase):
    """One deep cluster with two subclusters and two minima in total."""

    def setUp(s):
        """The maximum contour depth of the cluster is 7.
        The outermost 4 contours are shared by both subclusters.
        This results in a shared contour ratio of 4/7=0.57, which is
        above the threshold for DCCs.
        """
        s.kwargs = KWARGS
        s.depr = Depression.create(*cont2())[0]

    def test_return(s):
        """Return a DCC object."""
        cycl = Cyclone.create(s.depr, **s.kwargs)
        s.assertEqual(len(cycl), 1)
        s.assertEqual(type(cycl[0]), DoubleCenterCyclone)


class TestTwoSubclustersTwoMinimaShallow(TestCase):
    """One shallow cluster with two subclusters and two minima in total."""

    def setUp(s):
        """The maximum contour depth of the cluster is 7.
        The outermost 2 contours are shared by both subclusters.
        This results in a shared contour ratio of 2/7=0.28, which is
        below the threshold for DCCs.
        """
        s.kwargs = KWARGS
        s.depr = Depression.create(*cont3())[0]

    def test_return(s):
        """Return two SCC objects."""
        cycl = Cyclone.create(s.depr, **s.kwargs)
        s.assertEqual(len(cycl), 2)
        s.assertEqual(type(cycl[0]), SingleCenterCyclone)
        s.assertEqual(type(cycl[1]), SingleCenterCyclone)


class TestThreeSubclustersThreeMinimaDeep(TestCase):
    """One deep cluster with three subclusters and three minima in total."""

    def setUp(s):
        """The maximum contour depth of the cluster is 11.
        The outermost 8 contours are shared by all three subclusters.
        This results in a shared contour ratio of 8/11=0.73, which is
        above the threshold for TCCs.
        """
        s.kwargs = KWARGS
        s.depr = Depression.create(*cont4())[0]

    def test_return(s):
        """Return a TCC object."""
        cycl = Cyclone.create(s.depr, **s.kwargs)
        s.assertEqual(len(cycl), 1)
        s.assertEqual(type(cycl[0]), TripleCenterCyclone)


class TestTwoSubclustersThreeMinima(TestCase):
    """One cluster with two major subclusters and three minima in total."""

    def setUp(s):
        """The maximum contour depth of the cluster is 10.
        The outermost 3 contours are shared by both subclusters.
        This results in a shared contour ratio of 3/10=0.30, which is
        below the threshold for TCCs. The cluster is therefore split up.

        The maximum depth of the subcluster which contains two minimums
        is 7, which, given 4 shared contours, results in a shared contour
        ratio of 4/7=0.57, which is above the threshold for DCCs.
        """
        s.kwargs = KWARGS
        s.depr = Depression.create(*cont5())[0]

    def test_return(s):
        """Return an SCC and a DCC object."""
        cycl = Cyclone.create(s.depr, **s.kwargs)
        s.assertEqual(len(cycl), 2)
        types = [type(c) for c in cycl]
        s.assertEqual(types.count(SingleCenterCyclone), 1)
        s.assertEqual(types.count(DoubleCenterCyclone), 1)


class TestThreeSubclustersFourMinima(TestCase):
    """One cluster with three major subclusters and four minima in total."""

    def setUp(s):
        """The cluster contains more than 3 minima. Thus, it is split up
        into 2 clusters which both contain 2 minima. They correspond
        to cont3 and cont5, with shared contour ratios of 0.28 and 0.57,
        respectively.
        """
        s.kwargs = KWARGS
        s.depr = Depression.create(*cont6())[0]

    def test_return(s):
        """Return 2 SCC and 1 DCC objects."""
        cycl = Cyclone.create(s.depr, **s.kwargs)
        s.assertEqual(len(cycl), 3)
        types = [type(c) for c in cycl]
        s.assertEqual(types.count(SingleCenterCyclone), 2)
        s.assertEqual(types.count(DoubleCenterCyclone), 1)


if __name__ == "__main__":
    # plot_cont(cont1)
    # plot_cont(cont2)
    # plot_cont(cont3)
    # plot_cont(cont4)
    # plot_cont(cont5)
    # plot_cont(cont6)
    unittest.main()
