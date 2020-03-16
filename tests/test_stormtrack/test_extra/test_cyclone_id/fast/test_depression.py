#!/usr/bin/env python3

# Standard library
import json
import logging as log
import unittest
from pprint import pprint as pp
from unittest import TestCase

# Third-party
from numpy import array as arr
from numpy.testing import assert_almost_equal
from shapely.geometry import Polygon

# First-party
import stormtrack.extra.io_misc as io
from stormtrack.extra.cyclone_id.cyclone import Depression
from stormtrack.extra.cyclone_id.cyclone import DepressionFactory

# Local
from ...testing_utilities import ContourSimple
from ...testing_utilities import PointSimple
from ...testing_utilities import contours_are_sorted
from ...testing_utilities import create_nested_circular_contours as cncc
from ...testing_utilities import shuffle_contours_assert


# CONTOUR CLUSTER DEFINITIONS


def plot_cont(fct):
    cont, min = fct()
    outfile = "depr_{n}.png".format(n=fct.__name__)
    io.plot_contours(outfile, cont, min, labels=lambda x: x.lvl)


# IDENTIFICATION


def cont1():
    """
    One simple perfectly nested cluster (5 contours). Contains 1 minimum.
    """
    cont, min = cncc(5, (3.0, 3.0), 0.2, (1, 1))
    return cont, min


def cont2():
    """
    Two simple perfectly nested clusters (5 and 3 contours, resp.).
    The enclosing countours of the two clusteres have the same value.
    Contains 2 minima.
    """
    cont_min = [cncc(5, (3.0, 3.0), 0.2, (1, 1)), cncc(3, (4.0, 4.0), 0.1, (3, 1))]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def cont3():
    """
    One cluster (2 shared contours) with two subclusters (5 and 3 contours).
    Contains 2 minima.
    """
    cont_min = [
        cncc(5, (6.00, 3.00), 0.2, (1, 1)),
        cncc(2, (7.00, 4.00), 0.1, (4, 1), rmin=0.15),
        cncc(2, (6.25, 3.25), 0.3, (6, 1), rmin=1.6, no_min=True),
    ]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def cont4():
    """
    Two clusters, namely <cont1> (5 contours) and <cont3> 4 contours).
    The enclosing contours of the clusters have a different value.
    Contains 3 minima.
    """
    cont_min = [
        cncc(5, (6.00, 3.00), 0.2, (1, 1)),
        cncc(2, (7.00, 4.00), 0.1, (4, 1), rmin=0.15),
        cncc(2, (6.25, 3.25), 0.3, (6, 1), rmin=1.6, no_min=True),
        cncc(5, (3.00, 3.00), 0.2, (1, 1)),
    ]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def cont5():
    """
    1 cluster with 3 sub-clusters (1 shared contour, 2 contours each)
    Contains 3 minima.
    """
    cont_min = [
        cncc(2, (1.55, 1.6), 0.2, (1, 1)),
        cncc(2, (2.45, 1.6), 0.2, (1, 1)),
        cncc(2, (2.00, 2.4), 0.2, (1, 1)),
        cncc(1, (2.00, 1.9), 1.1, (3, 1), no_min=True),
    ]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


def cont6():
    """
    1 cluster (2 shared contours) with 2 subclusters (those from <cont4>).
    Contains 3 minima (subclusters contain 1 and 2, resp.).
    """
    cont_min = [
        cncc(5, (6.00, 3.00), 0.2, (1, 1)),
        cncc(2, (7.00, 4.00), 0.1, (4, 1), rmin=0.15),
        cncc(2, (6.25, 3.25), 0.3, (6, 1), rmin=1.6, no_min=True),
        cncc(5, (3.00, 3.00), 0.2, (3, 1)),
        cncc(2, (5.20, 3.00), 0.4, (8, 1), rmin=3.4, no_min=True),
    ]
    cont = [e for lst in cont_min for e in lst[0]]
    min = [e for lst in cont_min for e in lst[1]]
    return cont, min


# TEST CASES


class TestVarious(TestCase):
    """Various short tests that don't need separate TestCases."""

    def test_count_children_single(s):
        """Check whether the number of children is correctly returned.

        This test is for the case with only one child.
        """
        cont, min = cncc(3, (3.0, 3.0), 0.2, (1, 1), no_min=True)
        clust = Depression.create(cont, min)[0]
        s.assertEqual(clust.n_children(), 1)

    def test_count_children_multiple(s):
        """As previous test, but for the case with multiple children.

        This test is for the case with only one child.
        """
        clust = Depression.create(*cont5())[0]
        s.assertEqual(clust.n_children(), 3)

    def test_perfectly_nested_true(s):
        """Test Depression.is_perfectly_nested(), which returns True if all
        contours are perfectly nested, and False if the cluster branches at
        some point into separate sub-clusters.

        This test is for a perfectly nested cluster.
        """
        s.assertTrue(Depression.create(*cont1())[0].is_perfectly_nested())

    def test_perfectly_nested_false(s):
        """As previous test, but for an imperfectly nested cluster."""
        s.assertFalse(Depression.create(*cont3())[0].is_perfectly_nested())

    def test_contours_simple(s):
        """Test the contours method for a simple cluster (perfectly nested)."""
        cont, min = cont1()
        clust = Depression.create(cont, min)[0]
        s.assertSetEqual(set(clust.contours()), set(cont))

    def test_contours_complex(s):
        """As ~_simple, but for a complex cluster (imperfectly nested).

        All contours of the cluster and all subclusters are collected.
        """
        cont, min = cont3()
        clust = Depression.create(cont, min)[0]
        s.assertSetEqual(set(clust.contours()), set(cont))

    def test_path(s):
        """Return the path of the contour as a list of coordinate tuples."""
        outer_path = [(1, 1), (4, 1), (4, 5), (1, 5), (1, 1)]
        inner_path = [(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]
        cont = [ContourSimple(outer_path, 2), ContourSimple(inner_path, 1)]
        depr = Depression.create(cont)[0]
        s.assertCountEqual(depr.path(), outer_path)


class TestInitializationPerfectlyNested(TestCase):
    """Initialize a cluster with unsorted contours.

    The contours are perfectly nested ("simple" case).
    Check that tha contours are sorted, linked to each other, etc.

    """

    def setUp(s):
        s.cont, s.min = cont1()
        s.enclosing = s.cont[-1]  # sorted ascendingly by contour value
        assert contours_are_sorted(s.cont)
        shuffle_contours_assert(s.cont)
        assert not contours_are_sorted(s.cont)
        s.clust = Depression.create(s.cont, s.min)[0]

    def test_contours_sorted(s):
        """Check that Depression.contours() returnes a sorted array
        (descending).
        """
        cont = s.clust.contours()
        s.assertTrue(contours_are_sorted(cont, reverse=True))

    def test_enclosing_contour(s):
        """Check that Depression.enclosing_contour returns the correct contour.

        Also all other contours check for non-equality.
        """
        s.assertEqual(s.clust.contour, s.enclosing)
        for cont in s.cont:
            if not cont == s.enclosing:
                s.assertNotEqual(cont, s.enclosing)

    def test_linking(s):
        """Check that the coutours are correctly linked to each other."""
        cont_list = []
        cont = s.clust
        while True:
            cont_list.append(cont.contour)
            if len(cont.children) < 1:
                break
            cont = cont.children[0]
            # !! cont.children[0] only works because of perfect nesting !!
        s.assertListEqual(cont_list, s.clust.contours())


class TestInitializationComplex(TestCase):
    """Test the initialization of a cluster with nested subclusters."""

    def setUp(s):

        # Define the contours (based on cont5)
        cont_min = [
            cncc(2, (1.55, 1.6), 0.2, (1, 1)),
            cncc(2, (2.45, 1.6), 0.2, (1, 1)),
            cncc(2, (2.00, 2.4), 0.2, (1, 1)),
            cncc(1, (2.00, 1.9), 1.1, (3, 3), no_min=True),
        ]
        s.cont = [e for lst in cont_min for e in lst[0]]
        s.min_complete = [e for lst in cont_min for e in lst[1]]
        shuffle_contours_assert(s.min_complete)

        # Save innermost contours
        s.cont_inner = [s.cont[0], s.cont[2], s.cont[4]]

        # Split the minima into two lists
        s.min_incomplete = s.min_complete[0:2]
        s.min_missing = s.min_complete[2:3]
        shuffle_contours_assert(s.min_incomplete)
        shuffle_contours_assert(s.min_missing)

        # Add some additional points
        s.pt_addit = [
            PointSimple(1.57, 1.6, 0.5),
            PointSimple(2.46, 1.59, 0.3),
            PointSimple(2.4, 1.7, 4),
        ]
        s.pt_all = s.min_complete + s.pt_addit
        shuffle_contours_assert(s.pt_addit)
        shuffle_contours_assert(s.pt_all)

    def test_innermost_contours(s):
        """Return a list containing the innermost contours."""
        clust = Depression.create(s.cont, s.pt_all)[0]
        foo = clust.innermost_contours()
        s.assertSetEqual(set(foo), set(s.cont_inner))

    def test_assign_minima_complete_list(s):
        """Assign a list of points that contains all minima (i.e. at least
        one for every innermost contour). The deepest minimum in each inner-
        most contour is retained, all other minima are discarded.
        """
        clust = Depression.create(s.cont, s.pt_all)[0]
        s.assertSetEqual(set(clust.minima()), set(s.min_complete))


# CONTOUR- AND DEPTH-RELATED TESTS


class TestDepthNoSubcluster(TestCase):
    """Test the case of a single cluster without any nested subclusters.

    This is a trivial control case for many methods.
    For the method descriptions, see the next test case.
    """

    def setUp(s):
        """The cluster consists of 5 contours and one minimum, which amounts
        to a total depth of 5.0.
        """
        s.clust = Depression.create(*cont1())[0]

    def test_count_shared_contours(s):
        s.assertEqual(s.clust.count_shared_contours(), 5)

    def test_shared_depth(s):
        s.assertAlmostEqual(s.clust.shared_depth(), 5.0)

    def test_min_contour_depth(s):
        s.assertEqual(s.clust.min_contour_depth(), 5)

    def test_max_contour_depth(s):
        s.assertEqual(s.clust.max_contour_depth(), 5)

    def test_min_depth(s):
        s.assertAlmostEqual(s.clust.min_depth(), 5.0)

    def test_max_depth(s):
        s.assertAlmostEqual(s.clust.max_depth(), 5.0)

    def test_shared_contour_ratio(s):
        s.assertAlmostEqual(s.clust.shared_contour_ratio(), 1.0)

    def test_shared_depth_ratio(s):
        s.assertAlmostEqual(s.clust.shared_depth_ratio(), 1.0)


class TestDepthTwoSubclusters(TestCase):
    """Test the case of a single cluster with two subclusters."""

    def setUp(s):
        """
        2 + 2|5 contours = 4|7 contours
        2 + 2|5 contours + minimum = 4.0|7.0
        """
        s.clust = Depression.create(*cont3())[0]

    def test_count_shared_contours(s):
        """Test the identification of shared contours in complex clusters.

        Shared contours are those not unique to either subcluster.
        The total number of contours is from (and including) the main
        enclosing contour to (and including) the immermost contour of
        the deepest subcluster.
        """
        s.assertEqual(s.clust.count_shared_contours(), 2)

    def test_shared_depth(s):
        """Test the computation of the depth of the cluster shared by all
        subclusters, i.e. above any branching into sublusters.
        """
        s.assertAlmostEqual(s.clust.shared_depth(), 2.0)

    def test_min_contour_depth(s):
        """Test contour depth-related methods.

        Note that these methods measure the depth in number of contours,
        not taking the actual contours levels into account. Both the enclosing
        and the innermost contours are included in the count.
        """
        s.assertEqual(s.clust.min_contour_depth(), 4)

    def test_max_contour_depth(s):
        s.assertEqual(s.clust.max_contour_depth(), 7)

    def test_min_depth(s):
        """As the previous test case, but these methods measure the actual
        (physicsl) depth of the clusters in terms of contour levels.

        The depth of a cluster corresponds to the distance between the
        enclosing contour and the centran minimum (and NOT only to the
        innermost contour).
        """
        s.assertAlmostEqual(s.clust.min_depth(), 4.0)

    def test_max_depth(s):
        s.assertAlmostEqual(s.clust.max_depth(), 7.0)

    def test_shared_contour_ratio(s):
        s.assertAlmostEqual(s.clust.shared_contour_ratio(), (2.0 / 7.0))

    def test_shared_depth_ratio(s):
        s.assertAlmostEqual(s.clust.shared_depth_ratio(), (2.0 / 7.0))

    def test_innermost_shared_contour(s):
        """Find the innermost contour containing all minima."""
        cont = s.clust.innermost_shared_contour()
        s.assertEqual(cont.lvl, 6)


class TestReduceClusterToNMinima(TestCase):
    """Reduce a cluster to one or more subclusters that contain at most a
    certain number of minima (if the cluster contains too many to begin with).
    """

    def setUp(s):
        """The cluster contains 1 subclusters, one of which contains
        1 minimum, the other 2.
        """
        cont, min = cont6()
        s.depr = Depression.create(cont, min)[0]

    def test_already_below_threshold(s):
        """Control case: Threshold 3: Main cluster already meets condition.
        1 cluster is returned (the original one, at level 9).
        """
        children = s.depr.reduce_to_max_n_minima(3)
        s.assertEqual(len(children), 1)
        levels = set(c.lvl for c in children)
        s.assertSetEqual(levels, {9})
        parents = set(c.parent for c in children)
        s.assertSetEqual(parents, {None})

    def test_split_in_2_subclusters(s):
        """Threshold 2: Both main subclusters meet the condition.

        2 clusters are returned, at levels 5 and 7, resp.
        """
        children = s.depr.reduce_to_max_n_minima(2)
        s.assertEqual(len(children), 2)
        levels = set(c.lvl for c in children)
        s.assertSetEqual(levels, {7})
        parents = set(c.parent for c in children)
        s.assertSetEqual(parents, {None})

    def test_split_in_3_subclusters(s):
        """Threshold 1: Only one main subcluster meets the condition, the
        other is split up further.
        3 clusters are returned, all at level 5 (incidentally).
        """
        children = s.depr.reduce_to_max_n_minima(1)
        s.assertEqual(len(children), 3)
        levels = set(c.lvl for c in children)
        s.assertSetEqual(levels, {5, 7})
        parents = set(c.parent for c in children)
        s.assertSetEqual(parents, {None})


class TestCenter(TestCase):
    """Compute the Depression center (focal point of all minima, weighted by
    their relative depth).
    """

    def setUp(s):
        pass

    def test_two_minima(s):
        """Two centers of different depths.

         - Enclosing contour: level 6
         - Minimum 1 @ (6,3): level 0 -> rel. depth 6 -> weight 0.66
         - Minimum 2 @ (7,4): level 3 -> rel. depth 3 -> weight 0.33

         -> Center: 2/3*(6,3) + 1/3*(7,4) = (4+2.33,2+1.33) = (6.33,3.33)
        """
        clust = Depression.create(*cont3())[0]
        res = clust.center()
        sol = (6.33, 3.33)
        assert_almost_equal(sol, res, 2)

    def test_three_minima(s):
        """Three centers of different depths.

         - Enclosing contour: level 8
         - Minimum 1 @ (3,3): level 2 -> rel. depth 6 -> weight 0.32
         - Minimum 2 @ (6,3): level 0 -> rel. depth 8 -> weight 0.42
         - Minimum 3 @ (7,4): level 3 -> rel. depth 5 -> weight 0.26

         -> Center: 0.32*(3,3) + 0.42*(6,5) + 0.26*(7,4) = (5.32,3.26)
        """
        clust = Depression.create(*cont6())[0]
        res = clust.center()
        sol = (5.32, 3.26)
        assert_almost_equal(sol, res, 2)


# DEPRESSION FACTORY


class TestCreationComplex(TestCase):
    """Similar to test case TestInitializationSimple, but for the complex
    case of imperfectly nested contours, i.e. with subclusters, as
    well as multiple top-level clusters.

    Note that in contrast to the other case, the factory method
    Depression.create() is used to create objects, not
    direct initialization (this is necessary to seamlessly deal with
    multiple top-level clusters).
    """

    def setUp(s):
        pass

    def test_separate_toplevel_clusters_same_level(s):
        """Contours belonging to two separate top-level clusters are
        clustered correctly. In this case, the enclosing contours
        of the two clusters have the same value.
        """
        clust = Depression.create(*cont2())
        s.assertEqual(len(clust), 2)

    def test_separate_toplevel_clusters_different_levels(s):
        """Contours belonging to two separate top-level clusters are
        clustered correctly. In this case, the enclosing contours
        of the two clusters have a different value.
        """
        clust = Depression.create(*cont4())
        s.assertEqual(len(clust), 2)


if __name__ == "__main__":
    # plot_cont(cont1)
    # plot_cont(cont2)
    # plot_cont(cont3)
    # plot_cont(cont4)
    # plot_cont(cont5)
    # plot_cont(cont6)
    unittest.main()
