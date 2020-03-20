#!/usr/bin/env python

# Standard library
import json
import logging as log
import os
import re
import sys
from collections import OrderedDict
from functools import total_ordering
from pprint import pformat
from pprint import pprint as pp

# Third-party
import numpy as np
import scipy as sp

# Local
from ..io_misc import IOReaderBinaryBase
from ..io_misc import IOReaderJsonBase
from ..io_misc import IOWriterBinaryBase
from ..io_misc import IOWriterJsonBase
from ..tracking_old.tracking import FeatureCombination
from ..tracking_old.tracking import FeatureTrackBase
from ..tracking_old.tracking import FeatureTrackerBase
from ..tracking_old.tracking import FeatureTrackFactory
from ..tracking_old.tracking import FeatureTrackIOReaderJson
from ..tracking_old.tracking import FeatureTrackIOWriterJson
from ..tracking_old.tracking import TrackableFeatureBase
from ..tracking_old.tracking import TrackableFeatureMean
from ..utilities_misc import area_lonlat
from ..utilities_misc import Contour
from ..utilities_misc import FieldPoint
from ..utilities_misc import path_is_insignificant


__all__ = []


class Depression:

    next_id = 0

    def __init__(self, *, contour, parent, children, minima, id=None):
        """Initialize a depression object.

        All arguments must be passed by name (i.e. with keyword).

        Find the outermost contour and recursively create nested Depression
        objects for all nested contours.

        Return a Depression object.

        Arguments:
        - contours: List of contours.
        - parent: The next-enclosing contour (None if outermost in cluster).
        - children: List of next-contained contours.
        - minima: List of minima.

        """
        # Set properties from arguments
        self.contour = contour
        self.parent = parent
        self.children = children
        self.minimum = None

        if id is None:
            self._id = self.__class__.next_id
            self.__class__.next_id += 1
        else:
            self._id = id

        # Connect children to self
        for child in self.children:
            child.parent = self

        # Set secondary properties
        self.lvl = self.contour.lvl
        self.lvl = self.lvl  # SR_TODO deprecate this!

        # Initialize the minima
        for dep in self.innermost():
            min_cont = [m for m in minima if dep.contour.contains(m)]
            if len(min_cont) > 0:
                dep.minimum = min(min_cont, key=lambda x: x.lvl)

        # For caching attributes
        self._area = None

    @classmethod
    def create(cls, *args, **kwargs):

        if len(args) == 1:

            # Check whether the object passed is already a Depression
            if isinstance(args[0], Depression):
                return [args[0]]

            # Check whether a list of Depression objects has been passed
            try:
                for obj in args[0]:
                    if not isinstance(obj, Depression):
                        break
                else:
                    return args[0]
            except TypeError:
                pass

        # Run the factory
        return DepressionFactory._create(*args, **kwargs)

    def __repr__(self):
        cls = self.__class__.__name__
        return "<{cls}[{id0}]: id={id1}, level={lvl}>".format(
            cls=cls, id0=id(self), id1=self.id(), lvl=self.lvl
        )

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        """Compare equality of the contour and some properties."""
        if not isinstance(other, Depression):
            return False

        if self.contour != other.contour:
            return False

        if self.lvl != other.lvl:
            return False

        if self.children != other.children:
            return False

        if self.minima() != other.minima():
            return False

        return True

    @total_ordering
    def __lt__(self, other):
        if not isinstance(other, Depression):
            raise ValueError("Only Depression objects supported!")
        return self.lvl < other.lvl

    def id(self):
        return self._id

    def as_polygon(self):
        return self.contour

    def get_info(self):
        cont_entry = OrderedDict()
        cont_entry["class"] = self.__class__.__name__
        cont_entry["id"] = self.id()
        cont_entry["minima_id"] = [m.id for m in self.minima()]
        cont_entry["contours_id"] = [c.id for c in self.contours()]
        cont_entry["n_contours"] = len(self.contours())
        return cont_entry

    def _init_child(self, contours, minima):
        """Initiate a child..."""
        if len(contours) > 0:
            self.children.append(
                Depression(contours=contours, parent=self, children=[], minima=minima)
            )

    def _split_children_iter(self, contours, minima):
        """Return lists of all contours and minima contained by each child.

        Returns an iterator (not a list).

        Arguments:
         - contours: ...
         - minima: ...
        """
        children = contours.pop(0)
        contours = [c for group in contours for c in group]  # flatten
        if self.n_children() == 1:
            yield [children[0]] + contours, minima
        else:
            for child in children:
                cont = [child] + [c for c in contours if child.contains(c)]
                min = [m for m in minima if child.contains(m)]
                yield cont, min

    def _preprocess_contours(self, contours):
        """...

        Step through the input contours list and group those with
        the same level. Returned is a list of tuples, each of which
        contains one or more contours.
        """
        contours = sorted(contours, reverse=True, key=lambda x: x.lvl)
        grouped = [[contours[0],]]
        for cont in contours[1:]:
            if cont.lvl == grouped[-1][0].lvl:
                grouped[-1].append(cont)
            else:
                grouped.append([cont])
        return [tuple(lst) for lst in grouped]

    def path(self):
        """Return the contour as a list of coordinate tuples."""
        return [(x, y) for x, y in np.array(self.contour.boundary.coords.xy).T]

    def path_indices(self):
        """Return the contour as a list of index tuples."""
        raise NotImplementedError

    def center(self):
        """Center of the Depression.

        In case of multiple minima, the center corresponds to the focal point,
        weighted by the relative depths of the minima relative to the inner-
        most contour enclosing all of them.
        """
        minima = [(m.lvl, m.xy) for m in self.minima()]
        ref_depth = self.innermost_shared_contour().lvl
        depths_abs, coords = (np.array(i) for i in list(zip(*minima)))
        depths_rel = ref_depth - depths_abs
        weights = depths_rel / sum(depths_rel)
        weighted_coords = (coords.T * weights).T
        coords_center = np.sum(weighted_coords, axis=0)
        return coords_center

    def radius(self):
        """Radius of a circle with the same area."""
        return np.sqrt(self.area() / np.pi)

    def n_children(self):
        """Return the number of children, i.e. nested Depression clusters."""
        return len(self.children)

    def n_minima(self):
        """Return the number of minima in the cluster.

        These minima belong to the innermost nested child clusters, of which
        there is exactly one in a perfectly nested cluster, and more if
        there are subclusters.
        """
        return len(self.minima())

    def n_contours(self):
        """Return the number of contours.

        The contours refer to the nested child clusters.
        """
        return len(self.contours())

    def is_perfectly_nested(self):
        """Check whether the cluster is perfectly nested.

        This means that is doesn't contain any subclusters, and only one
        single minimum.
        """
        if self.n_children() > 1:
            return False
        for child in self.children:
            if not child.is_perfectly_nested():
                return False
        return True

    def minima(self):
        """Return all minima inside the cluster.

        These minima belong the innermost nested child clusters.
        """
        return [dep.minimum for dep in self.innermost() if dep.minimum is not None]

    def is_innermost(self):
        """Check whether the contour is the innermost of a cluster."""
        return len(self.children) == 0

    def is_outermost(self):
        """Check whether the contour is the outermost of a cluster."""
        return self.parent is None

    def min_contour_depth(self):
        """Minimum depth of the cluster in levels.

        This corresponds to the number of contours from  and including the
        enclosing contour to and including the innermost contour of the
        shallowest sub-cluster.
        """
        return self._contour_depth(lambda c: c.min_contour_depth(), lambda x: min(x))

    def max_contour_depth(self):
        """Maximum depth of the cluster in levels.

        This corresponds to the number of contours from and including the
        enclosing contour to and including the innermost contour of the
        deepest sub-cluster.
        """
        return self._contour_depth(lambda c: c.max_contour_depth(), lambda x: max(x))

    def _contour_depth(self, fct1, fct2):
        """Core helper method for min_contour_depth and max_contour_depth."""
        if len(self.children) == 0:
            return 1
        depths = []
        for child in self.children:
            depths.append(fct1(child))
        return 1 + fct2(depths)

    def min_depth(self):
        """Minimum physical depth of the cluster.

        This corresponds to the level distance between the enclosing contour
        and the shallowest minimum.
        """
        return self._depth(lambda c: c.min_depth(), lambda x: min(x))

    def max_depth(self):
        """Maximum physical depth of the cluster.

        This corresponds to the level distance between the enclosing contour
        and the deepest minimum.
        """
        return self._depth(lambda c: c.max_depth(), lambda x: max(x))

    def _depth(self, fct1, fct2):
        """Core helper method for min_depth and max_depth."""
        if len(self.children) == 0:
            if len(self.minima()) > 0:
                return fct2([(self.lvl - m.lvl) for m in self.minima()])
            return 0.0
        depths = []
        for child in self.children:
            depths.append(fct1(child))
        return (self.lvl - child.lvl) + fct2(depths)

    def count_shared_contours(self, other=None):
        """Count the contours that are shared among sub-/superclusters.

        Whitout another cluster as argument, the number of contours
        shared by all subclusters is returned.

        If a subcluster is given as argument, ...
        If a supercluster is given as argument, ...
        """
        if other is not None:
            raise NotImplementedError  # SR_TODO
        if len(self.children) == 1:
            return 1 + self.children[0].count_shared_contours()
        else:
            return 1

    def shared_depth(self, other=None):
        """Compute the cluster depth before any branching into subclasses.

        This corresponds to the number of shared contours times the contour
        interval.

        If the cluster is perfectly nested and contains one or more
        minima, the distance is measured down to the deepest minimum.
        """
        if other is not None:
            raise NotImplementedError  # SR_TODO
        if len(self.children) == 0:
            if len(self.minima()) > 0:
                return max([(self.lvl - m.lvl) for m in self.minima()])
            return 0.0
        if len(self.children) == 1:
            return (self.lvl - self.children[0].lvl) + self.children[0].shared_depth()
        else:
            return self.lvl - self.children[0].lvl

    def shared_contour_ratio(self, other=None):
        """Compute the ratio of shared contours with sub-/superclusters.

        If no other cluster is given as an argument, shared contours
        are those that are not unique to any subcluster.

        If another cluster is given, then any other subclusters are
        ignored, and all contours not unique to that specific cluster
        are considered shared.

        For perfectly nested clusters, the ratio is always 1.0.
        For two clusters without overlap, is it 0.0.
        """
        if other is not None:
            raise NotImplementedError  # SR_TODO
        return float(self.count_shared_contours()) / float(self.max_contour_depth())

    def shared_depth_ratio(self, other=None):
        """Like shared_contour_ratio, but using the physical depth.

        Physical depth means the level difference between the outermost
        contour and the deepest minimum (i.e. not only to the deepest
        innermost contour).
        """
        if other is not None:
            raise NotImplementedError
        return self.shared_depth() / self.max_depth()

    def innermost(self):
        """Return a list of the innermost nested Depression contours.

        Return a list of Depression objects.
        """
        if self.is_innermost():
            return [self]
        return [cont for child in self.children for cont in child.innermost()]

    def innermost_contours(self):
        """Return a list of the innermost contours of all subclusters.

        Return a list of Contour objects.
        """
        return [dep.contour for dep in self.innermost()]

    def contours(self):
        """Collect all contours of the cluster and all subclusters.

        Return an array of contours, sorted in no particular order
        with respect to the subclusters.
        """
        contours = [self.contour]
        for child in self.children:
            contours += child.contours()
        return contours

    def innermost_shared_contour(self, minima=None):
        """Get the innermost contour shared by all minima.

        If no minima are passed, all minima are used.
        """
        if minima is None:
            minima = self.minima()

        if len(minima) == 0:
            err = "No minima!"
            raise ValueError(err)

        # Get the depressions (i.e. contours) immediately enclosing the minima
        innermost = [
            c for c in self.innermost() if any(c.contour.contains(m) for m in minima)
        ]

        # Step inside-out along the contours until one contains all minima
        def find_innermost_shared_contour_rec(minima, depression):
            if all(depression.contour.contains(m) for m in minima):
                return depression
            if depression.parent is None:
                err = ("No innermost shared contour found for minima: " "{}").format(
                    minima
                )
                raise Exception(err)
            return find_innermost_shared_contour_rec(minima, depression.parent)

        # Start the search from the shallowest minimum/depression
        shallowest = sorted(innermost, key=lambda m: m.lvl)[0]
        depression = find_innermost_shared_contour_rec(minima, shallowest)

        return depression.contour

    def bcc_fraction(self):
        """Compute the fraction of boundary-crossing contours.

        In case of multiple sub-clusters, only the deepest cluster is
        considered.
        """
        innermost = self.innermost()  # SR_TODO select deepest one
        min_lvl = min(innermost, key=lambda e: e.lvl).lvl
        min_cont = list(filter(lambda e: e.lvl == min_lvl, innermost))
        # Compute the bcc_fraction for all deepest inner-most contours
        # (normally just one) by first counting the contours fully inside
        # and partially outside the domain and then computing the fraction
        # of the latter relative to the total number of contours.
        n_in_out = [c._comp_bcc_fraction_rec(self, 0, 0) for c in min_cont]
        return max((float(o) / float(i + o) for i, o in n_in_out))

    def _comp_bcc_fraction_rec(self, self0, ni, no):
        if self.contour.is_boundary_crossing():
            no += 1
        else:
            ni += 1
        if self != self0 and self.parent is not None:
            ni, no = self.parent._comp_bcc_fraction_rec(self0, ni, no)
        return ni, no

    # SR_TODO Refactor this (along with reduce_to_max_n_minima!)
    def reduce_to_max_bcc_fraction(self, bcc_frac_max):
        """Split into subclusters according to a bcc_fraction threshold.

        Return a list containing one or more Depression objects, none of
        which has a higher bcc_fraction than <bcc_frac_max>.
        """
        if self.bcc_fraction() <= bcc_frac_max:
            self.parent = None
            return [self]
        fct = lambda d: d.reduce_to_max_bcc_fraction(bcc_frac_max)
        return [c for child in self.children for c in fct(child)]

    def reduce_to_max_n_minima(self, n_minima):
        """Split into subclusters until none has more then N minima.

        Step inward from the enclosing contour until there are at
        most <n_minima> minima inside the remaining cluster(s).
        Returns a list containing one or more clusters.
        """
        if self.n_minima() <= n_minima:
            # SR_TODO Add test to make sure the 'self.parent=None' is there!
            # SR_TODO Just bug-hunted forever because the line was missing...
            self.parent = None
            return [self]
        fct = lambda d: d.reduce_to_max_n_minima(n_minima)
        return [c for child in self.children for c in fct(child)]


class DepressionFactory:
    def __init__(
        self,
        contours,
        *,
        minima=None,
        maxima=None,
        parent=None,
        contours_nested=False,
        contours_valid=True,
        contours_sorted=False,
        id0=None
    ):

        self.contours = contours
        self.minima = [] if minima is None else minima
        self.maxima = [] if maxima is None else maxima
        self.parent = parent
        self.contours_nested = contours_nested
        self.contours_valid = contours_valid
        self.contours_sorted = contours_sorted

        if id0 is None:
            self.next_id = 0
        else:
            self.next_id = id0

    def run(self, ncont_min=0, bcc_frac=0.5, len_max=None, len_min=None):

        if len(self.contours) == 0:
            return []

        self._ncont_min = ncont_min
        self._bcc_frac = bcc_frac
        self._len_min = len_min
        self._len_max = len_max

        # Identify valid contours and group them
        try:
            contour_clusters = self._cluster_contours()
        except Exception:
            err = (
                "An error occurred during contour clustering. "
                "Probably the objects passed to the DepressionFactory "
                "are not valid Contour objects!"
            )
            raise ValueError(err)

        # Create Depression objects from the grouped contours
        depressions = [
            cont
            for cluster in contour_clusters
            for cont in self._create_depressions(cluster)
        ]

        # Remove unwanted boundary-crossing contours
        depressions = remove_surplus_bcc_clusters(depressions, self._bcc_frac)

        # Remove too shallow clusters
        depressions = remove_shallow_clusters(depressions, self._ncont_min)

        return depressions

    @classmethod
    def _create(
        cls,
        contours,
        minima=None,
        *,
        id0=None,
        maxima=None,
        parent=None,
        contours_nested=False,
        contours_valid=True,
        contours_sorted=False,
        ncont_min=0,
        bcc_frac=0.5,
        len_min=None,
        len_max=None
    ):
        """Method called by Depression.create()."""
        factory = cls(
            contours=contours,
            minima=minima,
            maxima=maxima,
            parent=parent,
            contours_nested=contours_nested,
            contours_valid=contours_valid,
            contours_sorted=contours_sorted,
            id0=id0,
        )

        depressions = factory.run(
            ncont_min=ncont_min, bcc_frac=bcc_frac, len_min=len_min, len_max=len_max
        )

        return depressions

    def _create_depressions(self, cluster):
        """For each child, gather all contained contours and minima."""

        child_contours, remaining_contours = self._find_children_by_level(cluster)

        fct = lambda cont: self._create_depression(cont, remaining_contours)
        children = [fct(cont) for cont in child_contours]

        return children

    def _create_depression(self, cont, contours):

        contained = lambda lst, cont: [e for e in lst if cont.contains(e)]

        child_factory = DepressionFactory(
            contours=contained(contours, cont), minima=None, parent=None
        )

        children = child_factory.run(ncont_min=self._ncont_min, bcc_frac=self._bcc_frac)

        depr = Depression(
            id=self.next_id,
            contour=cont,
            parent=self.parent,
            children=children,
            minima=contained(self.minima, cont),
        )
        self.next_id += 1

        return depr

    def _find_children_by_level(self, contours):
        """Find the outermost contours of the next level.

        Return these direct child contours in one list, and all other
        contours in a second list.

        The children are distinguished by level distance, because in a
        realistic case all direct children are of the same level (there are
        no gaps in the contours). (This must be kept in mind for unit tests.)
        """
        children = []
        remains = contours.copy()
        lvl_children = remains[0].lvl
        # Loop over contours until all have been used up, or until
        # the next contour has a lower level than the "child level".
        while len(remains) > 0:
            try:
                if remains[0].lvl < lvl_children:
                    break
            except TypeError:
                err = ("Level comparison with object {} failed: " "{} < {}").format(
                    repr(remains[0]), remains[0].lvl, lvl_children
                )
                raise TypeError(err)
            children.append(remains.pop(0))
        return children, remains

    def _cluster_contours(self):
        """Identify valid contours and cluster them by overlap.

        For each contour, several criteria are checked. All contours that
        fulfill these criteria are returned as a list.
        """
        if self.contours_valid:
            valid_contours = self.contours
        else:
            # Check all contours for validity in (assumedly) random order.
            # Points not to be contained by any contour are blacklisted.
            is_valid = lambda cont: self._contour_is_valid(cont, blacklist)
            blacklist = []
            # SR_DBG<
            for contour in [c for c in self.contours]:
                try:
                    is_valid(contour)
                except Exception as e:
                    _n = len(contour.boundary.xy[0])
                    log.warning("skip invalid contour (n={})".format(_n))
                    self.contours.remove(contour)
            # SR_DBG>
            valid_contours = [cont for cont in self.contours if is_valid(cont)]

            # Remove contours that contain blacklisted points.
            valid_contours = [
                cont
                for cont in valid_contours
                if not any(cont.contains(pt) for pt in blacklist)
            ]

        # Sort the contours by level in descending order
        if not self.contours_sorted:
            valid_contours = sorted(valid_contours, reverse=True, key=lambda c: c.lvl)

        # Group the contours by overlap into clusters
        if self.contours_nested:
            contour_clusters = [valid_contours]
        else:
            contour_clusters = group_contours_by_overlap(valid_contours)

        return contour_clusters

    def _contour_is_valid(self, contour, blacklist=None):
        """Check whether a contour meets all criteria to be valid.

        Return a bool indicating whether the contour meets all validity
        criteria.

        Optional arguments:
         - blacklist: List to store blacklisted points.

        Blacklisted points are points that should not be contained by any
        contour. If a blacklist is passed, those points should be added to
        the list. Once all contours have been analyzed, and the blacklist
        is thus complete, contours that contain any of these points  be
        removed in a second iteration.
        """

        if not check_length_great_circle(
            contour, lmin=self._len_min, lmax=self._len_max
        ):
            return False

        if not contour.contains_minimum(self.minima):
            if blacklist is not None:
                blacklist.append(contour.representative_point())
            return False

        contained_maxima = contour.contained_maxima(self.maxima)
        if len(contained_maxima) > 0:
            if blacklist is not None:
                blacklist.extend(contained_maxima)
            return False

        return True


class Cyclone(TrackableFeatureBase):

    cls_factory = None
    cls_periodic = None

    def __init__(self, depression, *, id=-1, track=None, event=None, domain=None):

        super().__init__(id=id, track=track, event=event, domain=domain)

        if not isinstance(depression, Depression):
            err = (
                "A Cyclone object must be initialized with a single "
                "Depression object. Try to use the factory method "
                "Cyclone.create()!"
            )
            raise ValueError(err)

        self._depr = depression

        # Aliases to Depression object
        # SR_TODO Consider inheritance in case of too many aliases..!
        self.minima = self._depr.minima
        self.contours = self._depr.contours
        self.min_depth = self._depr.min_depth
        self.max_depth = self._depr.max_depth
        self.n_minima = self._depr.n_minima
        self.n_contours = self._depr.n_contours

    @classmethod
    def create(cls, depression, **kwargs):
        try:
            factory = CycloneFactory(**kwargs)
            return factory.run(depression)
        except Exception as e:
            err = (
                "Cannot create Cyclone objects from argument:\n{}\n" "(error: {})"
            ).format(repr(depression), e)
            raise ValueError(err)

    def as_mean(self):
        """Return a mean-feature containing this feature."""
        return CycloneMean(features=[self])

    def __hash__(self):
        return id(self)

    def __repr__(self):
        cls = self.__class__.__name__
        return "<{cls}[{id0}]: id={id1}, depr={d}>".format(
            cls=cls, id0=id(self), id1=self.id(), d=self._depr
        )

    def __eq__(self, other):
        if not isinstance(other, Cyclone):
            return False

        # SR_TODO improve
        return self.id() == other.id() and self._depr == other._depr

    def __lt__(self, other):
        return self.id() < other.id()

    def as_depression(self):
        return self._depr

    # SR_TODO Turn Depression.contour into Depression.contour()
    @property
    def contour(self):
        return self.as_depression().contour

    def get_info(self, path=None):
        """Return the most important properties as a dict."""
        if path:
            log.debug(
                (
                    "WARNING: {}.get_info: Received argument path={}, but not "
                    "implemented"
                ).format(self.__class__.__name__, path)
            )
        info = OrderedDict()
        info["class"] = self.__class__.__name__
        info["id"] = self.id()
        info["min_depth"] = self.min_depth()
        info["max_depth"] = self.max_depth()
        fct = lambda lvl, xy: OrderedDict([("level", lvl), ("coordinates", xy)])
        info["minima_id"] = [m.id for m in self.minima()]
        info["n_contours"] = self.n_contours()
        info["contours_id"] = [c.id for c in self.contours()]
        return info

    def bcc_fraction(self):
        return self._depr.bcc_fraction()

    # Methods that need to be overridden by the subclass

    def copy(self, path=None, id_=None, event=None, domain=None):
        if not path:
            self.as_polygon().boundary.coords
        if not id_:
            id_ = -1
        return self.__class__(path, id=id_, event=event, domain=domain)

    def as_polygon(self):
        return self._depr.as_polygon()

    def intersection(self, other):
        return self.as_polygon().intersection(other.as_polygon())

    def area(self):
        lon, lat = self.as_polygon().boundary.xy
        return area_lonlat(lon, lat)

    def overlap_area(self, other):

        if isinstance(other, FeatureCombination):
            return sum([self.overlap_area(f) for f in other.features()])

        if not isinstance(other, (Cyclone, CycloneMean)):
            err = (
                "{}.overlap_fraction can only be used with other "
                "instances of Cyclone"
            ).format(self.__class__.__name__)
            # ipython(globals(), locals(), err)
            raise ValueError(err)

        intersection = self.intersection(other)
        try:
            return area_lonlat(*intersection.boundary.xy)
        except NotImplementedError:
            return sum([area_lonlat(*p.boundary.xy) for p in intersection])

    def center(self):
        return self._depr.center()

    def radius(self):
        return np.sqrt(self.area() / np.pi)


class CycloneMean(TrackableFeatureMean):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# SR_TODO Find a way to interit from [Periodic]FeatureFactory
# SR_TODO (or just generally clean up the FeatureFactory "mess")
class CycloneFactory:
    def __init__(
        self,
        cls_default=None,
        min_depth=1.0,
        nmin_max=3,
        thresh_dcc=0.5,
        thresh_tcc=0.7,
        ncont_min=0,
        idealized=False,
        id0=None,
    ):
        """Initialize the Cyclone factory.

        Optional arguments:
         - nmin_max: Maximal number of minima per cyclone (default: 3).
         - thresh_dcc: Contour ratio threshold for DCCs (default: 0.5).
         - thresh_tcc: Contour ratio threshold for TCCs (default: 0.7).

        The default values for thresh_dcc and thresh_tcc follow HC12.

        """
        self._cls_default = cls_default
        self._min_depth = min_depth
        self._nmin_max = nmin_max
        self._thresh_dcc = thresh_dcc
        self._thresh_tcc = thresh_tcc
        self._ncont_min = ncont_min

        if id0 is None:
            self.next_id = 0
        else:
            self.next_id = id0

    def run(self, depression):
        """Create Cyclone object(s) from one Depression object.

        Return a list containing one or more Cyclone-like objects.
        These are subclasses of Cyclone:
         - SingleCenterCyclone
         - DoubleCenterCyclone
         - TripleCenterCyclone

        Arguments:
         - depression: A single or a list of multiple Depression object(s).

        """
        # If a list of Depression objects has been passed,
        # call the run method recursively for all of them.
        if not isinstance(depression, Depression):
            try:
                depressions = Depression.create(depression)
            except TypeError:
                err = "Cannot create a Cyclone from object:\n{}".format(depression)
                raise ValueError(err)

            cyclones = []
            for depr in depressions:
                try:
                    cyclones.extend(self.run(depr))
                except ValueError:
                    continue
            return cyclones

        # Check main argument
        if not isinstance(depression, Depression):
            err = "Cyclone.create currently only supports Depression objects."
            raise NotImplementedError(err)

        # Assert minimal depth
        if depression.max_depth() < self._min_depth:
            err = "Depression object too shallow ({0} < {1})!".format(
                depression.max_depth(), self._min_depth
            )
            raise ValueError(err)

        # No minimum
        if depression.n_minima() == 0:
            err = "Depression object must contain at least one minimum."
            raise ValueError(err)

        # One minimum
        elif depression.n_minima() == 1:
            cyclones = [SingleCenterCyclone(depression, id=self.next_id)]
            self.next_id += 1

        # Two minima
        elif depression.n_minima() == 2:
            if depression.shared_contour_ratio() >= self._thresh_dcc:
                cyclones = [DoubleCenterCyclone(depression, id=self.next_id)]
                self.next_id += 1
            else:
                splits = depression.reduce_to_max_n_minima(1)
                splits = remove_shallow_clusters(splits, self._ncont_min)
                cyclones = self.run(splits)

        # Three minima
        elif depression.n_minima() == 3:
            if depression.shared_contour_ratio() >= self._thresh_dcc:
                cyclones = [TripleCenterCyclone(depression, id=self.next_id)]
                self.next_id += 1
            else:
                splits = depression.reduce_to_max_n_minima(2)
                splits = remove_shallow_clusters(splits, self._ncont_min)
                cyclones = self.run(splits)

        # More than three minima
        else:
            splits = depression.reduce_to_max_n_minima(3)
            splits = remove_shallow_clusters(splits, self._ncont_min)
            cyclones = self.run(splits)

        return cyclones


Cyclone.cls_factory = CycloneFactory


class SingleCenterCyclone(Cyclone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "SCC"


class MultiCenterCyclone(Cyclone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DoubleCenterCyclone(MultiCenterCyclone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.type = "DCC"


class TripleCenterCyclone(MultiCenterCyclone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.type = "TCC"


# TRACKING


class CycloneTracker(FeatureTrackerBase):
    def __init__(
        self,
        min_overlap=None,
        max_area=None,
        f_overlap=0,
        f_area=0,
        threshold=0.5,
        max_dist_abs=-1.0,
        max_dist_rel=2.0,
        allow_missing=False,
        timestep_datetime=False,
        timestep=0,
        delta_timestep=1,
        ids_datetime=False,
        ids_digits=None,
        ids_base=None,
    ):
        super().__init__(
            min_overlap=min_overlap,
            max_area=max_area,
            f_overlap=f_overlap,
            f_area=f_area,
            threshold=threshold,
            max_dist_abs=max_dist_abs,
            max_dist_rel=max_dist_rel,
            allow_missing=allow_missing,
            timestep_datetime=timestep_datetime,
            timestep=timestep,
            delta_timestep=delta_timestep,
            ids_datetime=ids_datetime,
            ids_digits=ids_digits,
            ids_base=ids_base,
        )

    # Methods that need to be overridden by the subclass

    def new_track(self, feature, timestep, *, id=None, event_id=None):
        return CycloneTrack(
            feature=feature, timestep=timestep, id=id, event_id=event_id
        )


class CycloneTrack(FeatureTrackBase):
    def __init__(self, feature=None, id=-1, event_id=None, timestep=0):
        super().__init__(feature=feature, id=id, event_id=event_id, timestep=timestep)


class CycloneTrackFactory(FeatureTrackFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


CycloneTrack.cls_factory = CycloneTrackFactory

# IO


class CycloneIOReaderJson(IOReaderJsonBase):

    section_name_dict = {
        # "INTERNAL": "INTERNAL",
        # "GENERAL": "GENERAL",
        "IDENTIFY": "IDENTIFY"
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skipped_contours = []

    # SR_TODO Remove get_header? Doesn't seem to be used for anything!
    def read_string(self, jstr, get_header=None):

        jdat = json.loads(jstr, object_pairs_hook=OrderedDict)

        if "HEADER" in jdat:
            self._header = jdat["HEADER"]

        data = self.rebuild_cyclones_full(jdat, get_header=get_header)

        return data

    def rebuild_cyclones_full(
        self, jdat, get_header=None, include_tracker_config=False
    ):

        data = OrderedDict()

        if "CONFIG" in jdat:
            data["CONFIG"] = self.rebuild_config(jdat["CONFIG"], include_tracker_config)

        if any("POINTS_" in s for s in jdat.keys()):
            for key in (k for k in jdat.keys() if "POINTS_" in k):
                data[key] = self.rebuild_points(jdat[key])

        if "CONTOURS" in jdat:
            removed_contours = []
            data["CONTOURS"] = self.rebuild_contours(jdat["CONTOURS"], removed_contours)

        if "DEPRESSIONS" in jdat:
            if "CONTOURS" not in data:
                err = "No contours data available to re-build depressions!"
                raise Exception(err)
            data["DEPRESSIONS"] = self.rebuild_depressions(
                jdat["DEPRESSIONS"],
                data["CONTOURS"],
                data["POINTS_MINIMA"],
                removed_contours,
            )

        if "CYCLONES" in jdat:
            if "CONTOURS" not in data:
                err = "No contours data available to re-build cyclones!"
                raise Exception(err)
            if not "POINTS_MINIMA" in data:
                err = "No minima available to re-build cyclones!"
                raise Exception(err)
            data["CYCLONES"] = self.rebuild_cyclones(
                jdat["CYCLONES"], data["CONTOURS"], data["POINTS_MINIMA"]
            )

        return data

    def rebuild_config(self, data, include_tracker=False):
        """Re-construct config dict from a JSON dict."""
        config = {}
        section_names = self.__class__.section_name_dict
        if include_tracker and not "TRACKER" in section_names:
            section_names["TRACKER"] = "TRACKER"
        for name_conf, name_json in section_names.items():
            config[name_conf] = data.get(name_json, {})
        return config

    def rebuild_contours(self, data, removed_contours=None):
        """Re-construct Contour objects from a JSON dict.

        In case of binary output, the paths are read from a binary file.

        Return a list of Contour objects.
        """
        length_threshold, rtol = 10, 1e-4

        if "contour_path_file" in self._header:
            path_file = self._header["contour_path_file"]
            if not os.path.isfile(path_file) and self._data_path != "":
                path_file_tmp = "{}/{}".format(self._data_path, path_file)
                if os.path.isfile(path_file_tmp):
                    path_file = path_file_tmp
                elif self._data_path + "/" not in path_file:
                    path_file = "{p}/{f}".format(
                        p=self._data_path, f=os.path.basename(path_file)
                    )
            reader = CycloneIOReaderBinary()
            paths = reader.read_contour_paths(path_file)
            for cont in data:
                cont["path"] = paths[cont["id"]]

        # SR_TMP 20160503:
        # SR_TMP Only necessary because this check has been implemented in
        # SR_TMP tracking.utilities::Field2D after the identification has been
        # SR_TMP run on the data I'm running the tracking over right now!
        # SR_TMP<
        contours = []
        for c in data:
            path, cid = c["path"], c["id"]
            if len(path) <= length_threshold:
                if path_is_insignificant(path, rtol):
                    if removed_contours is not None:
                        removed_contours.append(cid)
                    log.warning(
                        ("skipping insignificant contour {}:\n" "{}").format(cid, path)
                    )
                    self._skipped_contours.append(cid)
                    continue
            contours.append(Contour(path, level=c["level"], id=c["id"]))
        return contours
        # return [Contour(c["path"], level=c["level"], id=c["id"]) for c in data]
        # SR_TMP>

    def rebuild_points(self, data):
        """Re-construct FieldPoint objects from a JSON dict.

        Return a list of FieldPoint objects.
        """
        point_list = []
        for d in data:
            for key in ("i", "j", "id"):
                if key not in d:
                    d[key] = None
            pt = FieldPoint(
                d["lon"], d["lat"], d["level"], i=d["i"], j=d["j"], id=d["id"]
            )
            point_list.append(pt)
        return point_list

    def rebuild_depressions(self, data, contours, minima, removed_contours=None):
        """Re-construct Depression objects from a JSON dict and contours.

        Along with the JSON dict containing the information about the
        depression objects, the a list containing the contour objects which
        are referenced in the depression data by ID have to be passed in.

        Arguments:
         - data: JSON dict containing the depression data.
         - contours: List of the contour objects referenced in the JSON dict.

        Return a list of Depression objects.
        """
        if removed_contours is None:
            removed_contours = []
        depr_list = []
        for d in data:
            contour_ids = [i for i in d["contours_id"] if i not in removed_contours]
            cont_list = self._collect_contours(contours, contour_ids)
            if len(cont_list) == 0:
                log.warning(
                    ("Cannot rebuild Depression {} (no contours found)").format(d["id"])
                )
                continue

            min_list = self._collect_referenced(minima, d["minima_id"])

            # SR_DBG<
            # depr_list.extend(Depression.create(cont_list, minima=min_list))
            # SR_DBG-
            new_deprs = Depression.create(cont_list, minima=min_list)
            assert len(new_deprs) == 1
            if "id" in d:
                new_deprs[-1]._id = d["id"]
            # SR_DBG>

            depr_list.extend(new_deprs)
        return depr_list

    def rebuild_cyclones(self, jdat, contours, minima):
        """Analogous to rebuild_depressions, see description there!"""

        # Look-up table for Cyclone class types
        cyclone_types = OrderedDict(
            [
                ("SCC", SingleCenterCyclone),
                ("DCC", DoubleCenterCyclone),
                ("TCC", TripleCenterCyclone),
            ]
        )

        # Valid kwargs for Cyclone object initialization
        cyclone_kwargs_keys = ["id", "track", "event", "domain"]

        # Iterate over the cyclone types (SCC, DCC, TCC)
        cyclones = []
        next_id = 0
        used_cyclone_ids = set()
        for cyclone_type_name, sub_data in jdat.items():
            if not sub_data:
                continue

            # Get the Cyclone class type
            cyclone_type = cyclone_types.get(cyclone_type_name, None)
            if cyclone_type is None:
                err = ("Invalid Cyclone type '{}'!\n" "Not found in table:\n{}").format(
                    pformat(cyclone_types)
                )
                raise Exception(err)

            # Loop over the cyclone objects
            for obj_data in sub_data:

                # Rebuild depression object
                depression = self.rebuild_depressions([obj_data], contours, minima)

                # Make sure only a single Depression object has been built
                if len(depression) != 1:
                    if not depression:
                        built = "No Depression object built!"
                    else:
                        built = "Depression objects:\n{}".format(
                            "\n".join([str(d) for d in depression])
                        )
                    err = (
                        "Rebuild of single Depression object failed!\n"
                        "\n JSON jdat:\n{}\n\n{}"
                    ).format(pformat(obj_data), built)
                    raise Exception(err)
                depression = depression[0]

                # Select arguments used to initialize Cyclone object
                cyclone_kwargs = {
                    k: v for k, v in obj_data.items() if k in cyclone_kwargs_keys
                }

                # Make sure the Cyclone gets a unique ID
                if "id" in cyclone_kwargs:
                    used_cyclone_ids.add(cyclone_kwargs["id"])
                else:
                    while next_id in used_cyclone_ids:
                        next_id += 1
                    cyclone_kwargs["id"] = next_id
                    used_cyclone_ids.add(next_id)
                    next_id += 1

                # Rebuild Cyclone object
                try:
                    cyclone = cyclone_type(depression, **cyclone_kwargs)
                except Exception as e:
                    err = (
                        "Rebuild of {} object failed!\n\n JSON jdat:\n{}"
                        "\n\nInitialization arguments:\n{}"
                    ).format(
                        cyclone_type_name, pformat(obj_data), pformat(cyclone_kwargs)
                    )
                    raise Exception(err)

                cyclones.append(cyclone)

        return cyclones

    def _collect_contours(self, cont_list_in, cont_id_list):
        cont_list_out = []
        cont_id_list_orig = [i for i in cont_id_list]
        for c in cont_list_in:
            if c.id in cont_id_list:
                cont_id_list.remove(c.id)
                cont_list_out.append(c)
        for cid in self._skipped_contours:
            if cid in cont_id_list:
                cont_id_list.remove(cid)
        if len(cont_id_list) > 0:
            msg = ("Contour{s} {i} not found among contours: {l}").format(
                s=("s" if len(cont_id_list) > 1 else ""),
                i=", ".join([str(i) for i in cont_id_list]),
                l=", ".join([str(c.id) for c in cont_list_in]),
            )
            raise Exception(msg)
        return cont_list_out

    def _collect_referenced(self, list_in, id_list):
        list_out = []
        for e in list_in:
            if e.id in id_list:
                id_list.remove(e.id)
                list_out.append(e)
        if len(id_list) > 0:
            msg = "Element{s} {i} not found in {l}".format(
                s=("s" if len(id_list) > 1 else ""), i=e.id, l=list_in
            )
            raise Exception(msg)
        return list_out

    def rebuild_mean_cyclones(self, jdat, cyclones):
        """Rebuild mean cyclone objects, which reference cyclone objects."""
        mean_cyclones = []
        for jdat_mean in jdat:

            # Check the class name in order to hard-code it below
            if jdat_mean["class"] != "CycloneMean":
                err = "Unsupported class '{}' for mean cyclone!".format(
                    jdat_mean["class"]
                )
                raise Exception(err)

            features = [c for c in cyclones if c.id() in jdat_mean["features"]]
            mean_cyclone = CycloneMean(features=features, id=jdat_mean["id"])

            mean_cyclones.append(mean_cyclone)

        return mean_cyclones


class CycloneIOReaderBinary(IOReaderBinaryBase):
    def __init__(self):
        super().__init__()

    def read_contour_paths(self, input_file, read_levels=False):
        with np.load(input_file) as f:
            data = {int(k): v for k, v in f.items() if k != "levels"}
            if read_levels:
                if "levels" not in f:
                    err = "'levels' not in file '{}'!".format(input_file)
                    raise Exception(err)
                return data, f["levels"]
            return data

    def read_points_file(self, input_file):
        points = {}
        with np.load(input_file) as f:
            for name, points_data in f.items():
                points[name] = []
                for point_data in points_data.T:
                    lon, lat, lvl, i, j, id = point_data
                    pt = FieldPoint(lon, lat, lvl, i=int(i), j=int(j), id=int(id))
                    points[name].append(pt)
        return points


_WRITERS_JSON = {}


def create_cyclone_writer_json(name):
    if name in _WRITERS_JSON:
        cls = "CycloneIOWriterJson"
        err = "{c} with name {n} already exists!".format(c=cls, n=name)
        raise Exception(err)
    _WRITERS_JSON[name] = CycloneIOWriterJson()
    return _WRITERS_JSON[name]


def get_cyclone_writer_json(name):
    try:
        return _WRITERS_JSON[name]
    except KeyError:
        err = "No writer with name {n}!".format(n=name)
        raise Exception(err)


def delete_cyclone_writer_json(name):
    try:
        del _WRITERS_JSON[name]
    except KeyError:
        err = "No writer with name {n}!".format(n=name)
        raise Exception(err)


class CycloneIOWriterJson(IOWriterJsonBase):

    section_name_dict = {"IDENTIFY": "IDENTIFY"}

    valid_header_param_list = {
        "block_order",
        "contour_path_file",
        "save_paths",
        "path_digits",
    }

    def __init__(self):
        super().__init__()

    def _write_string_method(self, name, register=None):
        new_register = {
            "CONFIG": self.write_string_config,
            "CONTOURS": self.write_string_contours,
            "DEPRESSIONS": self.write_string_depressions,
            "CYCLONES": self.write_string_cyclones,
            "POINTS*": lambda p: self.write_string_points(name[7:], p),
        }
        try:
            register.update(new_register)
        except AttributeError:
            register = new_register
        return super()._write_string_method(name, register)

    # SR_TODO Add tests for this method
    def write_string_points(self, group, points):
        """Write a list of points to JSON.

        Arguments:
         - points: List of points (FieldPoint objects).
        """
        name = "POINTS_{g}".format(g=group)
        log.info("write points {g} to {n}".format(g=group, n=name))
        return self.write_string_objs_info(name, points)

    def write_string_contours(self, contours):
        """Write a list of contours and selected attributes to JSON.

        Arguments:
         - contours: A list of Contour objects.

        Optional arguments:
         - save_paths: Whether to save contour paths to JSON file.
         - path_digits: Limit the number of digits of the path coordinates.

        Note that path_digits is implemented in a hacky way and should not
        be used "operationally"! It has been implemented only to write output
        of reduced file size for testing purposes.
        """
        save_paths = self._header.get("save_paths", True)
        path_digits = self._header.get("path_digits", None)

        name = "CONTOURS"  # SR_TODO
        log.info("write field contours to {n}".format(n=name))
        jstr = self.write_string_objs_info(name, contours, paths=save_paths)

        # Reduce precision of the contour coordinates to reduce output size
        if path_digits:
            if not isinstance(path_digits, int) or not path_digits > 0:
                raise ValueError("invalid path_digits: {}".format(path_digits))
            # Kinda hacky: limit path coordinates to a certain number of
            # digits by simply cutting of the rest (surely not the most
            # efficient implementation, but doesn't matter as it is (for now)
            # not used "operationally" but only to get some test output).
            log.info(
                "HACK: restrict path coordinates to {n} digits".format(n=path_digits)
            )
            jstr_reduced = ""
            rx = re.compile(
                (r"([\[, ]-?[.0-9]{, " + str(path_digits + 1) + r"})[0-9]*")
            )
            for line in jstr.split("\n"):
                if '"path"' in line:
                    line = re.sub(rx, r"\1", line)
                jstr_reduced += line + "\n"
            jstr = jstr_reduced.strip()

        return jstr

    def write_string_depressions(self, depressions):
        """Write the information about Depression objects.

        For every Depression object, write the ID of all contours and the
        coordinates of the associated minima.

        Note that the coordinates of the contours must be written
        separately for the resulting JSON file to be self-contained.

        The formatted JSON data is returned as a string.

        Arguments:
         - depressions: A list of Depression objects.
        """
        name = "DEPRESSIONS"  # SR_TODO
        log.info("write depressions to {n}".format(n=name))
        return self.write_string_objs_info(name, depressions, 2, 4)

    def write_string_cyclones(self, cyclones):
        """Write the information about Cyclone objects.

        For every Cyclones object, write the ID of all contours and the
        coordinates of the associated minima. Additional information
        includes the type etc... <= #SR_TODO

        Note that the coordinates of the contours must be written
        separately for the resulting JSON file to be self-contained.

        The formatted JSON data is returned as a string.

        Arguments:
         - cyclones: A list of Cyclone objects.
        """
        name = "CYCLONES"  # SR_TODO
        log.info("write cyclones to {n}".format(n=name))
        tags = ["SCC", "DCC", "TCC"]
        return self.write_string_objs_info(name, cyclones, tags=tags)

    def add_points(self, group, points):
        self._add_to_cache("POINTS_{}".format(group), points)

    def add_contours(self, contours):
        self._add_to_cache("CONTOURS", contours)

    def add_depressions(self, depressions):
        self._add_to_cache("DEPRESSIONS", depressions)

    def add_cyclones(self, cyclones):
        self._add_to_cache("CYCLONES", cyclones)


class CycloneIOWriterBinary(IOWriterBinaryBase):
    def __init__(self):
        super().__init__()

    def write_contour_path_file(self, file_name, contours, write_levels=False):
        log.info("write {} contour paths to {}".format(len(contours), file_name))
        log.debug(" -> prepare contours for output".format(len(contours), file_name))
        data = {str(contour.id): contour.path() for contour in contours}
        if write_levels:
            log.debug(" -> extract levels from contours".format(len(contours)))
            levels = np.array([[cont.id, cont.lvl] for cont in contours])
            data.update({"levels": levels})
        log.debug(" -> write file")
        np.savez_compressed(file_name, **data)

    def write_points_file(self, file_name, **points):
        data = {}
        for name, pts in points.items():
            data[name] = np.zeros((6, len(pts)))
            for i, p in enumerate(pts):
                data[name][:, i] = p.lon, p.lat, p.lvl, p.i, p.j, p.id
        np.savez_compressed(file_name, **data)


class CycloneTrackIOReaderJson(CycloneIOReaderJson, FeatureTrackIOReaderJson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_string(self, jstr, include_tracker_config=True):
        jdat = json.loads(jstr, object_pairs_hook=OrderedDict)

        if "HEADER" in jdat:
            self._header = jdat["HEADER"]

        data = self.rebuild_cyclones_full(
            jdat, include_tracker_config=include_tracker_config
        )

        try:
            self.minima = data["POINTS_MINIMA"]
        except KeyError:
            err = "Cannot rebuild cyclone tracks (missing minima)!"
            raise Exception(err)
        try:
            self.contours = data["CONTOURS"]
        except KeyError:
            err = "Cannot rebuild cyclone tracks (missing contours)!"
            raise Exception(err)

        data.update(self.rebuild_tracks_full(jdat))

        return data

    def rebuild_features(self, jdat, domain=None):

        # Sort cyclones by type
        jdat_sorted = {"SCC": [], "DCC": [], "TCC": [], "mean": []}
        jdat_mean = []
        for jdat_cyclone in jdat:
            if jdat_cyclone["class"] == "SingleCenterCyclone":
                jdat_sorted["SCC"].append(jdat_cyclone)
            elif jdat_cyclone["class"] == "DoubleCenterCyclone":
                jdat_sorted["DCC"].append(jdat_cyclone)
            elif jdat_cyclone["class"] == "TripleCenterCyclone":
                jdat_sorted["TCC"].append(jdat_cyclone)
            elif jdat_cyclone["class"] == "CycloneMean":
                jdat_mean.append(jdat_cyclone)

        # Rebuild Cyclone objects
        cyclones = self.rebuild_cyclones(jdat_sorted, self.contours, self.minima)

        # Rebuild mean cyclones
        mean_cyclones = self.rebuild_mean_cyclones(jdat_mean, cyclones)

        return cyclones + mean_cyclones


class CycloneTrackIOWriterJson(FeatureTrackIOWriterJson, CycloneIOWriterJson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# SR_TODO Where put this?
def group_contours_by_overlap(contours):
    contour_groups = []
    contour_list = contours.copy()
    while len(contour_list) > 0:
        contour_groups.append([])
        cont_main = contour_list.pop(0)
        contour_groups[-1].append(cont_main)
        remaining = []
        for cont in contour_list:
            if cont_main.contains(cont):
                contour_groups[-1].append(cont)
            else:
                remaining.append(cont)
        contour_list = remaining.copy()
    # SR_TMP<
    # Plot the contour clusters used to initialized the Depression objects
    # (might be useful again for debugging or documentation purposes)
    # from .utilities import DOMAIN_BOUNDARY_CONTOUR
    # from tracking.io import plot_contours
    # for group in contour_groups:
    #    cont = group+[DOMAIN_BOUNDARY_CONTOUR]
    #    name = "cont_{:03d}.png".format(group[0].id)
    #    plot_contours(name, cont)
    # SR_TMP>
    return contour_groups


# SR_TODO Where put this?
def check_length_great_circle(cont, lmin=None, lmax=None):
    """Check the min and/or max great circle length of the contour."""
    invalid = lambda val: (val is None) or (val < 0)
    cmin = True if invalid(lmin) else (cont.length_great_circle() >= lmin)
    cmax = True if invalid(lmax) else (cont.length_great_circle() <= lmax)
    return cmin and cmax


# SR_TODO Refactor this! (copy-paste from remove_shallow_clusters)
def remove_surplus_bcc_clusters(clusters, bcc_frac):
    """From a list of contour clusters, remove those with too many BCCs."""

    if bcc_frac == 1.0:
        return clusters

    good_clusters = []

    for clust in clusters:

        # Retain if not outermost contour
        if clust.parent is not None:
            good_clusters.append(clust)
            continue

        # Retain those where the bcc_fraction is already low
        if clust.bcc_fraction() <= bcc_frac:
            good_clusters.append(clust)
            continue

        # Reduce those in-between by one minimum and try again
        splits = clust.reduce_to_max_bcc_fraction(bcc_frac)
        splits = remove_surplus_bcc_clusters(splits, bcc_frac)
        good_clusters.extend(splits)

    return good_clusters


def remove_shallow_clusters(clusters, ncont_min):
    """From a list of contour clusters, remove those which are too shallow."""

    if ncont_min == 0:
        return clusters

    deep_clusters = []

    for clust in clusters:

        # Retain if not outermost contour
        if clust.parent is not None:
            deep_clusters.append(clust)
            continue

        # Retain those where the minimal depth is already enough
        if clust.min_contour_depth() >= ncont_min:
            deep_clusters.append(clust)
            continue

        # Discard those where not even the maximal depth is enough
        if clust.max_contour_depth() < ncont_min:
            continue

        # Reduce those in-between by one minimum and try again
        splits = clust.reduce_to_max_n_minima(clust.n_minima() - 1)
        splits = remove_shallow_clusters(splits, ncont_min)
        deep_clusters.extend(splits)

    return deep_clusters


if __name__ == "__main__":
    pass
