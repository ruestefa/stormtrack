#!/usr/bin/env python3

# Standard library
import itertools
import json
import logging as log
import pprint
import sys
from datetime import datetime
from datetime import timedelta
from collections import OrderedDict
from collections.abc import Sequence

# Third-party
import numpy as np
import scipy as sp
import shapely.geometry as geo
import shapely.validation

# Local
from ..io_misc import IOReaderJsonBase
from ..io_misc import IOWriterJsonBase
from ..utilities_misc import IDManager
from ..utilities_misc import intersection_objects


__all__ = []


DBG = False


class InfiniteLoopError(Exception):
    pass


class EndOfBranchException(Exception):
    pass


class ContinueIteration(Exception):
    pass


class TrackableFeatureBase:

    id_manager = IDManager()

    cls_factory = None  # FeatureFactory defined below
    cls_periodic = None  # PeriodicTrackableFeatureBase defined below

    def __init__(self, id=None, track=None, event=None, domain=None, attr=None):
        self._track = track
        self._event = event
        self._domain = domain
        self.attr = {} if attr is None else attr

        if id is not None:
            self._id = id
            TrackableFeatureBase.id_manager.blacklist(id)
        else:
            self._id = TrackableFeatureBase.id_manager.next()

    @classmethod
    def factory(cls):
        if cls.cls_periodic is None:
            return cls.cls_factory(cls_default=cls)
        return cls.cls_factory(cls_default=cls, cls_periodic=cls.cls_periodic)

    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def as_mean(self):
        """Return a mean-feature containing this feature."""
        return TrackableFeatureMean(features=[self])

    def is_mean(self):
        return False

    def __repr__(self):
        id0 = id(self)
        id1 = self.id()
        ids = self.id_str()
        event = self.event()
        return ("<{cls}[{id0}]: id={id1}, id_str={ids}, event={e}>").format(
            cls=self.__class__.__name__, id0=id0, id1=id1, ids=ids, e=event
        )

    # SR_TODO proper implementation
    def __eq__(self, other):
        try:
            return self.id() == other.id()
        except AttributeError:
            return False

    def __lt__(self, other):
        return self.id() < other.id()

    def __hash__(self):
        return id(self)

    def is_periodic(self):
        return False

    def get_info(self, path=True):
        jdat = OrderedDict()
        jdat["class"] = self.__class__.__name__
        jdat["id"] = self.id()
        if self.event():
            jdat["event"] = self.event().id()
        jdat["area"] = self.area()
        if path:
            jdat["path"] = self.path()
        jdat["center"] = list(self.center())
        for key, val in self.attr.items():
            jdat["attr_" + key] = val
        return jdat

    def id(self):
        return self._id

    def id_str(self):
        return str(self.id())

    def is_unassigned(self):
        return not self.event()

    def event(self):
        return self._event

    def link_event(self, event):
        self._event = event

    def unlink_event(self):
        self._event = None

    def remove_from_track(self):
        self._track = None

    def path(self):
        """Get the contour path as coordinate tuples."""
        return list(self.as_polygon().boundary.coords)

    def overlap_fraction(self, other):
        """Fraction of area overlap with other feature relative to own size."""
        debug = False
        overlap_area = self.overlap_area(other)
        try:
            other_area = other.area()
        except TypeError:
            other_area = other.area
        fraction = 2 * overlap_area / (self.area() + other_area)
        if debug:
            log.debug(
                "overlap_fraction: 2*{}/({} + {}) = {}".format(
                    overlap_area, self.area(), other_area, fraction
                )
            )
        return fraction

    def area_ratio(self, other):
        """Ratio of area of other feature relative to this feature's.

        If the features have the same area, return 1.
        If this feature is bigger than the other, return a value < 1.
        If the other feature is bigger than this, return a value > 1.
        """
        debug = False
        try:
            other_area = other.area()
        except TypeError:
            other_area = other.area
        ratio = max([self.area(), other_area]) / min([self.area(), other_area])
        if debug:
            log.debug(
                "area_ratio: {}/{} = {}".format(
                    max([self.area(), other.area()]),
                    min([self.area(), other.area()]),
                    ratio,
                )
            )
        return ratio

    def intersects(self, other):
        try:
            return not np.isclose(self.intersection(other).area(), 0)
        except TypeError:
            return not np.isclose(self.intersection(other).area, 0)

    def center_distance(self, other):
        if self._domain and self._domain.is_periodic():
            return self._center_distance_periodic(other)
        return self._distance(self.center(), other.center())

    def _center_distance_periodic(self, other):
        dlon = self._domain.dlon()
        if self.center()[0] <= other.center()[0]:
            left, right = self, other
        else:
            left, right = other, self
        dist0 = self._distance(left.center(), right.center())
        right_moved_center = right.center() - np.array([dlon, 0])
        dist1 = self._distance(left.center(), right_moved_center)
        dist = min([dist0, dist1])
        return dist

    def _distance(self, obj1, obj2):
        return geo.Point(obj1).distance(geo.Point(obj2))

    def is_at_boundary(self, location=None):
        objs = self.domain().boundary_intersection(self, location)
        return any(type(o) is geo.LineString for o in objs)

    def touches(self, other):

        # Check whether the features touch inside the domain
        objs = intersection_objects(self.as_polygon(), other.as_polygon())
        if any(type(o) is geo.LineString for o in objs):
            return True

        def touching_across_periodic_boundary(obj1, obj2):
            lines1 = [
                o
                for o in self.domain().boundary_intersection(self)
                if type(o) is geo.LineString
            ]
            lines2 = [
                o
                for o in self.domain().boundary_intersection(other)
                if type(o) is geo.LineString
            ]
            for line1 in lines1:
                jmin = np.array(line1.xy).T[:, 1].min()
                jmax = np.array(line1.xy).T[:, 1].max()
                for line2 in lines2:
                    if any((jmin <= p[1] <= jmax) for p in line2.xy):
                        return True
            return False

        # For periodic domain, check whether they touch at the boundary
        if self.domain() and self.domain().is_periodic():
            return touching_across_periodic_boundary(self, other)
        return False

    def domain(self):
        if not self._domain:
            err = "Feature not associated with a domain!"
            raise Exception(err)
        return self._domain

    # Abstract methods (need to be defined by subclass)

    def copy(self, *args, **kwargs):
        self._error_not_overridden("copy")

    def as_polygon(self, *args, **kwargs):
        self._error_not_overridden("as_polygon")

    def intersection(self, *args, **kwargs):
        self._error_not_overridden("intersection")

    def area(self, *args, **kwargs):
        self._error_not_overridden("area")

    def overlap_area(self, *args, **kwargs):
        self._error_not_overridden("overlap_area")

    def center(self, *args, **kwargs):
        self._error_not_overridden("center")

    def _error_not_overridden(self, name):
        err = "Method '{}' must be overridden by subclass.".format(name)
        raise NotImplementedError(err)


# SR_TMP SR_TODO put in correct place
class TrackableFeatureMean(TrackableFeatureBase):
    def __init__(self, features, **kwargs):
        super().__init__(**kwargs)
        self._features = features

    def features(self):
        return [f for f in self._features]

    def add_feature(self, feature):
        self._features.append(feature)

    def remove_feature(self, feature):
        self._features.remove(feature)

    def get_info(self, path=True):
        jdat = OrderedDict()
        jdat["class"] = self.__class__.__name__
        jdat["id"] = self.id()
        jdat["features"] = [f.id() for f in self.features()]
        if self.event():
            jdat["event"] = self.event().id()
        jdat["area"] = self.area()
        # if path:
        #    jdat["path"] = self.path()
        jdat["center"] = list(self.center())
        for key, val in self.attr.items():
            jdat["attr_" + key] = val
        return jdat

    def center(self):
        return np.mean([f.center() for f in self._features], axis=0)

    def area(self):
        return np.mean([f.area() for f in self._features])

    def radius(self):
        return np.mean([f.radius() for f in self._features])

    def overlap_area(self, other):
        return np.mean([f.overlap_area(other) for f in self._features])

    def as_polygon(self):
        return geo.MultiPolygon([f.as_polygon() for f in self._features])

    def intersection(self, other):
        if isinstance(other, self.__class__):
            err = "intersection between TrackableFeatureMeans"
            raise NotImplementedError(err)
        return other.intersection(self)

    def n_minima(self):
        return 0

    def minima(self):
        return []

    def is_mean(self):
        return True


class PeriodicTrackableFeatureBase:
    """Trackable feature in a zonally periodic domain."""

    cls_default = TrackableFeatureBase
    cls_mean = TrackableFeatureMean

    def __init__(self, features, domain, **kwargs):

        if not domain.is_periodic():
            err = (
                "Periodic trackable features must be initialized with a "
                "periodic domain!"
            )
            raise ValueError(err)
        self._domain = domain

        # Store the original features separately for each boundary
        self._features_west = [f for f in features if f.is_at_boundary("west")]
        self._features_east = [f for f in features if f.is_at_boundary("east")]
        if not self._features_west or not self._features_east:
            err = (
                "{c}: Invalid set of features:\nALL  : {a}\nWEST : {w}\n" "EAST : {e}"
            ).format(
                c=self.__class__.__name__,
                a=features,
                w=self._features_west,
                e=self._features_east,
            )
            raise ValueError(err)

    @classmethod
    def create(cls, *args, **kwargs):
        try:
            domain = kwargs["domain"]
            id = kwargs["id"]
            ids = kwargs["ids"]
            paths = kwargs["paths"]
            event = kwargs["event"]
        except KeyError as e:
            err = ("Missing mandatory keyword argument to " "{cls}.create: {a}").format(
                cls=cls.__name__, a=e
            )
            raise ValueError(err)
        try:
            features = []
            for fid, path in zip(ids, paths):
                features.append(
                    cls.cls_default.create(path, id=fid, domain=domain, event=None)
                )
            obj = cls(features, id=id, domain=domain, event=event)
            return obj
        except Exception as e:
            err = (
                "Cannot initialize {cls} object!\nARGS   : {a}\n"
                "KWARGS : {k}\nERROR  : {e}"
            ).format(cls=cls.__name__, a=args, k=kwargs, e=e)
            raise ValueError(err)

    @classmethod
    def create_mean(cls, *args, **kwargs):
        return cls.cls_mean(*args, **kwargs)

    def get_info(self, path=True):
        jdat = OrderedDict()
        jdat["class"] = self.__class__.__name__
        jdat["id"] = self.id()
        jdat["ids"] = [feature.id() for feature in self.features()]
        jdat["event"] = self.event().id()
        jdat["area"] = self.area()
        if path:
            jdat["paths"] = self.paths()
        jdat["center"] = list(self.center())
        return jdat

    def features(self, location=None):
        if not location:
            return self._features_west + self._features_east
        elif location == "west":
            return self._features_west
        elif location == "east":
            return self._features_east
        err = "Invalid location {}".format(location)
        raise ValueError(err)

    def is_periodic(self):
        return True

    def as_polygon(self):
        return geo.MultiPolygon([f.as_polygon() for f in self.features()])

    def center(self):
        """Compute the center point of the feature.

        Compute the center point at both the western and the eastern boundary
        by respectively moving the eastern/western features west/east.
        From the two resulting candidate points, the one inside the domain
        is the overall center. If both fall exactly onto a domain boundary,
        the western point is returned.

        Note that this implementation might not suitable anymore in case
        weighted centers should be computed (e.g. for Cyclone objects), but
        for now this is not necessary (as long as no features with weighted
        centers are to be tracked on periodic domains).
        """
        domain = self.domain()
        lon0, lon1, dlon = domain.lon0(), domain.lon1(), domain.dlon()

        paths_west, paths_east = [], []
        for feature in self.features():
            dist_west = feature.center()[0] - lon0
            dist_east = lon1 - feature.center()[0]
            if dist_west <= dist_east:
                # feature at western border
                paths_west.append(feature.path())
                paths_east.append(np.add(feature.path(), [dlon, 0]))
            else:
                # feature at eastern border
                paths_west.append(np.add(feature.path(), [-dlon, 0]))
                paths_east.append(feature.path())

        polys_west = [geo.Polygon(path) for path in paths_west]
        polys_east = [geo.Polygon(path) for path in paths_east]

        center_west = geo.MultiPolygon(polys_west).centroid.coords[0]
        center_east = geo.MultiPolygon(polys_east).centroid.coords[0]

        if self._domain.contains(geo.Point(*center_west)):
            return center_west

        if self._domain.contains(geo.Point(*center_east)):
            return center_east

        if self.domain().lon0() == center_west[0]:
            return center_west

        err = "Error computing center!"
        raise Exception(err)

    # SR_TODO Implement method that returns one path crossing the boundar!
    def path(self):
        err = "{cls}: Use method 'paths' instead of 'path'!".format(
            cls=self.__class__.__name__
        )
        raise NotImplementedError(err)

    def paths(self):
        return [feature.path() for feature in self.features()]

    # Abstract methods (need to be defined by subclass)

    def _error_not_overridden(self, name):
        err = "Method '{}' must be overridden by subclass.".format(name)
        raise NotImplementedError(err)


TrackableFeatureBase.cls_periodic = PeriodicTrackableFeatureBase


class FeatureCombination(TrackableFeatureBase):
    """Combination of multiple trackable features."""

    def __init__(self, features):

        self._features = features

        super().__init__(id=-1, track=None, event=None)

        self._test_feature_validity()

        if all(f.event() for f in features):
            events = [f.event() for f in features]
            self._event = FeatureTrackEventCombination(events)

    def _test_feature_validity(self):

        # Supress Shapely warnings during validity check
        level = log.getLogger().level
        log.getLogger().level = log.ERROR
        poly = self.as_polygon()
        valid = poly.is_valid
        log.getLogger().level = level

        if not valid:
            explanation = shapely.validation.explain_validity(poly)
            err = (
                "Cannot combine features to valid MultiPolygon: {}\n" "Reason: {}"
            ).format(self._features, explanation)
            raise ValueError(err)

    def id_str(self):
        return "+".join([str(feature.id()) for feature in self.features()])

    def is_unassigned(self):
        return not any(f.event() for f in self.features())

    def features(self):
        return [feature for feature in self._features]

    def as_polygon(self):
        polygons = []
        for feature in self.features():
            if feature.is_periodic():
                for subfeature in feature.features():
                    polygons.append(subfeature.as_polygon())
            else:
                poly = feature.as_polygon()
                if isinstance(poly, geo.MultiPolygon):
                    polygons.extend(poly)
                else:
                    polygons.append(poly)
        return geo.MultiPolygon(polygons)

    def intersection(self, other):
        return self.as_polygon().intersection(other)

    def area(self):
        return sum([f.area() for f in self._features])

    def center(self):
        """Area-weighted mean center coordinates."""
        weights = [f.area() / self.area() for f in self._features]
        coords = [w * f.center() for w, f in zip(weights, self._features)]
        return np.sum(np.array(coords), axis=0)

    def radius(self):
        """Area-equivalent radius of all subfeatures combined."""
        return np.sqrt(self.area() / np.pi)


class FeatureFactory:
    def __init__(self, cls_default=None, cls_periodic=None, domain=None, *, id0=0):
        self.cls_default = cls_default
        self.cls_periodic = cls_periodic
        self._domain = domain

        if id0 is not None:
            TrackableFeatureBase._next_id = id0

    def cls(self, cls_str):
        if self.cls_default.__name__ == cls_str:
            return self.cls_default
        if self.cls_periodic.__name__ == cls_str:
            return self.cls_periodic
        err = "Invalid class requested: {}".format(cls_str)
        raise ValueError(err)

    def run(self, features_info, **kwargs_global):
        """Create multiple features.

        For each feature, a dict is passed which must at least contain the
        entry 'contour', which is a list of coordinates. All additional
        entries are passed as initialization arguments of the feature.
        """
        self._features = []
        for feature_info in features_info:
            contour = feature_info["contour"]
            self._check_coords(contour)
            kwargs = {k: v for k, v in feature_info.items() if k != "contour"}
            kwargs.update(kwargs_global)
            new_feature = self.cls_default(contour, domain=self._domain, **kwargs)
            self._features.append(new_feature)
        if self._domain.is_periodic():
            self._merge_periodic_features()
        return self._features

    def cls_ini_args(self):
        """Argument names allowed for class initialization."""
        return ["path", "id", "track", "event", "timestep", "domain", "paths", "ids"]

    def from_class_string(self, cls_str, **kwargs):
        cls = self.cls(cls_str)
        kwargs_ini = {}
        for key in self.cls_ini_args():
            if key in kwargs:
                kwargs_ini[key] = kwargs[key]
        try:
            obj = cls.create(**kwargs_ini)
        except Exception as e:
            err = (
                "Error during initialization of {c} object\n" "ARGS: {a}\nERROR: {e}"
            ).format(c=cls_str, a=kwargs_ini, e=e)
            raise Exception(err)
        else:
            return obj

    def _merge_periodic_features(self):

        # Fetch all features touching the boundary along a line
        bnd_features = [
            f
            for f in self._features
            if f.is_at_boundary("west") or f.is_at_boundary("east")
        ]
        for feature in bnd_features:
            self._features.remove(feature)

        # Group all features touching each other at the periodic boundary
        bnd_features_grouped = set()
        for feature in bnd_features.copy():
            neighbours = []
            for other_feature in bnd_features:
                if other_feature is feature:
                    continue
                if feature.touches(other_feature):
                    if other_feature in bnd_features:
                        bnd_features.remove(other_feature)
                    neighbours.append(other_feature)
            if neighbours:
                neighbours.append(feature)
                if feature in bnd_features:
                    bnd_features.remove(feature)
                bnd_features_grouped.add(frozenset(neighbours))

        # Merge grouped features into periodic features
        # Skip incomplete groups that are subsets of larger groups
        bnd_features_grouped = sorted(
            bnd_features_grouped, key=lambda s: len(s), reverse=True
        )
        bnd_features_joined = []
        for i, group in enumerate(bnd_features_grouped):
            if i == 0 or not any(
                s.issuperset(group) for s in bnd_features_grouped[: i + 1]
            ):
                periodic_feature = self.cls_periodic(group, self._domain)
                bnd_features_joined.append(periodic_feature)

        # Put non-periodic and new periodic features back into features list
        self._features.extend(bnd_features)
        self._features.extend(bnd_features_joined)

    def _check_coords(self, coords):
        """Check whether the input is a list of pairs of numbers."""
        if isinstance(coords, geo.Polygon):
            return
        try:
            for coord in coords:
                assert len(coord) == 2
                for f in coord:
                    1 + f
        except (TypeError, AssertionError):
            err = "Invalid input (not a coordinate lists):\n" + "\n".join(
                [str(c) for c in coords]
            )
            raise ValueError(err)


TrackableFeatureBase.cls_factory = FeatureFactory
TrackableFeatureBase.cls_combination = FeatureCombination


class FeatureTrackEventBase:

    id_manager = IDManager()

    cls_factory = None  # FeatureTrackEventFactory defined below

    def __init__(self, track, feature, timestep, *, id=None):
        self._feature = feature
        self._timestep = timestep

        if id is not None:
            self._id = id
            FeatureTrackEventBase.id_manager.blacklist(id)
        else:
            self._id = FeatureTrackEventBase.id_manager.next()

        self._prev_events = []
        self._next_events = []

        self._track = None
        if track:
            self.link_track(track)

        self.feature().link_event(self)

    @classmethod
    def factory(cls):
        return cls.cls_factory()

    def __repr__(self):
        try:
            fid = self.feature().id_str()
        except Exception:
            fid = "NaN"
        return ("<{cls}[{id0}]: id={id1}, ts={ts}, fid={fid}>").format(
            cls=self.__class__.__name__,
            id0=id(self),
            id1=self.id(),
            ts=self.timestep(),
            fid=fid,
        )

    def __str__(self):
        return "{}({}|{})".format(
            self.__class__.__name__.replace("FeatureTrackEvent", ""),
            self.id(),
            self.feature().id(),
        )

    def __eq__(self, other):
        return (
            self.id() == other.id()
            and self.__class__ == other.__class__
            and self.timestep() == other.timestep()
            and (
                sorted([e.id() for e in self.prev()])
                == sorted([e.id() for e in other.prev()])
            )
            and (
                sorted([e.id() for e in self.next()])
                == sorted([e.id() for e in other.next()])
            )
        )

    def __lt__(self, other):
        return self.id() < other.id()

    def __hash__(self):
        return id(self)

    def id(self):
        return self._id

    def get_info(self):

        # SR_TMP<
        try:
            timestep = int(self.timestep().strftime("%Y%m%d%H"))
        except AttributeError:
            timestep = self.timestep()
        # SR_TMP>

        jdat = OrderedDict()
        jdat["class"] = self.__class__.__name__
        jdat["id"] = self.id()
        jdat["track"] = self.track().id()
        jdat["feature"] = self.feature().id()
        jdat["timestep"] = timestep
        jdat["prev"] = sorted([e.id() for e in self.prev()])
        jdat["next"] = sorted([e.id() for e in self.next()])
        return jdat

    def link_track(self, track):
        self._track = track
        track.update_registers(event_add=self)

    def track(self):
        return self._track

    def feature(self):
        return self._feature

    def continued(self):
        return bool(self.next())

    def prev(self):
        return [e for e in self._prev_events]

    def next(self):
        return [f for f in self._next_events]

    def successors(self, direction):
        return list(self.next() if direction > 0 else self.prev())

    # SR_TODO Get rid of this, use events_all_branches!
    def events_branch(self, direction=1, distance=None):
        """Get all successor events, either forward or backward, in one list.

        Mergings and splittings are not considered.
        """
        events = [self]
        for event in self.successors(direction):
            if distance is not None:
                if distance == 0:
                    break
                distance -= 1
            events.extend(event.events_branch(direction=direction, distance=distance))
        return events

    # SR_TODO Give this method a more intuitive name!
    def events_all_branches(
        self,
        direction=None,
        distance=None,
        branch_back=False,
        exclude_event=None,
        stop_condition=None,
        start_event_inclusive=True,
        stop_event_inclusive=True,
        stop_condition_force=False,
        soft_stop=False,
    ):
        """Collect all events of all possible branches.

        Return a list of branch objects in either one or both directions.

        Events might be part of multiple lists, for instance those before/at
        splittings, or at/after mergings (relative to the direction).

        Optional arguments:
         - direction: Forward (+1) or backward (-1).
         - distance: Maximum length of branches.
         - branch_back: Continue branch backwards at mergings.
         - exclude_event: Discard branches containing this event.
         - start_event_inclusive: Include the start event in the branch.
         - stop_event_inclusive: Include the stop event in the branch.
         - stop_condition: Condition for an event to stop a branch.
         - stop_condition_force: Foce condition upon all branch ends.
         - soft_stop: Continue a second branch of stop is reached.

        Stop condition:
         All branches are terminated at (direction-relative) branch ends.
         An additional condition for events can be given as a lambda.
         Example: Branches might be stopped at merging or splitting events.

        Soft stop:
         If an event fulfils the stop_condition, the branch is terminated.
         If soft stop is active, a copy of the branch is continued.
         Example: If a branch contains one merging, and being a merging event
         is the stop condition, one branch is terminated at the merging,
         another (which is a superset of the terminated branch) is continued
         along the branch as if there had been no merging.
        """
        collector = EventBranchCollector(
            distance,
            branch_back,
            start_event_inclusive,
            stop_event_inclusive,
            stop_condition,
            stop_condition_force,
            soft_stop,
        )

        if direction:
            branches = collector.run(self, direction)
        else:
            forward, backward = 1, -1
            branches = collector.run(self, forward) + collector.run(self, backward)

        if exclude_event:
            branches = [b for b in branches if exclude_event not in b]

        return branches

    def is_directly_related(self, other, direction=1):
        """Check whether the event is directly related to another event."""
        return other in self.events_branch(direction)

    def count_events_branch(self, *, direction=1):
        """Count all successor events, either forward or backward."""
        return len(self.events_branch(direction=direction))

    def link_forward(self, event):
        if event in self.next():
            err = "event {e1} already linked to event {e0}: {n}".format(
                e1=event, e0=self, n=self.next()
            )
            raise Exception(err)
        self._next_events.append(event)
        if not self in event.prev():
            event._prev_events.append(self)

    def unlink_forward(self, event, check=True):
        if check and event not in self.next():
            err = "event {e0} not linked forward to event {e1}: {n}".format(
                e1=self, e0=event, n=self.next()
            )
            raise ValueError(err)
        if event in self._next_events:
            self._next_events.remove(event)
        if self in event._prev_events:
            event._prev_events.remove(self)

    def link(self, *, prev, next):
        for event in prev:
            event.link_forward(self)
        for event in next:
            self.link_forward(event)

    def unlink(self):
        for prev_event in self.prev():
            prev_event.unlink_forward(self)
        for next_event in self.next():
            self.unlink_forward(next_event)

    def is_merging(self):
        return False

    def is_splitting(self):
        return False

    def is_start(self):
        return False

    def is_genesis(self):
        return False

    def is_end(self):
        return False

    def is_lysis(self):
        return False

    def is_stop(self):
        return False

    def is_isolated(self):
        return False

    def is_enter(self):
        return False

    def is_exit(self):
        return False

    def is_dummy(self):
        return False

    def is_continuation(self, direction=None):
        if direction is None:
            return False
        else:
            return len(self.successors(direction)) == 1

    def is_head(self):
        return not self.next() and not self.is_end()

    def is_branch_end(self, direction):
        if direction > 0:
            return self.is_end()
        elif direction < 0:
            return self.is_start()
        return ValueError("Argument 'direction' must not be zero")

    # SR_TODO Rename method to sth better and distinguish between
    # SR_TODO "opening" and "closing" branchings (i.e. the direction-
    # SR_TODO independent equivalents of splittings and mergings).
    # SR_TODO Then replace all "is_branching(direction=-direction)"
    # SR_TODO accordingly.
    def is_branching(self, direction=None):
        if not direction:
            return self.is_splitting() or self.is_merging()
        if direction > 0:
            return self.is_splitting()
        elif direction < 0:
            return self.is_merging()
        return ValueError("Argument 'direction' must not be zero")

    def is_pure_splitting(self):
        return self.is_splitting() and not self.is_merging()

    def is_pure_merging(self):
        return self.is_merging() and not self.is_splitting()

    def is_merging_splitting(self):
        return self.is_merging and self.is_splitting()

    def timestep(self):
        return self._timestep

    def _as_cls(self, cls, id):
        if id is None:
            id = self.id()
        return cls(self.track(), self.feature(), self.timestep(), id=id)

    def as_genesis(self, id=None):
        if self.is_splitting():
            return self._as_cls(FeatureTrackEventGenesisSplitting, id)
        if self.is_lysis():
            return self._as_cls(FeatureTrackEventIsolated, id)
        return self._as_cls(FeatureTrackEventGenesis, id)

    def as_lysis(self, id=None):
        if self.is_merging():
            return self._as_cls(FeatureTrackEventMergingLysis, id)
        if self.is_genesis():
            return self._as_cls(FeatureTrackEventIsolated, id)
        return self._as_cls(FeatureTrackEventLysis, id)

    def as_stop(self, id=None):
        if self.is_merging():
            return self._as_cls(FeatureTrackEventMergingStop, id)
        return self._as_cls(FeatureTrackEventStop, id)

    def as_enter(self, id=None):
        if self.is_splitting():
            return self._as_cls(FeatureTrackEventEnterSplitting, id)
        if self.is_lysis():
            return self._as_cls(FeatureTrackEventIsolated, id)
        return self._as_cls(FeatureTrackEventEnter, id)

    def as_exit(self, id=None):
        if self.is_merging():
            return self._as_cls(FeatureTrackEventMergingExit, id)
        return self._as_cls(FeatureTrackEventExit, id)

    def as_isolated(self, id=None):
        return self._as_cls(FeatureTrackEventIsolated, id)

    def as_splitting(self, id=None):
        return self._as_cls(FeatureTrackEventSplitting, id)


class FeatureTrackEventStart(FeatureTrackEventBase):
    def is_start(self):
        return True

    def as_splitting(self, id=None):
        return self._as_cls(FeatureTrackEventGenesisSplitting, id)


class FeatureTrackEventContinuation(FeatureTrackEventBase):
    def is_continuation(self, direction=None):
        return True


class FeatureTrackEventMerging(FeatureTrackEventBase):
    def is_merging(self):
        return True

    def as_splitting(self, id=None):
        return self._as_cls(FeatureTrackEventMergingSplitting, id)

    def as_non_merging(self, id=None):
        return self._as_cls(FeatureTrackEventContinuation, id)


class FeatureTrackEventSplitting(FeatureTrackEventBase):
    def is_splitting(self):
        return True

    def as_merging(self, id=None):
        return self._as_cls(FeatureTrackEventMergingSplitting, id)

    def as_non_splitting(self, id=None):
        return self._as_cls(FeatureTrackEventContinuation, id)


class FeatureTrackEventEnd(FeatureTrackEventBase):
    def is_end(self):
        return True


class FeatureTrackEventGenesis(FeatureTrackEventStart):
    def is_genesis(self):
        return True


class FeatureTrackEventLysis(FeatureTrackEventEnd):
    def is_lysis(self):
        return True


class FeatureTrackEventStop(FeatureTrackEventEnd):
    def is_stop(self):
        return True


class FeatureTrackEventEnter(FeatureTrackEventStart):
    def is_enter(self):
        return True


class FeatureTrackEventExit(FeatureTrackEventEnd):
    def is_exit(self):
        return True


class FeatureTrackEventMergingSplitting(
    FeatureTrackEventMerging, FeatureTrackEventSplitting,
):
    def as_non_merging(self, id=None):
        return self._as_cls(FeatureTrackEventSplitting, id)

    def as_non_splitting(self, id=None):
        return self._as_cls(FeatureTrackEventMerging, id)


class FeatureTrackEventGenesisSplitting(
    FeatureTrackEventGenesis, FeatureTrackEventSplitting,
):
    def as_non_splitting(self, id=None):
        return self._as_cls(FeatureTrackEventGenesis, id)


class FeatureTrackEventEnterSplitting(
    FeatureTrackEventEnter, FeatureTrackEventSplitting,
):
    def as_non_splitting(self, id=None):
        return self._as_cls(FeatureTrackEventEnter, id)


class FeatureTrackEventMergingLysis(FeatureTrackEventMerging, FeatureTrackEventLysis):
    def as_non_merging(self, id=None):
        return self._as_cls(FeatureTrackEventLysis, id)


class FeatureTrackEventMergingLysis(FeatureTrackEventMerging, FeatureTrackEventExit):
    def as_non_merging(self, id=None):
        return self._as_cls(FeatureTrackEventExit, id)


class FeatureTrackEventMergingStop(FeatureTrackEventMerging, FeatureTrackEventStop):
    def as_non_merging(self, id=None):
        return self._as_cls(FeatureTrackEventStop, id)


class FeatureTrackEventMergingExit(FeatureTrackEventMerging, FeatureTrackEventExit):
    def as_non_merging(self, id=None):
        return self._as_cls(FeatureTrackEventExit, id)


class FeatureTrackEventIsolated(FeatureTrackEventGenesis, FeatureTrackEventEnd):
    def is_isolated(self):
        return True


class FeatureTrackEventIsolatedLysis(FeatureTrackEventIsolated, FeatureTrackEventLysis):
    pass


class FeatureTrackEventIsolatedStop(FeatureTrackEventIsolated, FeatureTrackEventStop):
    pass


class FeatureTrackEventDummyBase(FeatureTrackEventBase):
    def is_dummy(self):
        return True

    def as_continuation(self, id=None):
        return self._as_cls(FeatureTrackEventDummyContinuation, id)

    def as_merging(self, id=None):
        return self._as_cls(FeatureTrackEventDummyMerging, id)

    def as_splitting(self, id=None):
        return self._as_cls(FeatureTrackEventDummySplitting, id)

    def as_stop(self, id=None):
        err = "Cannot turn Dummy event into Stop event! Finish track earlier!"
        raise Exception(err)


class FeatureTrackEventDummyContinuation(
    FeatureTrackEventDummyBase, FeatureTrackEventContinuation,
):
    pass


class FeatureTrackEventDummyMerging(
    FeatureTrackEventDummyBase, FeatureTrackEventMerging,
):
    pass


class FeatureTrackEventDummySplitting(
    FeatureTrackEventDummyBase, FeatureTrackEventSplitting,
):
    pass


class FeatureTrackEventCombination:
    def __init__(self, events):
        self._check_event_compatibility(events)
        self._events = events
        self._timestep = events[0].timestep()

    def _check_event_compatibility(self, events):
        timesteps = set(event.timestep() for event in events)
        if len(timesteps) > 1:
            err = (
                "Events not compatible because they have different " "timesteps: {e}"
            ).format(e=events)
            raise ValueError(err)

    def timestep(self):
        return self._timestep

    def continued(self):
        return False


class FeatureTrackEventFactory:
    def __init__(self):
        pass

    def cls(self, cls_str):
        try:
            return globals()[cls_str]
        except KeyError:
            err = "Invalid class string: {}".format(cls_str)
            raise ValueError(err)

    def from_class_string(self, cls_str, **kwargs):
        return self.cls(cls_str)(**kwargs)

    def continuation(self, *args, **kwargs):
        return FeatureTrackEventContinuation(*args, **kwargs)

    def genesis(self, *args, **kwargs):
        return FeatureTrackEventGenesis(*args, **kwargs)

    def lysis(self, *args, **kwargs):
        return FeatureTrackEventLysis(*args, **kwargs)

    def merging(self, *args, **kwargs):
        return FeatureTrackEventMerging(*args, **kwargs)

    def splitting(self, *args, **kwargs):
        return FeatureTrackEventSplitting(*args, **kwargs)


FeatureTrackEventBase.cls_factory = FeatureTrackEventFactory


# SR_TODO Add some tests for this (as it's getting more and more features)
class EventBranchCollector:
    """Collect all events of all possible branches.

    Return a list of event lists.

    Events might be part of multiple lists, for instance those before/at
    splittings, or at/after mergings (relative to the direction).

    """

    def __init__(
        self,
        distance,
        branch_back,
        start_event_inclusive,
        stop_event_inclusive,
        stop_condition,
        stop_condition_force,
        soft_stop,
    ):
        self.distance = distance
        self.branch_back = branch_back
        self.start_event_inclusive = start_event_inclusive
        self.stop_event_inclusive = stop_event_inclusive
        self.stop_condition = stop_condition
        self.stop_condition_force = stop_condition_force
        self.soft_stop = soft_stop

    def run(self, event, direction, distance=None):
        self.direction = direction
        self.branches = []
        # SR_TODO Consider renaming the start event to reference event
        self.event_start = event
        self._run_rec(event, passed_distance=1)
        return self.branches

    def _run_rec(
        self, event, branch=None, passed_distance=0, direction=None, start_event=None
    ):

        if direction is None:
            direction = self.direction
        if start_event is None:
            start_event = self.event_start
        if branch is None:
            branch = []

        if event in branch:
            return

        # SR_DBG<
        # if DBG: log.debug("{} RUN_REC {:>10}|{:<10} {:>10}|{:<10} {:>10} {}".format(
        #        ">>>" if direction > 0 else "<<<",
        #        event.id(), event.feature().id(),
        #        start_event.id(), start_event.feature().id(),
        #        passed_distance,
        #        "/".join(["{}|{}".format(e.id(), e.feature().id())
        #                                            for e in branch])))
        # SR_DBG>

        # Append the event to the branch, unless it's the start event
        # and the start event should not be included in the branch
        if not (not self.start_event_inclusive and event == start_event):
            branch.append(event)

        # Check whether max. distance along the branch has been reached
        if self.distance is not None and passed_distance >= self.distance:
            self._finish_branch(event, branch, passed_distance)
            return

        # Check whether a stop has been reached
        if self.stop_condition and self.stop_condition(event, passed_distance):
            new_branch = None
            if self.stop_event_inclusive:
                if len(branch) > 0:
                    new_branch = FeatureTrackBranch(branch)
            else:
                if len(branch) > 1:
                    new_branch = FeatureTrackBranch(branch[:-1])
            if new_branch:
                self.branches.append(new_branch)
            if not self.soft_stop:
                return

        # Check for the end of the branch
        if event.is_branch_end(direction):
            self._finish_branch(event, branch, passed_distance)
            return

        # Check if the event branches against our direction (e.g. is a
        # merging if we're iterating in forward direction), and follow
        # the unrelated branch(es) back if we're supposed to do so
        if self.branch_back and event.is_branching(-direction):
            for successor in event.successors(-direction):
                if not successor.is_directly_related(start_event, -direction):
                    # SR_DBG<
                    # if DBG: log.debug(" -> RUN_REC")
                    # SR_DBG>
                    self._run_rec(
                        event=successor,
                        branch=branch.copy(),
                        passed_distance=passed_distance + 1,
                        direction=-direction,
                        start_event=event,
                    )

        # Continue forward
        for successor in event.successors(direction):
            # SR_DBG<
            # if DBG: log.debug("    SUCCESSOR 1: {:>}|{:<}".format(
            #        successor.id(), successor.feature().id()))
            # if DBG: log.debug(" -> RUN_REC")
            # SR_DBG>
            self._run_rec(
                event=successor,
                branch=branch.copy(),
                passed_distance=passed_distance + 1,
                direction=direction,
                start_event=start_event,
            )

    def _finish_branch(self, event, branch, passed_distance):
        # if DBG: log.debug("END OF BRANCH REACHED")

        # Check whether the branch fulfills the stop condition if it has to
        if branch and self.stop_condition and self.stop_condition_force:

            # If the branch doesn't fulfill the stop condition, discard it
            if not self.stop_condition(branch[-1], passed_distance):
                # if DBG: log.debug("  => NOT CONDITION") #SR_DBG
                return

            if not self.stop_event_inclusive:
                branch = branch[:-1]

            # Make sure this branch is unique
            elif branch in self.branches:
                # if DBG: log.debug("  => NOT UNIQUE") #SR_DBG
                return

        if len(branch) > 0:
            new_branch = FeatureTrackBranch(branch)
            self.branches.append(new_branch)
        # SR_DBG<
        # if DBG: log.debug("    BRANCH END: {}".format(
        #        "/".join(["{}|{}".format(e.id(), e.feature().id())
        #                                            for e in branch])))
        # SR_DBG>


class FeatureTrackBase:

    cls_factory = None  # FeatureTrackFactory defined below

    id_manager = IDManager()

    def __init__(
        self,
        feature=None,
        id=None,
        timestep=0,
        delta_timestep=1,
        event_id=None,
        event_factory=None,
    ):

        self._timestep = timestep
        self._delta_timestep = delta_timestep

        if id is not None:
            self._id = id
            FeatureTrackBase.id_manager.blacklist(id)
        else:
            self._id = FeatureTrackBase.id_manager.next()

        self._starts = set()
        self._heads = set()
        self._ends = set()

        self._events_ts = {}

        if feature is not None:
            factory = FeatureTrackEventFactory()
            event = factory.genesis(self, feature, timestep, id=event_id)
            # SR_TODO< Cyclone-specific!!!
            try:
                if feature.bcc_fraction() > 0:
                    event = event.as_start()
            except AttributeError:
                pass
            # SR_TODO>
            self.update_starts(event_add=event)

    @classmethod
    def factory(cls):
        return cls.cls_factory(cls_default=cls)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return (
            "<{cls}[{id0}]: id={id1:8}, dur={dur:4}, n={n:4}, " "ts={ts:4}>"
        ).format(
            cls=self.__class__.__name__,
            id0=id(self),
            id1=self.id(),
            n=self.n(),
            ts=self.timestep(),
            dur=self.duration(),
        )

    def __eq__(self, other):
        try:
            return (
                self.id() == other.id()
                and self.features() == other.features()
                and self.events() == other.events()
                and self.starts() == other.starts()
                and self.heads() == other.heads()
                and self.ends() == other.ends()
                and self.timestep() == other.timestep()
            )
        except AttributeError:
            return False

    def __lt__(self, other):
        return self.id() < other.id()

    def __iter__(self):
        """Iterate over all events in no particular order."""
        for events in self._events_ts.values():
            for event in events:
                yield event

    def iter_ts(self):
        for ts, events in sorted(self._events_ts.items()):
            yield list(events)

    def get_info(self):
        jdat = OrderedDict()
        for key, val in [
            ("id", self.id()),
            ("n", self.n()),
            ("duration", self.duration()),
        ]:
            jdat[key] = val
        for key, objs in [
            ("starts", self.starts()),
            ("ends", self.ends()),
            ("events", self.events()),
        ]:
            jdat[key] = sorted([obj.id() for obj in objs])
        if len(self.heads()) > 0:
            jdat["heads"] = sorted([obj.id() for obj in self.heads()])
        return jdat

    def timesteps(self, format=None):
        dts, ts_end = self.delta_timestep(), self.ts_end()
        if format is None:
            fct = lambda ts: ts
        elif format == "str":
            if isinstance(dts, timedelta):
                fct = lambda ts: ts.strftime("%Y%m%d%H")
            else:
                fct = lambda ts: str(ts)
        else:
            err = "timesteps(format='{}')".format(format)
            raise NotImplementedError(err)
        timesteps = []
        ts = self.ts_start()
        while ts <= ts_end:
            timesteps.append(fct(ts))
            ts += dts
        return timesteps

    def id(self):
        return self._id

    def n(self):
        return len(list(iter(self)))

    def duration(self):
        """Total lengths in hours."""
        # To retrieve an arbitrary element from a set 'foo': e=next(iter(foo))
        try:
            # Get one event each from the earliest and latest timesteps
            event0 = next(iter(self._events_ts[min(self._events_ts.keys())]))
            event1 = next(iter(self._events_ts[max(self._events_ts.keys())]))
        except StopIteration:
            raise Exception("DURATION")
        delta_timestep = event1.timestep() - event0.timestep()
        try:
            return delta_timestep.total_seconds() / 3600
        except AttributeError:
            return delta_timestep / self._delta_timestep

    def timestep(self):
        try:
            return max([e.timestep() for e in self.events()])
        except ValueError:
            return -1

    def delta_timestep(self):
        return self._delta_timestep

    def ts_end(self):
        return self.timestep()

    def ts_start(self):
        if isinstance(self.timestep(), datetime):
            ts = self.timestep() - timedelta(hours=self.duration())
            return ts
        return self.timestep() - self.duration()

    def is_finished(self):
        return len(self.heads()) == 0

    def events(self, direction=None):
        events = [event for event in self]
        if not direction:
            return events
        if direction > 0:
            return sorted(events, key=lambda e: e.timestep())
        return sorted(events, key=lambda e: e.timestep(), reverse=True)

    def features(self):
        """Return all features in an unsorted list."""
        features = []
        for feature in (event.feature() for event in self):
            features.append(feature)
            if feature.is_periodic():
                features.extend(feature.features())
        return features

    def events_ts(self, ts=None):
        if ts is None:
            return [list(evs) for ts, evs in sorted(self._events_ts.items())]
        try:
            return [e for e in self._events_ts[ts]]
        except KeyError:
            err = (
                "No features found in track for timestep {}\n"
                "Track: {}\nFeatures:\n{}"
            ).format(
                ts,
                self,
                "\n".join(
                    [
                        "{:5} : {}".format(i, ", ".join([str(e) for e in es]))
                        for i, es in sorted(self._events_ts.items())
                    ]
                ),
            )
            raise ValueError(err)

    def features_ts(self, ts, format=None, dt_str_format="%Y%m%d%H"):
        if format is None:
            fct = lambda ts: ts
        elif format == "str":
            if isinstance(self.delta_timestep(), timedelta):
                fct = lambda ts: datetime.strptime(ts, dt_str_format)
            else:
                fct = lambda ts: int(ts)
        else:
            err = "features_ts(ts, format='{}')".format(format)
            raise NotImplementedError(err)
        return [e.feature() for e in self.events_ts(fct(ts))]

    def linear_segments(self, unique_branchings=True):
        """Return all linear segments of the track (as branch objects).

        These are the linear event chains between branching events.
        Like the events, the branches are linked to each other.
        """

        passed_mergings = {}

        def _linear_segments_rec(event, *, prev=None):
            def stop_condition(event, passed_distance):

                if event.is_splitting():
                    return True

                if event.next() and event.next()[0].is_merging():
                    return True

                return False

            branches = event.events_all_branches(
                direction=1, stop_condition=stop_condition
            )

            for branch in branches.copy():
                if prev:
                    prev.link_forward(branch)

                events_next = branch.continuation_next()

                if not events_next:
                    continue

                for event_next in events_next:
                    if event_next.is_merging():
                        if event_next in passed_mergings:
                            for segment in segments:
                                if event_next in segment:
                                    branch.link_forward(segment)
                            continue
                        passed_mergings[event_next] = branch
                    _linear_segments_rec(event_next, prev=branch)

            # Extend global list segments (defined in the enclosing function)
            segments.extend(branches)

        segments = []
        for event in self.starts():
            _linear_segments_rec(event)

        for segment in segments:
            segment.set_track(self)

        # SR_TODO< Better implementation necessary (e.g. somewhere above)?
        if not unique_branchings:
            for segment in segments:
                first_event = segment._events[0]
                if first_event.is_merging():
                    for prev_segment in segment.prev():
                        if prev_segment._events[-1] != first_event:
                            prev_segment._events.append(first_event)
                last_event = segment._events[-1]
                if last_event.is_splitting():
                    for next_segment in segment.next():
                        if next_segment._events[0] != last_event:
                            next_segment._events.insert(0, last_event)
        # SR_TODO>

        return segments

    def mean_width(self):
        return np.mean([len(e) for e in self.events_ts()])

    def max_width(self):
        return np.max([len(e) for e in self.events_ts()])

    def starts(self):
        return list(self._starts)

    def heads(self):
        return [head for head in self._heads]

    def ends(self):
        return list(self._ends)

    def replace_event(self, *, event_del, event_add):
        event_add.link(prev=event_del.prev(), next=event_del.next())
        event_del.unlink()
        self.update_registers(event_add=event_add, event_del=event_del)

    def remove_event(self, event_del):
        event_del.feature().unlink_event()
        event_del.unlink()
        self.update_registers(event_del=event_del)

    def update_registers(self, *, event_add=None, event_del=None):
        if event_add and event_del:
            if event_add.timestep() != event_del.timestep():
                err = "Only events with the same timestep can replace " "each other"
                raise ValueError(err)
        self.update_events_ts(event_add=event_add, event_del=event_del)
        self.update_starts(event_add=event_add, event_del=event_del)
        self.update_heads(event_add=event_add, event_del=event_del)
        self.update_ends(event_add=event_add, event_del=event_del)

    def update_events_ts(self, *, event_add=None, event_del=None):

        # Make sure at least one event has been passed
        if not event_add and not event_del:
            err = "Neither event to add nor to delete passed!"
            raise ValueError(err)

        # If two events have been passed, make sure their timesteps match
        if event_add and event_del and event_add.timestep() != event_del.timestep():
            err = "Timesteps of events differs: {} {}".format(event_add, event_del)
            raise ValueError(err)
        ts = event_add.timestep() if event_add else event_del.timestep()

        # Update the register corresponding to the timestep
        if ts not in self._events_ts:
            self._events_ts[ts] = set()
        self._events_ts[ts] = self._update_register(
            "events", self._events_ts[ts], event_add=event_add, event_del=event_del
        )

        # Remove register if it's empty
        if not self._events_ts[ts]:
            del self._events_ts[ts]

    def update_starts(self, *, event_add=None, event_del=None):
        """Add or remove an event from the list of starts."""
        self._starts = self._update_register(
            "starts",
            self._starts,
            event_add=event_add,
            event_del=event_del,
            condition=lambda e: e.is_start(),
        )

    def update_heads(self, *, event_add=None, event_del=None):
        """Add or event_del an event from the list of heads."""
        self._heads = self._update_register(
            "heads",
            self._heads,
            event_add=event_add,
            event_del=event_del,
            condition=lambda e: e.is_head(),
        )

    def update_ends(self, *, event_add=None, event_del=None):
        """Add or event_del an event from the list of ends."""
        self._ends = self._update_register(
            "ends",
            self._ends,
            event_add=event_add,
            event_del=event_del,
            condition=lambda e: e.is_end(),
        )

    def _update_register(
        self, name, register, *, event_add=None, event_del=None, condition=None
    ):
        if event_del:
            if event_del in register:
                register.remove(event_del)
        if event_add:
            register.add(event_add)
        if condition:
            register = {event for event in register if condition(event)}
        return register

    def merge(self, others):
        """Merge other tracks into this track."""
        starts = [start for track in others for start in track.starts()]
        heads = [start for track in others for start in track.heads()]
        ends = [end for track in others for end in track.ends()]
        events = [e for t in others for e in iter(t)] + [e for e in iter(self)]
        for event in events:
            event.link_track(self)
            self.update_registers(event_add=event)

    def finish_expired_branches(self, timestep, features_leftover, allow_missing=False):
        """Finish all track branches which lag in timestep."""
        debug = False
        if debug:
            log.debug("HEADS {}".format(self.heads()))  # SR_DBG
        for event in self.heads():
            if event.timestep() < timestep:

                if DBG:
                    log.debug(
                        ("finish branch of track #{tid} at feature " "#{fid}").format(
                            tid=self.id(), fid=event.feature().id()
                        )
                    )

                # SR_TODO< Clean up cyclone-specific code!!!!
                try:
                    if event.feature().bcc_fraction() > 0:
                        new_event = event.as_exit()
                    else:
                        new_event = event.as_lysis()
                except AttributeError:
                    new_event = event.as_lysis()
                # SR_TODO>

                self.replace_event(event_del=event, event_add=new_event)

    def finish(self):
        """Finish all active track segements."""
        for event in self.heads():
            if event.is_dummy():
                events_prev = event.prev()
                self.remove_event(event)
                for event_prev in events_prev:
                    event_stop = event_prev.as_stop()
                    self.replace_event(event_del=event_prev, event_add=event_stop)
                continue
            if event.is_genesis():
                self.replace_event(
                    event_del=event, event_add=event.as_isolated(id=event.id())
                )
                continue
            self.replace_event(event_del=event, event_add=event.as_stop(id=event.id()))

    def remove_stubs(self, n=1):
        FeatureTrackStubRemoverEnds(self, n).run()

    def remove_short_splits(self, n=1):
        FeatureTrackStubRemoverSplits(self, n).run()


# SR_TODO Consider pulling out a base class for FeatureTrackBranch and
# SR_TODO FeatureTrackEventBase for the whole prev/next linking functionality
# SR_TODO Maybe there's even a "linked-list" abstract base class available?
class FeatureTrackBranch(Sequence):

    id_manager = IDManager()

    def __init__(self, events, *, prev=None, next=None, active=True, end=None, id=None):
        if not events:
            err = "Cannot initialize instance of {} with events: {}".format(
                self.__class__.__name__, events
            )
            raise ValueError(err)

        self._events = sorted(events.copy(), key=lambda e: e.timestep())
        self._prev = prev if prev else []
        self._next = next if next else []
        self._active = active

        if id is not None:
            self._id = id
            FeatureTrackBranch.id_manager.blacklist(id)
        else:
            self._id = FeatureTrackBranch.id_manager.next()

        self._prev_events = self._events[0].prev()
        self._next_events = self._events[-1].next()

        self._track = None

    def id(self):
        return self._id

    def __repr__(self):
        sid, eid = ["-/-"] * 2
        return (
            "<{cls}[{id0}]: id={id1}, active={a}, n={n}, "
            "events={eids}, features={fids}, "
            "prev={prev}, next={next}"
        ).format(
            cls=self.__class__.__name__,
            id0=id(self),
            id1=self.id(),
            a=("T" if self.is_active() else "F"),
            n=self.n(),
            eids=", ".join([str(e.id()) for e in self]),
            fids=", ".join([str(e.feature().id()) for e in self]),
            prev=", ".join(
                [
                    "({}:{})".format(
                        b.id(),
                        ", ".join(
                            ["{}/{}".format(e.id(), e.feature().id()) for e in b]
                        ),
                    )
                    for b in self.prev()
                ]
            ),
            next=", ".join(
                [
                    "({}:{})".format(
                        b.id(),
                        ", ".join(
                            ["{}/{}".format(e.id(), e.feature().id()) for e in b]
                        ),
                    )
                    for b in self.next()
                ]
            ),
            e=eid,
        )

    def __str__(self):
        return "{}<{}>{}".format(
            "".join(
                ["({}|{})".format(e.id(), e.feature().id()) for e in self.prev_events()]
            ),
            "".join(["({}|{})".format(e.id(), e.feature().id()) for e in self]),
            "".join(
                ["({}|{})".format(e.id(), e.feature().id()) for e in self.next_events()]
            ),
        )

    def __hash__(self):
        return id(self)

    # SR_TODO Might be necessary to improve this definition
    def __eq__(self, other):
        try:
            return set([e.id() for e in self]) == set([e.id() for e in other])
        except Exception:
            return False

    def __lt__(self, other):
        return self.n() < other.n()

    def __len__(self):
        return len(self._events)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.__class__(self._events[i])
        return self._events[i]

    def set_track(self, track):
        self._track = track

    def track(self):
        return self._track

    def duration(self):
        """Total lengths in hours."""
        event0 = min(self.events(), key=lambda e: e.timestep())
        event1 = max(self.events(), key=lambda e: e.timestep())
        delta_timestep = event1.timestep() - event0.timestep()
        try:
            return delta_timestep.total_seconds() / 3600
        except AttributeError:
            return delta_timestep / self.track()._delta_timestep

    def events(self, direction=None):
        events = [event for event in self]
        if not direction:
            return events
        if direction > 0:
            return sorted(events, key=lambda e: e.timestep())
        return sorted(events, key=lambda e: e.timestep(), reverse=True)

    def features(self):
        return [event.feature() for event in self]

    def total_area(self):
        """Sum of areas of all features."""
        return sum([feature.area() for feature in self.features()])

    def is_active(self):
        return self._active

    def set_active(self, active):
        self._active = active

    def n(self):
        return len(self._events)

    def is_short(self, n):
        return self.n() <= n

    def tot_area(self):
        return sum([event.feature().area() for event in self])

    def contains_branching(self, direction=None):
        if not direction:
            return self.contains_merging() or self.contains_splitting()
        if direction > 0:
            return self.contains_splitting()
        return self.contains_merging()

    def contains_merging(self, not_splitting=False):
        if not_splitting:
            return any(e.is_merging() and not e.is_splitting for e in self)
        return any(e.is_merging() for e in self)

    def contains_splitting(self, not_merging=False):
        if not_merging:
            return any(e.is_splitting() and not e.is_merging() for e in self)
        return any(e.is_splitting() for e in self)

    def continuation_prev(self):
        """Get the successor event(s) in negative direction."""
        if len(self) == 0:
            return []
        return self[0].next()

    def continuation_next(self):
        """Get the successor event(s) in positive direction."""
        if len(self) == 0:
            return []
        return self[-1].next()

    def prev_events(self):
        return [e for e in self._prev_events]

    def next_events(self):
        return [e for e in self._next_events]

    def successor_events(self, direction):
        if direction > 0:
            return self.next_events()
        elif direction < 0:
            return self.prev_events()
        err = "Invalid value '{}' for argument 'direction'".format(direction)
        raise ValueError(err)

    # SR_TODO Copied from FeatureTrackEventBase; consider common base class!

    def prev(self):
        return [branch for branch in self._prev]

    def next(self):
        return [branch for branch in self._next]

    def link_forward(self, branch):
        if branch in self.next():
            err = "branch {e1} already linked to branch {e0}: {n}".format(
                e1=branch, e0=self, n=self.next()
            )
            raise Exception(err)
        self._next.append(branch)
        if not self in branch.prev():
            branch._prev.append(self)


class FeatureTrackTruncatorBase:
    def __init__(self, track, n):

        self.track = track
        self.n = n

        self._removal_register = dict()

    def _remove_stub_events(self):

        events = []
        for event in self.track.events():
            # if ( event.id() in self._removal_register and
            #        self._removal_register[event.id()] ):
            if self._removal_register.get(event.id(), False):
                events.append(event)

        for i, event in enumerate(events):

            if self._removal_register.get(event.id(), False):
                # print("REMOVE EVENT: {}".format(self.event_info(event)))
                self.track.remove_event(event)
                continue

            if event.is_merging():
                events[i - 1].unlink_forward(event, check=False)

            if events[i - 1].is_splitting():
                event.unlink()

    def _adapt_truncated_branching_event_types(self):
        for event in self.track.events():
            if event.is_splitting() and len(event.next()) <= 1:
                new_event = event.as_non_splitting()
                self.track.replace_event(event_del=event, event_add=new_event)
                event = new_event
            if event.is_merging() and len(event.prev()) <= 1:
                new_event = event.as_non_merging()
                self.track.replace_event(event_del=event, event_add=new_event)
                event = new_event
            if not event.is_start() and not event.prev():
                # SR_TODO< Cyclone-specific code!!!
                try:
                    if event.feature().bcc_fraction() > 0:
                        new_event = event.as_enter()
                    else:
                        new_event = event.as_genesis()
                except AttributeError:
                    new_event = event.as_genesis()
                # SR_TODO>
                self.track.replace_event(event_del=event, event_add=new_event)
                event = new_event
            if not event.is_end() and not event.next():
                # SR_TODO<
                try:
                    if event.feature().bcc_fraction() > 0:
                        new_event = event.as_exit()
                    else:
                        new_event = event.as_lysis()
                except AttributeError:
                    new_event = event.as_lysis()
                # SR_TODO>
                self.track.replace_event(event_del=event, event_add=new_event)
                event = new_event

    def _eliminate_relative_splitting(self, event, direction):
        if direction > 0:
            new_event = event.as_non_splitting()
        else:
            new_event = event.as_non_merging()
        self.track.replace_event(event_del=event, event_add=new_event)
        return new_event

    # SR_DBG
    def event_info(self, e):
        return "{:<4}({:4}:{:4})".format(
            "".join(
                [
                    c
                    for c in e.__class__.__name__.replace("FeatureTrackEvent", "")
                    if c.isupper()
                ]
            ),
            e.id(),
            e.feature().id(),
        )

    # SR_DBG
    def event_info_full(self, e, subinfo=None):
        if subinfo is None:
            subinfo = self.event_info
        return "{{ {p:>50} <= {e:^20} => {n:<50} }}".format(
            e=self.event_info(e),
            p=", ".join([subinfo(p) for p in e.prev()]),
            n=", ".join([subinfo(n) for n in e.next()]),
        )

    # SR_DBG
    def branch_info(self, branch):
        return "-".join(
            "{}".format(self.event_info(e).replace("  ", "") for e in branch)
        )


class FeatureTrackStubRemoverEnds(FeatureTrackTruncatorBase):
    """Remove insignificantly short branches from mergings/splittings.

    Tracks can become unnecessarily complicated if small fragments often
    split off only to disappear after one or a few timesteps, or merge
    with a major fragment shortly after appearing. Cutting off these
    stubby branches can considerably simplify a track.

    Branches are considered stubs if they only exist for <n> timesteps
    before/after a merging/splitting (regardless of their size). If a
    branch starts/ends only with stubs, the most significant of them is
    retained. Significance is determnied by length (the longer the better,
    if <n> >1) and size (the bigger the better). Total track duration
    is thus unaffected.
    """

    def __init__(self, track, n):
        super().__init__(track, n)

        self._continued = None

    def run(self, n=None):

        for n_iter in range(10):
            if DBG:
                log.debug("\nITERATION {}".format(n_iter))

            # Initialize removal status on all events
            # (1: retain, 0: unset, +1: remove)
            for event in self.track.events():
                if not hasattr(event, "remove"):
                    event.remove = 0
                if event.remove == -0.5:
                    event.remove = 0

            # First, mark event of long linear segments for retaining
            if DBG:
                log.debug("--- PRE-PROC: SEGMENTS ---")
            for segment in self.track.linear_segments():
                if len(segment) > self.n or not (
                    e.is_start() or e.is_end() for e in segment
                ):
                    for event in segment:
                        if DBG:
                            log.debug("+++ RETAIN EVENT(0): {}".format(event))
                        event.remove = -2
            if DBG:
                log.debug("")  # SR_DBG

            # Mark all events for retaining which have are far enough
            # from the track edges
            if DBG:
                log.debug("--- PRE-PROC: EDGES ---")
            events_near_edges = set()
            for edge_event in self.track.starts() + self.track.ends():
                edge_branches = edge_event.events_all_branches(
                    distance=self.n + 1,
                    stop_condition=lambda e, _: e.is_branching(),
                    stop_event_inclusive=False,
                    stop_condition_force=False,
                    soft_stop=True,
                )
                events_near_edges.update([e for b in edge_branches for e in b])
            # SR_DBG<
            if DBG:
                log.debug(
                    "EVENTS NEAR EDGES: {}".format(
                        ", ".join([str(e) for e in events_near_edges])
                    )
                )
            # SR_DBG>
            for event in self.track.events():
                if event not in events_near_edges:
                    # SR_DBG<
                    if DBG:
                        log.debug("RETAIN EVENT(1): {}".format(event))
                    # SR_DBG>
                    event.remove = -1
            if DBG:
                log.debug("")  # SR_DBG

            # Alternate direction
            if n_iter % 2 == 0:
                direction = 1
            else:
                direction = -1

            # Mark events belonging to stubs for removal
            self._identify_stubs(direction=direction)

            # SR_TMP<
            self._removal_register = {}
            for event in self.track.events():
                if event.remove > 0:
                    self._removal_register[event.id()] = True
            # SR_TMP>

            if n_iter and n_iter % 2 == 1:
                if not any(v for k, v in self._removal_register.items()):
                    break

            self._remove_stub_events()
            self._adapt_truncated_branching_event_types()
        else:
            err = (
                "Max. number of iterations reached: {}\nCould not "
                "eliminate all active short branches"
            ).format(n_iter)
            raise InfiniteLoopError(err)

    def _identify_stubs(self, direction=1):

        # Get earliest/latest events of track (depending on direction)
        temporal_edge_events = self.track.events_ts()[0 if direction > 0 else -1]
        if DBG:
            log.debug(
                "\n{} TEMPORAL EDGE EVENTS: {}".format(
                    ">>>" if direction > 0 else "<<<",
                    ", ".join([str(e) for e in temporal_edge_events]),
                )
            )

        short_branches_all = []

        edges = self.track.starts() if direction > 0 else self.track.ends()
        if DBG:
            log.debug(
                "EDGES: {}".format(
                    ", ".join(["{}/{}".format(e.id(), e.feature().id()) for e in edges])
                )
            )
        for event in edges:
            if event.remove < 0:
                continue

            def stop_condition(e, _):
                return not e.is_branch_end(-direction) and e.is_branching()

            # Get all branches from this edge event to the next branching
            all_branches = event.events_all_branches(
                direction=direction,
                stop_condition=stop_condition,
                stop_event_inclusive=False,
                soft_stop=True,
            )

            # Select those branches that don't contain any events to retain
            short_branches = [
                b
                for b in all_branches
                if len(b) <= self.n and not any(e.remove < 0 for e in b)
            ]

            # SR_DBG<
            if DBG:
                log.debug(
                    "{} EVENT: {}".format("<<<" if direction < 0 else ">>>", event)
                )
            if DBG:
                log.debug(
                    " -> LONG : {}".format(
                        ", ".join(
                            [
                                str(b)
                                for b in sorted(all_branches, key=lambda b: len(b))
                                if b not in short_branches
                            ]
                        )
                    )
                )
            if DBG:
                log.debug(
                    " -> SHORT: {}".format(
                        ", ".join(
                            [
                                str(b)
                                for b in sorted(short_branches, key=lambda b: len(b))
                            ]
                        )
                    )
                )
            # SR_DBG>

            for branch in short_branches.copy():

                # Discard branches containing two ends
                if any(
                    e.is_branch_end(direction) and not e.is_branch_end(-direction)
                    for e in branch
                ):
                    if DBG:
                        log.debug("BRANCH CONTAINS TWO ENDS: {}".format(branch))
                    short_branches.remove(branch)
                    continue

                # Discard branches not followed by a merging
                # SR_TODO Try to make this obsolete by refining the
                # SR_TODO branch selection stop criterium!
                if not any(
                    e.is_branching(-direction)
                    for e in branch.successor_events(direction)
                ):
                    short_branches.remove(branch)
                    continue

            # Store short branches ("stubs")
            short_branches_all.extend([b for b in short_branches])
        if DBG:
            log.debug(
                "short branches: {}".format(
                    ", ".join([str(b) for b in short_branches_all])
                )
            )

        # SR_TODO fix description (bit a mess)
        # If a stub contains a direction-relative splitting/merging, it must
        # also contain a respective merging/splitting, and all branches
        # meeting/parting in that merging must originate from a
        # splitting/merging in the stub
        for branch in short_branches_all.copy():
            retain_branch = False

            successors = branch.events(direction)[-1].successors(direction)

            for splitting in (e for e in branch if e.is_branching(direction)):
                for merging in (
                    e
                    for e in (branch.events() + successors)
                    if e.is_branching(-direction)
                ):
                    if all(
                        merging in b
                        for b in splitting.events_all_branches(direction=direction)
                    ):
                        break
                else:
                    if DBG:
                        log.debug(
                            "SPLITTING W/O MERGING: {} {}".format(branch, splitting)
                        )
                    # Contains splitting, but no merging:
                    # Retain all events not belonging to other stubs
                    short_branches_all.remove(branch)
                    for event in branch:
                        if not any(event in b for b in short_branches_all):
                            if DBG:
                                log.debug("RETAIN EVENT(2): {}".format(event))
                            event.remove = -1
                    retain_branch = True
                if retain_branch:
                    break

        # SR_TODO Reconsider this (introduced only for one test)
        # SR_TODO Chec whether this can be removed or replaced...
        # Branches containing merging-splitting events cannot be removed
        for branch in short_branches_all.copy():
            if any(e.is_merging() and e.is_splitting() for e in branch):
                short_branches_all.remove(branch)

        # SR_TODO Should make the "first timestep" check obsolete!
        # If all branches in one direction are stubs,
        # retain the most significant one
        if DBG:
            log.debug("")  # SR_DBG
        successor_events = list(
            set([e for b in short_branches_all for e in b.successor_events(direction)])
        )
        successor_events.sort(key=lambda e: e.timestep(), reverse=(direction > 0))
        # SR_DBG<
        if DBG:
            log.debug(
                "SUCCESSOR EVENTS: {}".format(
                    ", ".join([str(e) for e in successor_events])
                )
            )
        # SR_DBG>
        for successor_event in successor_events.copy():
            if successor_event not in successor_events:
                continue
            branches_back = successor_event.events_all_branches(
                direction=-direction,
                start_event_inclusive=False,
                stop_event_inclusive=True,
            )
            if not branches_back:
                # SR_DBG<
                if DBG:
                    log.debug(
                        "NO BACK-BRANCHES FOUND FOR {}".format(str(successor_event))
                    )
                # SR_DBG>
                continue
            if any(e.remove == -2 for b in branches_back for e in b):
                # SR_DBG<
                if DBG:
                    log.debug(
                        "SKIP BRANCHES: {}".format(
                            ", ".join([str(b) for b in branches_back])
                        )
                    )
                # SR_DBG>
                continue
            # SR_DBG<
            if DBG:
                log.debug(
                    "BACK-BRANCHES {}: {}".format(
                        str(successor_event), ", ".join([str(b) for b in branches_back])
                    )
                )
            # SR_DBG>
            if not all(len(b) <= self.n for b in branches_back):
                if DBG:
                    log.debug("NOT ALL BRANCHES SHORT")
                continue
            for event in successor_events:
                if any(event in b for b in branches_back):
                    # SR_DBG<
                    if DBG:
                        log.debug("SKIP SUCCESSOR EVENT {}".format(event))
                    # SR_DBG>
                    successor_events.remove(event)
            most_sig_stub = self._find_most_significant_stub(branches_back)
            # SR_DBG<
            if DBG:
                log.debug("    REMOVE FROM SHORT BRANCHES: {}".format(most_sig_stub))
            # SR_DBG>
            if most_sig_stub in short_branches_all:
                short_branches_all.remove(most_sig_stub)
            for event in most_sig_stub:
                if DBG:
                    log.debug("RETAIN EVENT(3): {}".format(event))
                event.remove = -1
            # for event in (e for b in branches_back for e in b
            #        if b is not most_sig_stub):
            #    if DBG: log.debug("--- REMOVE EVENT: {}".format(event))
            #    event.remove = 1
        if DBG:
            log.debug("")  # SR_DBG

        # Retain stubs starting/ending at the first/last timestep
        edge_timestep = (
            self.track.ts_start() if direction > 0 else self.track.ts_start()
        )

        # Mark events of short branches for removal
        for branch in short_branches_all:
            if DBG:
                log.debug("REMOVAL CANDIDATE: {}".format(str(branch)))
            for event in branch:
                if event.remove == 0:
                    if DBG:
                        log.debug("--- REMOVE EVENT: {}".format(event))
                    event.remove = 1

        # SR_TODO Kinda hacky that this specific issue has to be fixed
        # SR_TODO here and not already earlier, but so be it...
        self._retain_bridge_events(short_branches_all, direction)

    def _retain_bridge_events(self, short_branches_all, direction):
        """Make sure there are no gaps in the track."""

        # Find the events nearest to the starts which are retained
        start_events = set()
        for event in self.track.starts():
            if event.remove <= 0:
                start_events.add(event)
                continue
            for branch in event.events_all_branches(
                direction=1,
                stop_condition=lambda e, _: e.remove > 0,
                stop_event_inclusive=False,
            ):
                start_events.add(branch[-1])

        # Find the events nearest to the ends which are retained
        end_events = set()
        for event in self.track.ends():
            if event.remove <= 0:
                end_events.add(event)
                continue
            for branch in event.events_all_branches(
                direction=-1,
                stop_condition=lambda e, _: e.remove > 0,
                stop_event_inclusive=False,
            ):
                start_events.add(branch[-1])

        # SR_DBG<
        if DBG:
            log.debug(
                "START EVENTS: {}".format(", ".join([str(e) for e in start_events]))
            )
        if DBG:
            log.debug(
                "END EVENTS  : {}".format(", ".join([str(e) for e in end_events]))
            )
        # SR_DBG>

        # Find all branches between these start and end events
        branches = []
        for event in start_events:
            branches.extend(
                event.events_all_branches(
                    direction=1,
                    stop_condition=lambda e, _: e in end_events,
                    stop_event_inclusive=True,
                    stop_condition_force=True,
                )
            )
        branches = [b for b in branches if len(b) > 2]
        # SR_DBG<
        if DBG:
            log.debug(
                "CONNECTING BRANCHES: {}".format(", ".join([str(b) for b in branches]))
            )
        # SR_DBG>
        for branch in branches:
            parallel_branches = [
                b
                for b in branches
                if b[0] == branch[0] and b[-1] == branch[-1] and b != branch
            ]
            if parallel_branches:
                # SR_DBG<
                if DBG:
                    log.debug(
                        "PARALLEL BRANCHES TO {}: {}".format(
                            str(branch), ", ".join([str(b) for b in parallel_branches])
                        )
                    )
                # SR_DBG>
            else:
                if DBG:
                    log.debug("NO PARALLEL BRANCHES FOUND TO {}".format(str(branch)))
            # SR_DBG<
            for event in branch[1:-1]:
                # SR_DBG>
                if event.remove <= 0:
                    continue
                if parallel_branches and not all(
                    event in b[1:-1] for b in parallel_branches
                ):
                    # SR_DBG<
                    if DBG:
                        log.debug(
                            (
                                "DON'T RETAIN EVENT (NOT IN ALL "
                                "PARALLEL BRANCHES): {}"
                            ).format(event)
                        )
                    # SR_DBG>
                    continue
                # SR_DBG<
                if DBG:
                    log.debug("RETAIN EVENT(4): {}".format(event))
                # SR_DBG>
                event.remove = -0.5

    def _branch_is_short(self, branch):
        return len(branch) <= self.n

    def _mark_stubs_for_removal(self, event, successors, direction):

        # Find all possible branches (i.e. pathways to the end)
        branches = [
            b
            for e in event.successors(direction)
            for b in e.events_all_branches(direction=direction)
        ]

        # SR_DBG<
        if DBG:
            log.debug(
                "BRANCHES FROM {}:".format(self.event_info(event).replace("  ", ""))
            )

    def _find_most_significant_stub(self, stubs):
        """Find the most significant "stub branch" in a list."""

        # First sort by length (only meaningful for n>1)
        stubs.sort(key=lambda events: len(events), reverse=True)

        # Check the length of the stubs
        stubs_maxlen = [stub for stub in stubs if len(stub) == len(stubs[0])]

        # Return longest stub if there's only one of that length
        if len(stubs_maxlen) == 1:
            if DBG:
                log.debug("RETURN 0")
            return stubs[0]

        # If there are multiple stubs with maximal length,
        # select the one with the biggest cumulative feature area
        branch_areas = [stub.total_area() for stub in stubs_maxlen]
        max_branch_area = max(branch_areas)
        inds = [i for i, a in enumerate(branch_areas) if a == max_branch_area]
        if DBG:
            log.debug("RETURN 1")
        return stubs_maxlen[inds[0]]


class FeatureTrackStubRemoverSplits(FeatureTrackTruncatorBase):
    def __init__(self, track, n):
        super().__init__(track, n)

    def run(self):

        # Identify and remove short branches
        self._identify_short_splits(direction=1)
        # SR_TODO What about the direction?
        self._remove_stub_events()

        # Adapt truncated branching events
        self._adapt_truncated_branching_event_types()

    def _identify_short_splits(self, direction):

        # SR_TODO If direction is indeed meaningful (currently hard-coded
        # SR_TODO to forward), distinguish between starts and ends!
        events = self.track.starts() if direction > 0 else self.track.ends()
        for event in events:
            self._identify_short_splits_rec(event, direction)

    def _identify_short_splits_rec(self, event, direction):

        if event.is_branch_end(direction):
            return

        if event.is_branching(direction):
            # SR_DBG<
            if DBG:
                log.debug("BRANCHING {}".format(self.event_info(event)))
            # SR_DBG>

            self._determine_branches_to_remove(event, direction)

            # SR_DBG
            # print("TO BE REMOVED: {}".format(
            #    [eid for eid in self._removal_register]))

        # Continue along track
        for other_event in event.successors(direction):
            self._identify_short_splits_rec(other_event, direction)

    def _determine_branches_to_remove(self, event, direction):

        for n_iter in range(9):
            if DBG:
                log.debug("\n[{}] {}".format(n_iter, self.event_info(event)))
            branches = self._find_branches(event, direction)

            # SR_DBG<
            if DBG:
                log.debug(
                    "BRANCHES BEFORE:{}".format(
                        "\n  {}".format("\n  ".join([str(e) for e in branches]))
                        if branches
                        else " NONE"
                    )
                )
            # SR_DBG>

            branches_grouped = self._group_branches(branches)
            # SR_DBG<
            if DBG:
                log.debug(
                    "GROUPED BRANCHES: {}".format(
                        "\n  {}".format(
                            "\n  ".join(
                                [
                                    "{} to #{}: {}".format(i, m.id(), str(e))
                                    for (i, m), e in branches_grouped.items()
                                ]
                            )
                        )
                        if branches_grouped
                        else " NONE"
                    )
                )
            # SR_DBG>

            for (length, merging), branches_group in sorted(branches_grouped.items()):
                if DBG:
                    log.debug("LEN: {} / N: {}".format(length, self.n))
                # SR_TODO Find out why this works with length-1 > n,
                # SR_TODO but not with length > n ...
                # if length > self.n:
                if length - 1 > self.n:
                    # SR_DBG<
                    if DBG:
                        log.debug("BREAK: {}>{}".format(length, self.n))
                    # SR_DBG>
                    break
                self._mark_insignificant_branches_for_removal(length, branches_group)

            # SR_DBG<
            if DBG:
                log.debug(
                    "BRANCHES AFTER:{}".format(
                        "\n  {}".format("\n  ".join([str(e) for e in branches]))
                        if branches
                        else " NONE"
                    )
                )
            if DBG:
                log.debug(
                    "ACTIVE SHORT BRANCHES: {}".format(
                        self._count_active_short_branches(branches_grouped)
                    )
                )
            if DBG:
                log.debug(
                    "EVENTS TO REMOVE: {}".format(
                        ", ".join(
                            [str(eid) for eid, f in self._removal_register.items() if f]
                        )
                    )
                )
            # SR_DBG>

            if self._count_active_short_branches(branches_grouped) <= 1:
                break
        else:
            err = (
                "Max. number of iterations reached: {}\nCould not "
                "eliminate all active short branches"
            ).format(n_iter)
            raise InfiniteLoopError(err)

    # SR_TODO Move some code into a designated method
    def _find_branches(self, event, direction):
        def condition(event_to_check, passed_distance):

            # Event is direction-relative merging
            is_branching = event_to_check.is_branching(-direction)

            # Event is not registered for removal
            is_retained = not self._removal_register.get(event_to_check.id(), False)

            # Event is directly related to reference event
            is_related = (
                len(
                    [
                        e
                        for e in event_to_check.successors(-direction)
                        if e.is_directly_related(event, -direction)
                    ]
                )
                > 1
            )

            # Event is active direction-relative merging, i.e. more
            # than one branches back to reference event are not removed
            n_active_branches_back_to_ref = 0
            for event_back in event_to_check.successors(-direction):
                for branch_back in event_back.events_all_branches(
                    direction=-direction,
                    stop_condition=lambda e, d: e == event,
                    stop_condition_force=True,
                ):
                    if not any(
                        self._removal_register.get(event.id(), False)
                        for event in branch_back
                    ):
                        n_active_branches_back_to_ref += 1
            is_active_branching = n_active_branches_back_to_ref > 1
            # SR_DBG<
            # if DBG: log.debug("COND: {}+{}+{}+{}={} ({})".format(
            #        str(is_branching)[0], str(is_retained)[0],
            #        str(is_related)[0], str(is_active_branching)[0],
            #        str(is_branching and is_retained and is_related and
            #            is_active_branching)[0],
            #        event_to_check))
            # SR_DBG>
            return is_branching and is_retained and is_related and is_active_branching

        branches = []
        for event_next in event.successors(direction):

            # Find all branches leading to merging events where at
            # least two branches starting from the current event meet
            branches.extend(
                event_next.events_all_branches(
                    direction=direction,
                    # distance    = self.n,
                    # distance    = self.n+2,
                    # soft_stop   = True,
                    soft_stop=False,
                    stop_condition=condition,
                    stop_condition_force=True,
                )
            )

        # Eliminate inactive branches containing events marked for removal
        branches = [
            branch
            for branch in branches
            if not any(self._removal_register.get(e.id(), False) for e in branch)
        ]

        # SR_TMP<
        # If all branches share the first event, discard them
        # SR_TODO Find a way to not even get so far...
        # SR_TODO (This came from a GenericFeature issue)
        if len(set(branch[0] for branch in branches)) == 1:
            return []
        # SR_TMP>

        return branches

    def _count_active_short_branches(self, branches_grouped):
        n_active = 0
        for branches in branches_grouped.values():
            for branch in branches:
                if branch.is_short(self.n) and branch.is_active():
                    n_active += 1
        return n_active

    def _mark_insignificant_branches_for_removal(self, length, branches):
        """Mark all but the most significant branch for removal."""

        # Preliminarily mark all short track events for removal
        # SR_DBG<
        if DBG:
            log.debug("Mark short branches preliminarily for removal:")
        # SR_DBG>
        for branch in branches:
            # SR_DBG<
            if DBG:
                log.debug("  {}".format(branch))
            # SR_DBG>
            for event in branch:
                self._removal_register[event.id()] = True

        # Identify the most significant one and retain it
        most_sig_branch = self._find_most_significant_branch(branches)
        # SR_DBG<
        if DBG:
            log.debug("Unmark most significant short branch from removal:")
        if DBG:
            log.debug("  {}".format(most_sig_branch))
        # SR_DBG>
        for eid in self._removal_register:
            if eid in (e.id() for e in most_sig_branch):
                self._removal_register[eid] = False

        # If a branch contains an unrelated merging/splitting
        # retain all events after/before that merging/splitting event
        self._handle_unrelated_mergings(branches)
        self._handle_unrelated_splittings(branches)

        # Update active flag
        for branch in branches:
            branch.set_active(self._branch_is_active(branch))

    def _handle_unrelated_mergings(self, branches):

        for branch in branches:

            contains_merging = False
            if branch.contains_merging():
                contains_merging = True
                after_merging = False

            for event in branch:

                if contains_merging:
                    if event.is_merging():
                        after_merging = True
                    if after_merging:
                        self._removal_register[event.id()] = False

    def _handle_unrelated_splittings(self, branches):

        for branch in branches:

            contains_splitting = False
            if branch.contains_splitting(not_merging=True):
                contains_splitting = True
                before_splitting = True

            for event in branch:

                if contains_splitting:
                    if before_splitting:
                        self._removal_register[event.id()] = False
                    if event.is_splitting():
                        before_splitting = False

    def _branch_is_active(self, branch):
        return all(
            not self._removal_register.get(event.id(), False) for event in branch
        )

    def _find_most_significant_branch(self, branches):
        """Find the "most significant" among multiple short branches.

        In case there are exclusively short branches at a splitting, one of
        them needs to be selected to be retained. The first criterium is
        length, the second the total area of the features.
        """
        # If only one branch has maximum length, it is the chosen one.
        branches.sort(reverse=True)
        branches_maxlen = [b for b in branches if b.n() == branches[0].n()]
        if len(branches_maxlen) == 1:
            return branches_maxlen[0]

        # Otherwise, select the one with the largest total feature area
        tot_area = lambda b: sum([e.feature().area() for e in b[1]])
        branch_total_area = sorted(
            [(b.tot_area(), b) for i, b in enumerate(branches_maxlen)], reverse=True
        )
        return branch_total_area[0][1]

    def _group_branches(self, branches):
        """Group the branches by length and merging event.

        The input is a list of "(length, events, merging)" tuples, where
        "length" corresponds to the number of events in the branch,
        "events" is a list of these events, and "merging" es the merging
        event in which the branch ends.

        Double-check that all branches associated with a given merging event
        are of the same length.
        """
        debug = False
        branches_grouped = {}
        for branch in branches:

            key = (branch.n(), branch[-1])
            if key not in branches_grouped:
                branches_grouped[key] = []

            # Check whether branch is active, i.e. whether the events have not
            # already been marked for removal (in which case it is not active)
            branches_grouped[key].append(branch)

        # Consistency check
        mergings = [m for n, m in branches_grouped.keys()]
        if len(set(mergings)) < len(branches_grouped):
            err = (
                "Branches of differents lengths end in the same merging " "event:\n {}"
            ).format(branches_grouped)
            raise Exception(err)

        # DBG_BLOCK<
        if debug:
            log.debug("BRANCHES:")
            for (length, merging), branches in branches_grouped.items():
                log.debug("  {:2} {}".format(length, merging))
                for branch in branches:
                    log.debug(
                        "      {} - {}".format(
                            "ACTIVE" if branch.is_active() else "INACTIVE",
                            ", ".join(
                                [
                                    self.event_info(event).replace("  ", "")
                                    for event in branch
                                ]
                            ),
                        )
                    )
        # DBG_BLOCK>

        return branches_grouped


class FeatureTrackFactory:
    def __init__(self, cls_default=FeatureTrackBase):
        self.cls_default = cls_default

    def run(self):
        raise NotImplementedError("FeatureTrackFactory.run")
        return self.cls_default()

    def rebuild(
        self, *, id, events, starts=None, ends=None, heads=None, delta_timestep=None
    ):
        """Rebuild a track from data read from disk.

        Necessary for the rebuild are the track id and a list of all events
        part of the track. Optionally, lists of events expected to be starts,
        ends, or active heads of the track can be passed as well, which are
        not necessary for the rebuild, but are used to double-check the track
        if passed.
        """

        # Initialize "empty" track
        kwargs_ini = {"id": id}
        if delta_timestep:
            # kwargs_ini["delta_timestep"] = delta_timestep
            kwargs_ini["delta_timestep"] = delta_timestep
        try:
            track = self.cls_default(**kwargs_ini)
        except Exception as e:
            err = "Error creating {c} object.\nARGS: {a}\nERROR: {e}".format(
                c=self.cls_default.__name__, a=kwargs_ini, e=e
            )
            raise Exception(err)

        # Assign the track to the events
        for event in events:
            event.link_track(track)

        # Assign the events to the track
        for event in events:
            track.update_registers(event_add=event)

        # Sanity checks: Neither starts, heads, or events are necessary
        # to rebuild the track, only the full event lists (which events
        # belong to which category is determined automatically). If the
        # expected lists are passed, however, they are double-checked.
        def check_events(track, etype, events):
            if not events:
                return
            track_events = getattr(track, etype)()
            if not sorted(events) == sorted(track_events):
                raise Exception(err)
                err = (
                    "Error rebuilding track: {et} don't match!\n"
                    "TRACK: {t}\n{etu}: {te}\nEXPECTED: {e}"
                ).format(
                    et=etype, etu=etype.upper(), t=track, te=track_events, e=events
                )
                raise err

        check_events(track, "starts", starts)
        check_events(track, "ends", ends)
        check_events(track, "heads", heads)

        return track


FeatureTrackBase.cls_factory = FeatureTrackFactory


class FeatureTrackerBase:

    # SR_TODO Extract timestep code into separate class
    def __init__(
        self,
        min_overlap=None,
        max_area=None,
        threshold=0.5,
        require_overlap=False,
        f_overlap=0,
        f_area=0,
        max_children=-1,
        prob_area_min=0.0,
        max_dist_abs=-1,
        max_dist_rel=2.0,
        allow_missing=False,
        timestep=0,
        delta_timestep=1,
        timestep_datetime=False,
        ids_datetime=False,
        ids_digits=None,
        ids_base=None,
    ):

        self.require_overlap = require_overlap
        self.f_overlap = f_overlap
        self.f_area = f_area
        self.max_children = max_children
        self.prob_area_min = prob_area_min
        self.min_overlap = min_overlap
        self.max_area = max_area
        self.max_dist_abs = max_dist_abs
        self.max_dist_rel = max_dist_rel
        self.threshold = threshold
        self.allow_missing = allow_missing

        self.delta_timestep = delta_timestep
        self.set_timestep(timestep)
        self.timestep_datetime = timestep_datetime

        self.ids_datetime = ids_datetime
        if ids_datetime:
            self.reset_ids(base=ids_base, digits=ids_digits)

        if min_overlap is None:
            self.min_overlap = 0.1
        if max_area is None:
            self.max_area = 1.9

        self._check_input_validity()

        self._active_tracks = []
        self._finished_tracks = []

        self.extender = FeatureTrackExtender(self)

    def timestep(self):
        return self._timestep

    def set_timestep(self, ts):
        self._timestep = ts

    def _check_input_validity(self):

        # Check thresholds
        if not 0 < self.min_overlap < 1:
            err = (
                "Invalid minimal overlap:\n  min_overlap : {}\n"
                "Must be between 0 and 1!"
            ).format(self.min_overlap)
            raise ValueError(err)
        if self.max_area < 1:
            err = (
                "Invalid maximal area fraction:\n  max_area : {}\n" "Must be > 1!"
            ).format(self.max_area)
            raise ValueError(err)

        # Make sure the factors are significant
        if np.isclose(self.f_overlap, 0) or self.f_overlap < 0:
            self.f_overlap = 0
        if np.isclose(self.f_area, 0) or self.f_area < 0:
            self.f_area = 0
        f_tot = self.f_overlap + self.f_area

        # Scale the factors
        if f_tot > 0:
            self.f_overlap /= f_tot
            self.f_area /= f_tot

        if not self.timestep_datetime:
            if self.delta_timestep <= 0:
                err = "Timestep delta must be above-zero!"
                raise ValueError(err)

    def reset_ids(self, **kwargs):
        """Reset event and features IDs."""
        TrackableFeatureBase.id_manager.reset(**kwargs)
        FeatureTrackEventBase.id_manager.reset(**kwargs)
        FeatureTrackBase.id_manager.reset(**kwargs)
        FeatureTrackBranch.id_manager.reset(**kwargs)

    def get_config(self):

        # SR_TMP< This code should eventually be part of a Timestep class
        try:
            timestep = int(self._timestep.strftime("%Y%m%d%H"))
        except AttributeError:
            timestep = self._timestep
        try:
            delta_timestep = self.delta_timestep.total_seconds() / 3600
        except AttributeError:
            delta_timestep = self.delta_timestep
        # SR_TMP>

        return {
            "require_overlap": self.require_overlap,
            "f_overlap": self.f_overlap,
            "f_area": self.f_area,
            "max_children": self.max_children,
            "prob_area_min": self.prob_area_min,
            "min_overlap": self.min_overlap,
            "max_area": self.max_area,
            "max_dist_abs": self.max_dist_abs,
            "max_dist_rel": self.max_dist_rel,
            "threshold": self.threshold,
            "timestep_datetime": self.timestep_datetime,
            "delta_timestep": delta_timestep,
            "ids_datetime": self.ids_datetime,
        }

    def extend_tracks(self, features, id0=None, timestep=None):
        """Extend all active tracks from a list of features.

        If a features cannot be matched with a track, a new track is started.
        """
        # Check validity of features
        if any(f is None for f in features):
            err = "{c}.extend_tracks: features must not be None!\n{f}".format(
                c=self.__class__.__name__, f=pprint.pformat(features)
            )
            raise ValueError(err)

        if timestep is not None:
            self.set_timestep(timestep)

        self.extender.extend_tracks(features, id0=id0, allow_missing=self.allow_missing)

        if timestep is None:
            self.increment_timestep()

    def increment_timestep(self):
        self._timestep += self.delta_timestep

    def track_heads(self):
        """Get the heads of all active tracks."""
        return [head for track in self.active_tracks() for head in track.heads()]

    def n_active(self):
        return len(self.active_tracks())

    def n_finished(self):
        return len(self.finished_tracks())

    def active_tracks(self):
        return [t for t in self._active_tracks]

    def finished_tracks(self):
        return self._finished_tracks

    def pop_finished_tracks(self):
        finished = self._finished_tracks
        self._finished_tracks = []
        return finished

    def start_track(self, feature=None, track_id=None, feature_id=-1):
        # raise Exception #SR_DBG
        new_track = self.new_track(
            feature=feature, timestep=self.timestep(), id=track_id
        )
        fid = feature_id if feature is None else feature.id_str()
        if DBG:
            log.debug(
                "start new track #{tid} with feature #{fid}".format(
                    tid=new_track.id(), fid=fid
                )
            )
        self._active_tracks.append(new_track)
        return new_track

    def finish_tracks(self, active_tracks=None):
        if active_tracks is None:
            active_tracks = self.active_tracks()
        for track in active_tracks:
            self.finish_track(track, reason="forced")

    def finish_expired_tracks(self, features_leftover):
        """Finish all tracks without continuation at the current timestep."""
        for track in self.active_tracks():
            track.finish_expired_branches(
                self.timestep(), features_leftover, self.allow_missing
            )
            if track.is_finished():
                self.finish_track(track, reason="expired")

    def finish_track(self, track, reason=None):
        if DBG:
            log.debug(
                "finish track #{id}{r}".format(
                    id=track.id(), r=" ({})".format(reason) if reason else ""
                )
            )
        self._active_tracks.remove(track)
        track.finish()
        self._finished_tracks.append(track)

    def merge_tracks(self, tracks, feature_id=-1):
        """Merge multiple tracks into one. Return a new track with new ID."""
        for track in tracks:
            if not track in self._active_tracks:
                err = (
                    "{t} not an active track:\n{a}\n"
                    "track {n}found among finished tracks:\n{f}"
                ).format(
                    t=track,
                    a="\n".join([str(t) for t in sorted(self._active_tracks)]),
                    f="\n".join([str(t) for t in sorted(self._finished_tracks)]),
                    n="" if track in self._finished_tracks else "not ",
                )
                raise Exception(err)
            self._active_tracks.remove(track)
        new_track = self.start_track(feature_id=feature_id)
        new_track.merge(tracks)

        if DBG:
            log.debug(
                "merged tracks {oid} into new track #{nid}".format(
                    oid=", ".join(["#{}".format(track.id()) for track in tracks]),
                    nid=new_track.id(),
                )
            )

        return new_track

    # Abstract methods (need to be defined by subclass)

    def new_track(self, *args, **kwargs):
        self._error_not_overridden("new_track")

    def _error_not_overridden(self, name):
        err = "Method '{}' must be overridden by subclass.".format(name)
        raise NotImplementedError(err)


class FeatureTrackExtender:
    """Start/finish/extend tracks given the features from a new timestep."""

    def __init__(self, tracker, event_factory=None, feature_factory=None):
        self.tracker = tracker
        self.conf = tracker.get_config()

        self.event_factory = (
            event_factory if event_factory else FeatureTrackEventFactory()
        )
        self.feature_factory = feature_factory if feature_factory else FeatureFactory()

    def extend_tracks(self, features_new, id0=None, timestep=None, allow_missing=False):
        """Extend all active tracks from a list of features.

        If a features cannot be matched with a track, a new track is started.

        Method:
         - Find all successors for every track among the features.
         - If a feature is a successor for multiple tracks, merge them.
         - Likewise, split a track if multiple features are likely successors.
         - At most three tracks/features can be involved in merging/splitting.
           If there are more, only retain the most likely three.
         - Tracks without successor are finished.

        If allow_missing is set, branches are not immediately terminated
        if no successor could be found. Instead, the feature is matched again
        against all predecessor-less features of the next time step. This way,
        tracks are not immediately terminated if a feature could not be
        identified for one timestep.
        """
        if id0 is not None:
            self._next_track_id = id0
        if timestep is not None:
            self.tracker.set_timestep(timestep)
        else:
            timestep = self.tracker.timestep()

        # LOG: timestep
        if DBG:
            log.debug("+" * 60)
        if DBG:
            log.debug("TIMESTEP {ts}".format(ts=timestep))
        if DBG:
            log.debug("+" * 60)

        # LOG: tracks
        self._log_table_tracks("\nfinished tracks", self.tracker.finished_tracks())
        self._log_table_tracks("\nactive tracks", self.tracker.active_tracks())

        # LOG: features
        self._log_table_features(features_new)

        # Get track heads
        features_now = [
            head.feature() for head in self.tracker.track_heads() if not head.is_dummy()
        ]

        # LOG: track heads
        self._log_table_heads(features_now)

        if DBG:
            log.debug("\n" + "-" * 60)

        # Identify and assign successors
        self.find_successors(features_now, features_new)

        # Deal with missing features ("dummy events")
        features_new_leftover = [f for f in features_new if f.is_unassigned()]
        if allow_missing:
            self.add_dummy_events(timestep)
            self.handle_dummy_events(timestep, features_new_leftover)

        # Finish all tracks for which no successor has been found
        self.tracker.finish_expired_tracks(features_new_leftover)

        # Start a new track for all unassigned features
        for feature in features_new:
            if not feature.event():
                self.tracker.start_track(feature)

        if DBG:
            log.debug("")

    def find_successors(self, features_now, features_new):
        debug = True

        if debug:
            log.debug(
                "find successors for {} old features among {} new features".format(
                    len(features_now), len(features_new)
                )
            )

        # Compute probabilities between all successor candidate combinations
        if debug:
            log.debug("find all successor candidates")
        all_candidates = self.successor_candidates(features_now, features_new)
        if debug:
            log.debug(" all_candidates: {}".format(all_candidates))
        if debug:
            log.debug("compute all successor candidate probabilities")
        probabilities = self.compute_successor_probabilities(all_candidates)

        # SR_DBG<
        # if len(features_now) > 0 and "Mean" in features_now[0].__class__.__name__:
        #    ipython(globals(), locals())
        # SR_DBG>

        # LOG: successor probabilities
        self._log_successor_probabilities(probabilities)

        # Assign successors to tracks, highest probability first
        self.assign_successors(probabilities)

    # Handle missing features

    def add_dummy_events(self, timestep):
        for event in self.tracker.track_heads():
            if event.timestep() < timestep:
                if not event.is_dummy():
                    self._add_dummy_event(event, timestep)

    def _add_dummy_event(self, event, timestep):
        """Add a dummy event if the branch might still be continued."""
        debug = False

        if debug:
            log.debug(" => current event is not dummy: {}".format(event))
        dummy_feature = event.feature().as_mean()

        if DBG:
            log.debug("Create new dummy for {}".format(event.feature()))
        track = event.track()
        event_new = FeatureTrackEventDummyContinuation(
            track=track, feature=dummy_feature, timestep=timestep
        )

        if DBG:
            log.debug("Add dummy {} to track {}".format(event_new, track))
        event.link_forward(event_new)
        track.update_heads(event_del=event, event_add=event_new)

    def handle_dummy_events(self, timestep, features_leftover):
        debug = False

        # Identify dummy events and features
        is_dummy = lambda e: e.is_dummy() and e.timestep() < timestep
        dummy_events = [e for e in self.tracker.track_heads() if is_dummy(e)]
        dummy_features = [e.feature() for e in dummy_events]

        # Identify and assign successors across dummy events
        self.find_successors(dummy_features, features_leftover)

        # Clean up
        for event in dummy_events.copy():
            if event.next():
                # Add new features to mean features at dummy events
                for event_next in event.next():
                    if debug:
                        log.debug("FOO {}".format(event))
                    event.feature().add_feature(event_next.feature())
            else:
                # Remove dummy events without successors
                track = event.track()
                events_prev = event.prev()
                # Note: event must first be removed from track, such that
                # events_prev have no more successors, and only then can the
                # events_prev be added back to the heads! If they are still
                # connected to event, they are silently rejected as heads...
                # SR_TODO: Reconsider this; just remove heads check?
                track.remove_event(event)
                for event_prev in events_prev:
                    track.update_heads(event_add=event_prev)
                dummy_events.remove(event)

        self._merge_dummy_events_at_merging(dummy_events)

    def _merge_dummy_events_at_merging(self, dummy_events):
        """Merge dummy events adjacent to the same merging event.

        If two dummy events are the predecessors of a merging event, they
        are merged into a single dummy-merging event in order to reduce
        the number of dummy events.

        Note that dummy-splitting events don't require such explicit
        treatment, as they are naturally handled correctly.
        """
        pairs = itertools.combinations(dummy_events, 2)
        for event1, event2 in pairs:

            # Determine adjacent features and timesteps
            features1 = event1.feature().features()
            features2 = event2.feature().features()

            # Sort features by timestep
            ts1 = max([f.event().timestep() for f in features1])
            f1_ts1 = [f for f in features1 if f.event().timestep() == ts1]
            f2_ts1 = [f for f in features2 if f.event().timestep() == ts1]

            # Check for merging
            if set(f1_ts1) != set(f2_ts1):
                continue

            if len(f1_ts1) > 1:
                err = "missing feature: merge dummy events: multiple"
                raise NotImplementedError(err)

            event_next_del = f1_ts1[0].event()
            events_prev = event1.prev() + event2.prev()
            features = set(event1.feature().features() + event2.feature().features())
            track = event_next_del.track()

            # Unlink the old dummy events
            event1.unlink()
            event2.unlink()

            # Replace the merging event by a continuation
            event_next_add = event_next_del.as_non_merging()
            track.replace_event(event_del=event_next_del, event_add=event_next_add)

            # Turn one of the dummy events into a dummy-merging and relink it
            dummy_event = event1.as_merging()
            track.replace_event(event_del=event1, event_add=dummy_event)
            track.remove_event(event2)
            dummy_event.link(prev=events_prev, next=[event_next_add])

            # The manual heads update should not be necessary... Not nice!
            track.update_heads(event_del=dummy_event, event_add=event_next_add)

            # Merge the features of the dummy events
            for feature in event2.feature().features():
                if feature not in dummy_event.feature().features():
                    dummy_event.feature().add_feature(feature)

    # Debug log

    def _log_table_tracks(self, name, tracks):
        if DBG:
            log.debug("{name}: {n}".format(name=name, n=len(tracks)))
        if len(tracks) == 0:
            return
        if DBG:
            log.debug("{id:>3} {n:>3}".format(id="id", n="n"))
        for track in tracks:
            if DBG:
                log.debug("{id:3} {n:3}".format(id=track.id(), n=track.n()))

    def _log_table_features(self, features):
        name = "\nfeatures_new"
        if DBG:
            log.debug("{name}: {n}".format(name=name, n=len(features)))
        if len(features) == 0:
            return
        if DBG:
            log.debug(
                "{id:>3} {lon:>7} {lat:>7} {a:>10}".format(
                    id="id", lon="lon", lat="lat", a="area"
                )
            )
        for feature in features:
            lon, lat = feature.center()
            if DBG:
                log.debug(
                    "{id:>3} {lon:7.3f} {lat:7.3f} {a:10.3e}".format(
                        id=feature.id_str(), lon=lon, lat=lat, a=feature.area()
                    )
                )

    def _log_table_heads(self, features):
        name = "\nfeatures_now"
        if DBG:
            log.debug("{name}: {n}".format(name=name, n=len(features)))
        if len(features) == 0:
            return
        if DBG:
            log.debug(
                "{tid:>3} {fid:>3} {lon:>7} {lat:>7} {a:>10}".format(
                    tid="tid", fid="fid", lon="lon", lat="lat", a="area"
                )
            )
        for feature in features:
            lon, lat = feature.center()
            tid, fid = feature.event().track().id(), feature.id_str()
            if DBG:
                log.debug(
                    ("{tid:>3} {fid:3} {lon:7.3f} {lat:7.3f} {a:10.3e}").format(
                        tid=tid, fid=fid, lon=lon, lat=lat, a=feature.area()
                    )
                )

    def _log_successor_probabilities(self, probabilities):
        if DBG:
            log.debug("\nsuccessor probabilities:")
        if len(probabilities) == 0:
            if DBG:
                log.debug("  NONE\n")
            return
        if DBG:
            log.debug(
                "{id0:>8} {dir:2} {id1:>8} {pt:>5} {po:>5} {pa:>5}".format(
                    id0="id0", dir="<>", id1="id1", pt="p_tot", po="p_ovl", pa="p_are"
                )
            )
        for prob_tot, prob_overlap, prob_area, feat0, feat1 in probabilities:
            if feat0.event():
                now, new = feat0, feat1
                arrow = "=>"
            else:
                now, new = feat1, feat0
                arrow = "<="
            if DBG:
                log.debug(
                    (
                        "{id0:>8} {dir:2} {id1:>8} {pt:5.2f} {po:5.2f} " "{pa:5.2f}"
                    ).format(
                        id0=now.id_str(),
                        dir=arrow,
                        id1=new.id_str(),
                        pt=prob_tot,
                        po=prob_overlap,
                        pa=prob_area,
                    )
                )

    # Compute successor probabilities

    def compute_successor_probabilities(self, features_candidates):
        """Compute the successor probabilities for a set of features.

        The input is a dictionary with single features as keys, and lists
        of the corresponding successor candidate features as values.

        Return a list of tuples (Pt, Po, Pa, X, Y), where:
         - Pt: total probability that Y succeeds Y
         - Po: overlap component of total probability
         - Pa: area component of total probability
         - X: single feature (the "keys" in the input dictionary)
         - Y: either one feature, or a combination of multiple

        If Y represents a combination of multiple features, it is a
        FeatureCombination object.

        The list is sorted reversely by probability (highest first).
        """
        debug = False
        if debug:
            log.debug(
                "< compute_successor_probabilities {}".format(features_candidates)
            )
        probabilities = []
        for feature, candidates in features_candidates.items():
            combinations = self.successor_candidate_combinations(candidates)
            if debug:
                log.debug(
                    "[{}] found {} combinations among {} candidates".format(
                        feature.id_str(), len(combinations), len(candidates)
                    )
                )
            for i, candidate in enumerate(combinations):
                probs = self.successor_probability(feature, candidate)
                # DBG_BLOCK<
                if debug:
                    try:
                        n0 = len(feature.features())
                    except AttributeError:
                        n0 = 1
                    try:
                        n1 = len(candidate.features())
                    except AttributeError:
                        n1 = 1
                    if debug:
                        log.debug(
                            " {:5} /{:5} : {:2}->{:2} tot {:5.3} overlap {:5.3} area {:5.3}".format(
                                i + 1, len(combinations), n0, n1, *probs[:3]
                            )
                        )
                # DBG_BLOCK>
                if probs[0] > 0:
                    probabilities.append(probs)
        probabilities.sort(reverse=True)
        return probabilities

    def successor_probability(self, feature, candidate):
        debug = False
        f_overlap = self.conf["f_overlap"]
        f_area = self.conf["f_area"]
        prob_area_min = self.conf["prob_area_min"]
        prob_area = self.successor_probability_area(feature, candidate, debug)
        if prob_area_min > 0 and prob_area < prob_area_min:
            prob_overlap, prob_tot = 0.0, 0.0
        else:
            prob_overlap = self.successor_probability_overlap(feature, candidate, debug)
            prob_tot = f_overlap * prob_overlap + f_area * prob_area
        return prob_tot, prob_overlap, prob_area, feature, candidate

    def successor_probability_overlap(self, obj1, obj2, debug=False):
        try:
            fraction = obj1.overlap_fraction(obj2)
        except ZeroDivisionError:
            if debug:
                log.debug(" - overlap: 0.0 (ZeroDivisionError)")
            return 0.0
        min_overlap = self.conf["min_overlap"]
        if fraction < min_overlap:
            if debug:
                log.debug(" - overlap: 0.0 ({} < {})".format(fraction, min_overlap))
            return 0.0
        prob_overlap = (fraction - min_overlap) / (1 - min_overlap)
        if debug:
            log.debug(
                " - overlap: {} (({} - {})/(1 - {}))".format(
                    prob_overlap, fraction, min_overlap, min_overlap
                )
            )
        return prob_overlap

    def successor_probability_area(self, obj1, obj2, debug=False):
        try:
            ratio = obj1.area_ratio(obj2)
        except ZeroDivisionError:
            if debug:
                log.debug(" - area: 0.0 (ZeroDivisionError)")
            return 0.0
        max_area = self.conf["max_area"]
        prob_area = max(max_area - ratio, 0) / (max_area - 1)
        if debug:
            log.debug(
                " - area: {} (max({} - {}, 0)/({} - 1))".format(
                    prob_area, max_area, ratio, max_area
                )
            )
        return prob_area

    # Find and combine successor candidates

    def successor_candidate_combinations(self, candidates):
        """
        powerset([1, 2, 3])

                |
                v

        () (1, ) (2, ) (3, ) (1, 2) (1, 3) (2, 3) (1, 2, 3)"

        src: https://docs.python.org/2/library/itertools.html
        """
        debug = False
        max_children = self.conf["max_children"]
        nmax = len(candidates) if max_children < 0 else max_children
        if debug:
            log.debug(
                "< successor_candidate_combinations {} (nmax={})".format(
                    candidates, nmax
                )
            )
        combinations = list(
            itertools.chain.from_iterable(
                itertools.combinations(candidates, r) for r in range(nmax + 1)
            )
        )
        feature_combinations = []
        for features in combinations:
            if len(features) == 0:
                continue
            elif len(features) == 1:
                feature_combinations.append(features[0])
            else:
                # cls_combination = FeatureCombination
                cls_combination = features[0].cls_combination
                feature_combinations.append(cls_combination(features))
        return feature_combinations

    def successor_candidates(self, features1, features2):
        """Identify all successor candidates between two sets of features.

        For every feature in the first list, find all successor candidates
        in the second list, and vice versa.

        Return a dictionary of lists.
        """
        debug = False
        if debug:
            log.debug("< successor_candidates for {}: {}".format(features1, features2))
        cand = {}
        for obj1 in features1:
            cand[obj1] = self._successor_candidates_oneside(obj1, features2)
        for obj2 in features2:
            cand[obj2] = self._successor_candidates_oneside(obj2, features1)
        return cand

    def _successor_candidates_oneside(self, obj1, features2):
        cand = []
        for obj2 in features2:
            try:
                if self.is_successor_candidate(obj1, obj2):
                    cand.append(obj2)
            except Exception as e:
                err = (
                    "Successor candidate check of events "
                    "{} and {} failed with {}({})! Skipping!"
                ).format(obj1, obj2, e.__class__.__name__, e)
                log.error(err)
                continue
        return cand

    def is_successor_candidate(self, obj1, obj2):
        """Check whether a feature is a successor candidate of another feature.

        To be a succesor candidate for a feature, another feature must
         - overlap with the feature (aka. intersect it), or
         - be at most f*r_eq away from the feature (f = tuning factor).

        Thereby, r_eq is the equivalent radius, i.e. the radius of a circle
        with the same area as the feature. The distance is measured between
        the centers of the two features in question.
        """
        debug = False
        require_overlap = self.conf["require_overlap"]
        max_dist_rel = self.conf["max_dist_rel"]
        max_dist_abs = self.conf["max_dist_abs"]
        if debug:
            log.debug("< is_successor_candidate: {} -> {}".format(obj1.id(), obj2.id()))

        # Check overlap
        if obj1.intersects(obj2):
            return True

        # Check distance based on equivalent radius thresholds
        dist = obj1.center_distance(obj2)
        r_eq = obj1.radius()
        if max_dist_rel > 0:
            if dist > max_dist_rel * r_eq:
                if debug:
                    log.debug(
                        " => NO! dist > max_dist_rel*r_eq: {} > {}*{}={}".format(
                            dist, max_dist_rel, r_eq, max_dist_rel * r_eq
                        )
                    )
                return False
            elif not require_overlap:
                if debug:
                    log.debug(
                        " => YES! dist <= max_dist_rel*r_eq: {} <= {}*{}={}".format(
                            dist, max_dist_rel, r_eq, max_dist_rel * r_eq
                        )
                    )
                return True
        elif max_dist_abs > 0:
            if dist > max_dist_abs:
                if debug:
                    log.debug(
                        " => NO! dist > max_dist_abs: {} > {}".format(
                            dist, max_dist_abs
                        )
                    )
                return False
            elif not require_overlap:
                if debug:
                    log.debug(
                        " => YES! dist > max_dist_abs: {} <= {}".format(
                            dist, max_dist_abs
                        )
                    )
                return True

        return False

    # Assign successors to tracks

    def assign_successors(self, probabilities):
        def mark_distributed(distributed, features):
            # cls_combination = FeatureCombination
            cls_combination = features[0].cls_combination
            for feature in features:
                if isinstance(feature, cls_combination):
                    mark_distributed(distributed, feature.features())
                else:
                    distributed.add(feature)

        def already_distributed(distributed, features):
            # cls_combination = FeatureCombination
            cls_combination = features[0].cls_combination
            for feature in features:
                if isinstance(feature, cls_combination):
                    if already_distributed(distributed, feature.features()):
                        return True
                else:
                    if feature in distributed:
                        return True
            return False

        distributed = set()
        for prob_tot, prob_overlap, prob_area, feat1, feat2 in probabilities:
            # cls_combination = FeatureCombination
            cls_combination = feat1.cls_combination

            # Abort if overall probability drops below threshold
            if prob_tot < self.conf["threshold"]:
                return

            # Check whether both features are still available
            if already_distributed(distributed, [feat1, feat2]):
                continue

            # Check if one of the features is not yet associated with an
            # event, which then must be a still-unassigned "NEW" feature.
            if feat1.is_unassigned() and feat2.is_unassigned():
                err = "both features not assigned: {}, {}".format(
                    feat1.id_str(), feat2.id_str()
                )
                raise Exception(err)
            elif feat2.is_unassigned():
                feature_now, feature_new = feat1, feat2
            elif feat1.is_unassigned():
                feature_now, feature_new = feat2, feat1
            else:
                # Both have an assigned event, i.e. nothing to assign!
                continue

            # Handle merging event
            if isinstance(feature_now, cls_combination):
                self._assign_successors_merging(feature_now, feature_new)
                mark_distributed(distributed, [feature_now, feature_new])
                continue

            # Handle splitting event
            if isinstance(feature_new, cls_combination):
                self._assign_successors_splitting(feature_now, feature_new)
                mark_distributed(distributed, [feature_now, feature_new])
                continue

            # Handle continuation event
            self._assign_successors_continuation(feature_now, feature_new)
            mark_distributed(distributed, [feature_now, feature_new])

    def _assign_successors_merging(self, fnow, fnew):
        events_now = [feature.event() for feature in fnow.features()]
        tracks = set(event.track() for event in events_now)
        track = self.tracker.merge_tracks(tracks, feature_id=fnow.id_str())
        event_new = self.event_factory.merging(track, fnew, self.tracker.timestep())
        for event_now in events_now:
            event_now.link_forward(event_new)
            track.update_heads(event_del=event_now, event_add=event_new)

        if DBG:
            log.debug(
                (
                    "continue tracks {t0} at features {f0} as track #{t1} "
                    "with feature #{f1}"
                ).format(
                    t0=", ".join(["#{}".format(t.id()) for t in tracks]),
                    f0=", ".join(["#{}".format(f.id()) for f in fnow.features()]),
                    t1=track.id(),
                    f1=event_new.feature().id(),
                )
            )

    def _assign_successors_splitting(self, fnow, fnew):
        event_now_old = fnow.event()
        track = event_now_old.track()
        event_now_new = event_now_old.as_splitting()
        events_new = []
        for feature in fnew.features():
            new_event = self.event_factory.continuation(
                track, feature, self.tracker.timestep()
            )
            events_new.append(new_event)
        for event in events_new:
            event_now_new.link_forward(event)
        feature_ids_new = [e.feature().id() for e in events_new]
        track.replace_event(event_del=event_now_old, event_add=event_now_new)

        if DBG:
            log.debug(
                (
                    "continue track #{tid} after feature #{oid} " "with features {nid}"
                ).format(
                    tid=track.id(),
                    oid=fnow.id(),
                    nid=", ".join(["#{}".format(i) for i in feature_ids_new]),
                )
            )

    def _assign_successors_continuation(self, feature_now, feature_new):
        event_now = feature_now.event()
        track = event_now.track()
        tracker = self.tracker
        event_new = self.event_factory.continuation(
            track, feature_new, tracker.timestep()
        )

        event_now.link_forward(event_new)

        track.update_heads(event_del=event_now, event_add=event_new)

        if DBG:
            log.debug(
                (
                    "continue track #{tid} after feature #{oid} " "with feature #{nid}"
                ).format(tid=track.id(), oid=feature_now.id(), nid=feature_new.id())
            )


class FeatureTrackIOReaderJson(IOReaderJsonBase):

    cls_feature = TrackableFeatureBase
    cls_event = FeatureTrackEventBase
    cls_track = FeatureTrackBase

    # def __init__(self, feature_factory, event_factory, track_factory):
    def __init__(self, *args, domain=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = domain
        # self.feature_factory = feature_factory
        # self.event_factory = event_factory
        # self.track_factory = track_factory

    def read_string(self, jstr, include_tracker_config=False):
        jdat = json.loads(jstr, object_pairs_hook=OrderedDict)

        if "HEADER" in jdat:
            self._header = jdat["HEADER"]

        data = self.rebuild_tracks_full(jdat)

        return data

    def rebuild_tracks_full(self, jdat):

        data = OrderedDict()

        if "FEATURES" in jdat and jdat["FEATURES"]:
            data["FEATURES"] = self.rebuild_features(
                jdat["FEATURES"], domain=self.domain
            )
        else:
            data["FEATURES"] = []

        if "EVENTS" in jdat and jdat["EVENTS"]:
            if "FEATURES" not in data:
                err = "No features data available to re-build events!"
                raise Exception(err)
            data["EVENTS"] = self.rebuild_events(jdat["EVENTS"], data["FEATURES"])
        else:
            data["EVENTS"] = []

        if "TRACKS" in jdat and jdat["TRACKS"]:
            if "EVENTS" not in data:
                err = "No events data available to re-build tracks!"
                raise Exception(err)
            config = jdat.get("CONFIG", {}).get("TRACKER", {})
            data["TRACKS"] = self.rebuild_tracks(jdat["TRACKS"], data["EVENTS"], config)
        else:
            data["TRACKS"] = []

        return data

    def rebuild_features(self, *args, **kwargs):
        err = "Method rebuild_features must be overridden by {}".format(
            self.__class__.__name__
        )
        raise Exception(err)

    def rebuild_events(self, data, features):
        events = {}
        prev_ids = {}
        next_ids = {}
        for event_data in data:
            event_id = event_data.pop("id")
            prev_ids[event_id] = event_data.pop("prev")
            next_ids[event_id] = event_data.pop("next")
            fid = event_data.pop("feature")
            for feature in features:
                if feature.id() == fid:
                    break
            # event = self.event_factory.from_class_string(
            del event_data["track"]

            timestep = event_data.pop("timestep")

            # Convert timestep to datetime, if necessary
            # SR_TODO: Consider checking the respective config param
            # SR_TODO: instead of this brute-force approach
            try:
                timestep = datetime.strptime(str(timestep), "%Y%m%d%H")
            except ValueError:
                pass

            event = self.__class__.cls_event.factory().from_class_string(
                cls_str=event_data.pop("class"),
                track=None,
                feature=feature,
                timestep=timestep,
                id=event_id,
                **event_data
            )
            events[event_id] = event
        for id, next in next_ids.items():
            for nid in next:
                events[id].link_forward(events[nid])
        return list(events.values())

    def rebuild_tracks(self, data, events, config):

        dts = config.get("delta_timestep", 1)
        if config.get("timestep_datetime", False):
            dts = timedelta(hours=dts)

        fetch_events = lambda ids: [e for e in events if e.id() in ids]
        tracks = []
        for track_data in data:
            kw = {"delta_timestep": dts}
            kw["id"] = track_data["id"]
            kw["events"] = fetch_events(track_data["events"])
            for key in ["starts", "ends", "heads"]:
                if key in track_data:
                    kw[key] = fetch_events(track_data[key])
            # track = self.track_factory.rebuild(**kw)
            track = self.__class__.cls_track.factory().rebuild(**kw)
            tracks.append(track)
        return tracks


class FeatureTrackIOWriterJson(IOWriterJsonBase):

    section_name_dict = {"tracker": "TRACKER"}

    # SR_TODO Adapt to TrackWriter (copied from CycloneWriter)
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
            "TRACKS": self.write_string_tracks_info,
            "EVENTS": self.write_string_events,
            "FEATURES": self.write_string_features,
        }
        try:
            register.update(new_register)
        except AttributeError:
            register = new_register
        return super()._write_string_method(name, register)

    def write_string_tracks_info(self, tracks):
        name = "TRACKS"  # SR_TODO
        log.info("write tracks to {n}".format(n=name))
        return self.write_string_objs_info(name, tracks, 2, 4)

    def write_string_events(self, events):
        name = "EVENTS"  # SR_TODO
        log.info("write events to {n}".format(n=name))
        return self.write_string_objs_info(name, events, 2, 4)

    def write_string_features(self, features, save_paths=True):
        name = "FEATURES"  # SR_TODO
        log.info("write features to {n}".format(n=name))
        return self.write_string_objs_info(name, features, 2, 4, path=save_paths)

    def add_tracks(self, tracks):
        self._add_to_cache("TRACKS", tracks)
        for track in tracks:
            self.add_events(track.events())
            self.add_features(track.features())

    def add_events(self, events):
        self._add_to_cache("EVENTS", events)

    def add_features(self, features):
        self._add_to_cache("FEATURES", features)
