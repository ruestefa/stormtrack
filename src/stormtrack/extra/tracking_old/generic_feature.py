#!/usr/bin/env python3

# Standard library
import logging as log

# Third-party
import numpy as np
import scipy as sp
import shapely.geometry as geo

# Local
from .tracking import FeatureFactory
from .tracking import FeatureTrackBase
from .tracking import FeatureTrackFactory
from .tracking import FeatureTrackIOReaderJson
from .tracking import FeatureTrackIOWriterJson
from .tracking import FeatureTrackerBase
from .tracking import PeriodicTrackableFeatureBase
from .tracking import TrackableFeatureBase


class GenericFeature(TrackableFeatureBase):

    cls_factory = None  # GenericFeatureFactory defined below

    def __init__(
        self,
        path,
        *,
        id=-1,
        timestep=None,
        domain=None,
        track=None,
        event=None,
        center=None,
        attr=None,
    ):
        self._poly = geo.Polygon(path)
        self._center = center
        super().__init__(id=id, track=track, event=event, domain=domain, attr=attr)

    def __repr__(self):
        return "<{cls}: id={id:3}, xy={xy}, a={a:5.1f}>".format(
            cls=self.__class__.__name__,
            id=self.id(),
            a=self.area(),
            xy="({:5.1f}, {:5.1f})".format(*self.center()),
        )

    # Methods that need to be overridden by the subclass

    def copy(self, path=None, id_=None, event=None, domain=None):
        if not path:
            self.as_polygon().boundary.coords
        if not id_:
            id_ = -1
        return self.__class__(path, id=id_, event=event, domain=domain)

    def as_polygon(self):
        return self._poly

    def intersection(self, other):
        return self.as_polygon().intersection(other.as_polygon())

    def area(self):
        return self.as_polygon().area

    def overlap_area(self, other):
        overlap = self.intersection(other)
        try:
            return overlap.area()
        except TypeError:
            return overlap.area

    def center(self):
        if self._center is not None:
            return self._center
        c = self.as_polygon().centroid
        return np.array([c.x, c.y])

    def radius(self):
        return np.sqrt(self.area() / np.pi)


class GenericFeatureFactory(FeatureFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


GenericFeature.cls_factory = GenericFeatureFactory


class PeriodicGenericFeature(PeriodicTrackableFeatureBase, GenericFeature):
    def __init__(self, features, domain, **kwargs):
        GenericFeature.__init__(self, None, **kwargs)
        PeriodicTrackableFeatureBase.__init__(self, features, domain)


PeriodicGenericFeature.cls_default = GenericFeature
GenericFeature.cls_periodic = PeriodicGenericFeature


class GenericFeatureTracker(FeatureTrackerBase):
    def __init__(
        self,
        min_overlap=None,
        max_area=None,
        f_overlap=0,
        f_area=0,
        max_children=-1,
        prob_area_min=0.0,
        threshold=0.5,
        require_overlap=False,
        max_dist_abs=-1.0,
        max_dist_rel=2.0,
        timestep=0,
        delta_timestep=1,
        timestep_datetime=False,
        track_id=0,
    ):

        super().__init__(
            min_overlap=min_overlap,
            max_area=max_area,
            threshold=threshold,
            require_overlap=require_overlap,
            f_overlap=f_overlap,
            f_area=f_area,
            max_children=max_children,
            prob_area_min=prob_area_min,
            max_dist_abs=max_dist_abs,
            max_dist_rel=max_dist_rel,
            timestep=timestep,
            timestep_datetime=timestep_datetime,
            delta_timestep=delta_timestep,
            # track_id            = track_id,
        )

    # Methods that need to be overridden by the subclass

    def new_track(self, feature, timestep, id):
        return GenericFeatureTrack(feature=feature, timestep=timestep, id=id)


class GenericFeatureTrack(FeatureTrackBase):
    def __init__(self, feature=None, id=-1, timestep=0):
        super().__init__(feature=feature, id=id, timestep=timestep)


class GenericFeatureTrackFactory(FeatureTrackFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


GenericFeatureTrack.cls_factory = GenericFeatureTrackFactory


class GenericFeatureTrackIOReaderJson(FeatureTrackIOReaderJson):

    cls_feature = GenericFeature
    cls_track = GenericFeatureTrack

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def rebuild_features(self, data, domain=None):
        factory = self.cls_feature.factory()
        features = []
        for feature_data in data:
            cls_str = feature_data.pop("class")
            if domain:
                feature_data["domain"] = domain
            if "feature_path_file" in self._header:
                raise NotImplementedError
            # Extract attributes
            attr = {}
            for key, val in feature_data.copy().items():
                if key.startswith("attr_"):
                    del feature_data[key]
                    attr[key[5:]] = val
            feature_data["attr"] = attr
            features.append(factory.from_class_string(cls_str, **feature_data))
        return features


class GenericFeatureTrackIOWriterJson(FeatureTrackIOWriterJson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
