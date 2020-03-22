#!/usr/bin/env python3

# Standard library
import pytest
import sys
import unittest
from unittest import TestCase
from collections import OrderedDict as odict
from copy import deepcopy

# Third-party
import numpy as np

# First-party
from stormtrack.core.tracking import FeatureTracker
from stormtrack.core.tracking import remerge_partial_tracks

# Local
from ...utils import circle
from ...utils import feature_circle
from ...utils import TestTracks_Base


class SimpleTrack__Base(TestTracks_Base):
    """Build simple tracks (i.e. no merging/splitting)."""

    plot = False

    def track1(s):
        return [
            None,
            feature_circle(9, 5, 4, 0, 1),
            feature_circle(10, 8, 6, 1, 2),
            feature_circle(13, 12, 8, 2, 3),
            feature_circle(16, 16, 10, 3, 4),
            feature_circle(21, 20, 12, 4, 5),
            feature_circle(28, 22, 13, 5, 6),
            feature_circle(36, 23, 14, 6, 7),
            feature_circle(44, 23, 14, 7, 8),
            feature_circle(51, 26, 13, 8, 9),
            feature_circle(56, 32, 10, 9, 10),
            feature_circle(58, 36, 7, 10, 11),
            feature_circle(59, 40, 5, 11, 12),
            feature_circle(60, 42, 4, 12, 13),
            None,
        ]

    def track2(s):
        return [
            None,
            None,
            None,
            feature_circle(39, 49, 4, 21, 3),
            feature_circle(44, 48, 4, 22, 4),
            feature_circle(49, 46, 5, 23, 5),
            feature_circle(54, 44, 5, 24, 6),
            feature_circle(59, 41, 6, 25, 7),
            feature_circle(64, 38, 7, 26, 8),
            feature_circle(69, 35, 6, 27, 9),
            feature_circle(74, 32, 5, 28, 10),
            feature_circle(79, 28, 5, 29, 11),
            feature_circle(82, 23, 4, 30, 12),
            None,
            None,
        ]

    def track3(s):
        return [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            feature_circle(20, 40, 4, 13, 7),
            feature_circle(22, 41, 4, 14, 8),
            feature_circle(24, 42, 5, 15, 9),
            feature_circle(28, 44, 6, 16, 10),
            feature_circle(33, 46, 8, 17, 11),
            feature_circle(40, 51, 11, 18, 12),
            feature_circle(43, 56, 11, 19, 13),
            feature_circle(45, 63, 12, 20, 14),
        ]

    def setUp(s):

        # Choose big enough to all features defined above fully fit!
        nx, ny = 200, 200

        s.features12 = lambda: [s.track1(), s.track2()]
        s.features23 = lambda: [s.track2(), s.track3()]
        s.features123 = lambda: [s.track1(), s.track2(), s.track3()]

        s.objs12 = lambda: [
            [f for f in t if f is not None] for t in list(zip(*s.features12()))
        ]

        s.objs23 = lambda: [
            [f for f in t if f is not None] for t in list(zip(*s.features23()))
        ]

        s.objs123 = lambda: [
            [f for f in t if f is not None] for t in list(zip(*s.features123()))
        ]

        if s.plot:
            raise NotImplementedError("{}.plot".format(s.__class__.__name__))
            objs = [o for t in s.objs123() for o in t]
            plot_contours("tracks_simple.png", objs)

        s.tracker = FeatureTracker(
            f_overlap=0.5,
            f_size=0.5,
            max_children=10,
            min_p_tot=0.0,
            min_p_overlap=0.0,
            min_p_size=0.0,
            minsize=0,
            maxsize=0,
            connectivity=8,
            split_tracks_n=-1,
            merge_features=False,
            nx=nx,
            ny=ny,
        )


class SimpleTrack(SimpleTrack__Base):
    def test_single_track_1(s):
        features = s.track1()
        features = [[f] if f else [] for f in features]
        tracks = s.run_tracking(features, event_id=0)
        features = [[f] for f in s.track1()]
        s.assert_tracks_features(tracks, [features])

    def test_single_track_2(s):
        features = s.track2()
        features = [[f] if f else [] for f in features]
        tracks = s.run_tracking(features, event_id=21)
        features = [[f] for f in s.track2()]
        s.assert_tracks_features(tracks, [features])

    def test_single_track_3(s):
        features = s.track3()
        features = [[f] if f else [] for f in features]
        tracks = s.run_tracking(features, event_id=13)
        features = [[f] for f in s.track3()]
        s.assert_tracks_features(tracks, [features])

    def test_two_tracks_12(s):
        features = s.objs12()
        tracks = s.run_tracking(features)
        features = [[[f] for f in ff if f] for ff in s.features12()]
        s.assert_tracks_features(tracks, features)

    def test_two_tracks_23(s):
        features = s.objs23()
        tracks = s.run_tracking(features)
        features = [[[f] for f in ff if f] for ff in s.features23()]
        s.assert_tracks_features(tracks, features)

    def test_all_tracks(s):
        features = s.objs123()
        tracks = s.run_tracking(features)
        features = [[[f] for f in ff if f] for ff in s.features123()]
        s.assert_tracks_features(tracks, features)


class SimpleTrack__SplitMergeTrack(SimpleTrack__Base):
    """Split and remerge track."""

    def split_track(s, tracks_split, ts):
        tracks_cut = []
        for track in tracks_split:
            partial_track = track.cut_off(until=ts)
            tracks_cut.append(partial_track)
        return tracks_cut

    def _test_split_once__prepare_tracks(s, ts):

        # Create tracks
        tracks_in = s.run_tracking(s.objs123())
        n_tracks = len(tracks_in)

        tracks_split = deepcopy(tracks_in)
        s.assertEqual(tracks_in, tracks_split)
        s.assertTrue(all([t.is_complete() for t in tracks_in]))
        s.assertTrue(all([t.is_complete() for t in tracks_split]))

        # Cut tracks apart
        tracks_cut = s.split_track(tracks_split, ts)
        s.assertEqual(len(tracks_cut), n_tracks)

        # Check consistency of track sizes
        for track_split, track_cut in zip(tracks_split, tracks_cut):
            n_s_f = track_split.size(total=False)
            n_c_f = track_cut.size(total=False)
            n_s_t = track_split.size(total=True)
            n_c_t = track_cut.size(total=True)
            s.assertEqual(n_s_t, n_c_t)
            s.assertEqual(n_s_f + n_c_f, n_s_t)

        # Ensure same track order
        tracks_in.sort(key=lambda t: t.id)
        tracks_split.sort(key=lambda t: t.id)
        tracks_cut.sort(key=lambda t: t.id)

        return tracks_in, tracks_split, tracks_cut

    def _test_split_twice__prepare_tracks(s, ts_1, ts_2):

        # Create tracks
        tracks_in = s.run_tracking(s.objs123())
        tracks_split = deepcopy(tracks_in)
        s.assertEqual(tracks_in, tracks_split)
        s.assertTrue(all([t.is_complete() for t in tracks_in]))
        s.assertTrue(all([t.is_complete() for t in tracks_split]))

        # Cut tracks apart
        tracks_cut_1 = s.split_track(tracks_split, ts_1)
        s.assertEqual(len(tracks_cut_1), 3)

        # Check consistency of track sizes
        for track_split, track_cut_1 in zip(tracks_split, tracks_cut_1):
            n_sp_f = track_split.size(total=False)
            n_c1_f = track_cut_1.size(total=False)
            n_sp_t = track_split.size(total=True)
            n_c1_t = track_cut_1.size(total=True)
            s.assertEqual(n_sp_t, n_c1_t)
            s.assertEqual(n_sp_f + n_c1_f, n_sp_t)

        # Cut tracks apart again
        tracks_cut_2 = s.split_track(tracks_split, ts_2)
        s.assertEqual(len(tracks_cut_2), 3)

        # Check consistency of track sizes
        for track_split, track_cut_1, track_cut_2 in zip(
            tracks_split, tracks_cut_1, tracks_cut_2
        ):
            n_sp_f = track_split.size(total=False)
            n_c1_f = track_cut_1.size(total=False)
            n_c2_f = track_cut_2.size(total=False)
            n_sp_t = track_split.size(total=True)
            n_c1_t = track_cut_1.size(total=True)
            n_c2_t = track_cut_2.size(total=True)
            s.assertEqual(n_sp_t, n_c1_t)
            s.assertEqual(n_sp_t, n_c2_t)
            s.assertEqual(n_sp_f + n_c1_f + n_c2_f, n_sp_t)

        return tracks_in, tracks_split, tracks_cut_1, tracks_cut_2

    def _test_split_once__track_properties__base(s, methods, kwas):

        ts_split = 8

        # Prepare tracks
        tracks_in, tracks_split, tracks_cut = s._test_split_once__prepare_tracks(
            ts_split
        )

        # Extract properties to compare
        def extract_props(tracks, methods, kwas):
            return [[getattr(t, m)(**kwas) for m in methods] for t in tracks]

        props_in = extract_props(tracks_in, methods, kwas)
        props_split = extract_props(tracks_split, methods, kwas)
        props_cut = extract_props(tracks_cut, methods, kwas)

        return props_in, props_split, props_cut

    def _test_split_twice__track_properties__base(s, methods, kwas):

        ts_split_1 = 8
        ts_split_2 = 10

        # Prepare tracks
        (
            tracks_in,
            tracks_split,
            tracks_cut_1,
            tracks_cut_2,
        ) = s._test_split_twice__prepare_tracks(ts_split_1, ts_split_2)

        # Extract properties to compare
        def extract_props(tracks, methods, kwas):
            return [
                odict([(m, getattr(t, m)(**kwas)) for m in methods]) for t in tracks
            ]

        props_in = extract_props(tracks_in, methods, kwas)
        props_split = extract_props(tracks_split, methods, kwas)
        props_cut_1 = extract_props(tracks_cut_1, methods, kwas)
        props_cut_2 = extract_props(tracks_cut_2, methods, kwas)

        return props_in, props_split, props_cut_1, props_cut_2

    def test_split_once__total_track_stats(s):
        """Check that total track stats are accurate after splitting once."""

        ts_split = 8

        # Prepare tracks
        tracks_in, tracks_split, tracks_cut = s._test_split_once__prepare_tracks(
            ts_split
        )

        # Collect total track stats
        tstats_in = [t.total_track_stats for t in tracks_in]
        tstats_split = [t.total_track_stats for t in tracks_split]
        tstats_cut = [t.total_track_stats for t in tracks_cut]

        # Compare total track stats
        try:  # SR_DBG
            s.assertEqual(tstats_split, tstats_in)
            s.assertEqual(tstats_cut, tstats_in)
        except AssertionError as e:  # SR_DBG
            # ipython(globals(), locals(), e) #SR_DBG
            raise

    def test_split_once__track_properties__total_default(s):
        """Check that some properties are accurate after splitting once."""

        methods = ["size", "duration", "ts_start", "ts_end"]
        kwas = dict()

        # Prepare tracks
        props_in, props_split, props_cut = s._test_split_once__track_properties__base(
            methods, kwas
        )

        # Compare properties
        s.assertEqual(props_split, props_in)
        s.assertEqual(props_cut, props_in)

    def test_split_once__track_properties__total_true(s):
        """Check that some properties are accurate after splitting once."""

        methods = ["size", "duration", "ts_start", "ts_end"]
        kwas = dict(total=True)

        # Prepare tracks
        props_in, props_split, props_cut = s._test_split_once__track_properties__base(
            methods, kwas
        )

        # Compare properties
        s.assertEqual(props_split, props_in)
        s.assertEqual(props_cut, props_in)

    def test_split_once__track_properties__total_false(s):
        """Check that some properties are accurate after splitting once."""

        methods = ["size", "duration", "ts_start", "ts_end"]
        kwas = dict(total=False)

        # Prepare tracks
        props_in, props_split, props_cut = s._test_split_once__track_properties__base(
            methods, kwas
        )

        # Compare properties
        s.assertNotEqual(props_split, props_in)
        s.assertNotEqual(props_cut, props_in)
        s.assertNotEqual(props_split, props_cut)

    def test_split_twice__total_track_stats(s):
        """Check that total track stats are accurate after splitting twice."""

        ts_split_1 = 8
        ts_split_2 = 10

        # Prepare tracks
        (
            tracks_in,
            tracks_split,
            tracks_cut_1,
            tracks_cut_2,
        ) = s._test_split_twice__prepare_tracks(ts_split_1, ts_split_2)

        # Collect total track stats
        tstats_in = [t.total_track_stats for t in tracks_in]
        tstats_split = [t.total_track_stats for t in tracks_split]
        tstats_cut_1 = [t.total_track_stats for t in tracks_cut_1]
        tstats_cut_2 = [t.total_track_stats for t in tracks_cut_2]

        # Compare total track stats
        s.assertEqual(tstats_split, tstats_in)
        s.assertEqual(tstats_cut_1, tstats_in)
        s.assertEqual(tstats_cut_2, tstats_in)

    def test_split_twice__track_properties__total_default(s):
        """Check that some properties are accurate after splitting twice."""

        methods = ["size", "duration", "ts_start", "ts_end"]
        kwas = dict()

        # Prepare tracks
        (
            props_in,
            props_split,
            props_cut_1,
            props_cut_2,
        ) = s._test_split_twice__track_properties__base(methods, kwas)

        # Compare properties
        s.assertEqual(props_split, props_in)
        s.assertEqual(props_cut_1, props_in)
        s.assertEqual(props_cut_2, props_in)

    def test_split_twice__track_properties__total_true(s):
        """Check that some properties are accurate after splitting twice."""

        methods = ["size", "duration", "ts_start", "ts_end"]
        kwas = dict(total=True)

        # Prepare tracks
        (
            props_in,
            props_split,
            props_cut_1,
            props_cut_2,
        ) = s._test_split_twice__track_properties__base(methods, kwas)

        # Compare properties
        s.assertEqual(props_split, props_in)
        s.assertEqual(props_cut_1, props_in)
        s.assertEqual(props_cut_2, props_in)

    def test_split_(s):
        """Check that some properties are accurate after splitting once."""

        methods = ["size", "duration", "ts_start", "ts_end"]
        kwas = dict(total=False)

        # Prepare tracks
        (
            props_in,
            props_split,
            props_cut_1,
            props_cut_2,
        ) = s._test_split_twice__track_properties__base(methods, kwas)

        # Compare properties
        s.assertNotEqual(props_split, props_in)
        s.assertNotEqual(props_cut_1, props_in)
        s.assertNotEqual(props_cut_2, props_in)
        s.assertNotEqual(props_cut_1, props_cut_2)
        s.assertNotEqual(props_split, props_cut_1)
        s.assertNotEqual(props_split, props_cut_2)

    def test_split_once__remerge(s):
        """Split tracks once at a certain timestep and remerge them."""

        ts_split = 8

        # Prepare tracks
        tracks_in, tracks_split, tracks_cut = s._test_split_once__prepare_tracks(
            ts_split
        )

        # Remerge tracks
        subtracks = tracks_cut + tracks_split
        tracks_out = remerge_partial_tracks(subtracks)

        # Compare tracks
        tracks_in.sort(key=lambda t: t.id)
        tracks_out.sort(key=lambda t: t.id)
        s.assertEqual(tracks_in, tracks_out)

    def test_split_twice__remerge_one_step(s):
        """Split tracks twice and remerge them in one step."""

        ts_split_1 = 8
        ts_split_2 = 10

        # Prepare tracks
        (
            tracks_in,
            tracks_split,
            tracks_cut_1,
            tracks_cut_2,
        ) = s._test_split_twice__prepare_tracks(ts_split_1, ts_split_2)

        # Remerge all tracks
        subtracks = tracks_cut_1 + tracks_cut_2 + tracks_split
        tracks_out = remerge_partial_tracks(subtracks)

        # Ensure same track order
        tracks_in.sort(key=lambda t: t.id)
        tracks_out.sort(key=lambda t: t.id)

        # Make sure tracks are complete etc.
        for track in tracks_out:
            s.assertTrue(track.is_complete())
            s.assertTrue(track.n_missing_features() == 0)

        # Compare tracks
        s.assertEqual(tracks_in, tracks_out)

    def test_split_twice__remerge_two_steps(s):
        """Split tracks twice and remerge them in two steps."""

        ts_split_1 = 8
        ts_split_2 = 10

        # Prepare tracks
        (
            tracks_in,
            tracks_split,
            tracks_cut_1,
            tracks_cut_2,
        ) = s._test_split_twice__prepare_tracks(ts_split_1, ts_split_2)

        # Remerge some tracks
        subtracks_1 = tracks_cut_1 + tracks_cut_2
        tracks_out_1 = remerge_partial_tracks(subtracks_1, is_subperiod=True)

        # Ensure same track order
        tracks_in.sort(key=lambda t: t.id)
        tracks_out_1.sort(key=lambda t: t.id)

        # Make sure tracks are not yet complete etc.
        for track in tracks_out_1:
            s.assertFalse(track.is_complete())
            s.assertTrue(track.n_missing_features() > 0)

        # Remerge remaining tracks
        subtracks_2 = tracks_out_1 + tracks_split
        tracks_out_2 = remerge_partial_tracks(subtracks_2)

        # Make sure tracks are now complete etc.
        for track in tracks_out_2:
            s.assertTrue(track.is_complete())
            s.assertTrue(track.n_missing_features() == 0)

        # Compare tracks
        s.assertEqual(tracks_in, tracks_out_2)


@pytest.mark.skip("not implemented")
class SimpleTrack_MissingFeature(TestTracks_Base):

    plot = False

    def setUp(s):

        s.track1 = [
            FeatureSimple(circle(2.1, 2.0, 1.2), id=0),
            FeatureSimple(circle(2.8, 2.2, 1.3), id=1),
            FeatureSimple(circle(3.6, 2.3, 1.4), id=2),
            FeatureSimple(circle(4.4, 2.3, 1.4), id=3),
            FeatureSimple(circle(5.1, 2.6, 1.3), id=4),
            FeatureSimple(circle(5.6, 3.2, 1.0), id=5),
        ]

        kwargs = dict(
            f_overlap=0.5,
            f_area=0.5,
            min_p_tot=0.0,
            min_p_overlap=0.0,
            min_p_size=0.0,
            max_area=2.0,
        )
        s.tracker0 = TrackerSimple(allow_missing=False, **kwargs)
        s.tracker1 = TrackerSimple(allow_missing=True, **kwargs)

    def test_no_missing(s):
        """Control test with all features."""
        features = [[f] if f else [] for f in s.track1]
        tracks = s.run_tracking(features, event_id=0, tracker=s.tracker0)
        features = [[f] for f in s.track1]
        s.assert_tracks_features(tracks, [features])

    @pytest.mark.skip("TODO")
    def test_one_missing_split(s):
        """Split the track at the missing feature."""

    def test_one_missing_continue(s):
        """Connect the track across the missing feature."""
        features = [[f] if f else [] for f in s.track1]
        features[3] = []
        tracks = s.run_tracking(features, event_id=0, tracker=s.tracker1)

        # Check regular features
        s.assertEqual(len(tracks), 1)
        features = [[f] for f in s.track1]
        s.assert_tracks_features(tracks, [features], ignore_ts=3)

        # Check features in mean feature (dummy event)
        sol = [features[2][0], features[4][0]]
        res = tracks[0].events_ts(3)[0].feature().features()
        s.assertEqual(sol, res)

        # Check event types
        ets = [es[0] for es in tracks[0].events_ts()]
        s.assertFalse(ets[0].is_dummy())
        s.assertTrue(ets[0].is_genesis())
        s.assertFalse(ets[1].is_dummy())
        s.assertTrue(ets[1].is_continuation())
        s.assertFalse(ets[2].is_dummy())
        s.assertTrue(ets[2].is_continuation())
        s.assertTrue(ets[3].is_dummy())
        s.assertTrue(ets[3].is_continuation())
        s.assertFalse(ets[4].is_dummy())
        s.assertTrue(ets[4].is_continuation())
        s.assertFalse(ets[5].is_dummy())
        s.assertTrue(ets[5].is_end())

    def test_two_missing_split(s):
        """Track is split because of too many missing features."""
        features = [[f] if f else [] for f in s.track1]
        features[3] = []
        features[4] = []
        tracks = s.run_tracking(features, event_id=0, tracker=s.tracker1)

        s.assertEqual(len(tracks), 2)
        features = [[[f] for f in ff] for ff in (s.track1[:3], s.track1[5:])]
        s.assert_tracks_features(tracks, features)

        s.assertEqual(s.track1[4].event(), None)


if __name__ == "__main__":
    import logging as log

    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
    log.getLogger().setLevel(log.DEBUG)
    unittest.main()
