#!/usr/bin/env

import unittest
import sys
from unittest import TestCase

import numpy as np

from stormtrack.core.typedefs import Constants

from stormtrack.core.identification import find_features_2d_threshold
from stormtrack.core.identification import find_features_2d_threshold_seeded

#==============================================================================

class IdentifyRegions_Base(TestCase):

    @classmethod
    def setUpClass(c):
        _ = 0
        c.fld_raw = np.swapaxes(np.array([
                [   # 0  1  2  3  4  5  6  7  8  9
                    [ _, 1, 2, 1, _, _, _, _, _, _], # 0
                    [ 1, 2, 3, 2, 1, _, _, _, _, _], # 1
                    [ _, 1, 2, 1, _, _, _, _, _, _], # 2
                    [ _, _, 1, _, _, _, _, _, _, _], # 3
                    [ _, _, _, _, _, _, 1, 1, 1, _], # 4
                    [ _, _, _, _, _, 1, 2, 2, 2, 1], # 5
                    [ _, _, _, _, _, 1, 2, 3, 2, 1], # 6
                    [ _, _, _, _, _, 1, 2, 2, 2, 1], # 7
                    [ _, _, _, _, _, _, 1, 1, 1, _], # 8
                    [ _, _, _, _, _, _, _, _, _, _]  # 9
                    # 0  1  2  3  4  5  6  7  8  9
                ],
                [   # 0  1  2  3  4  5  6  7  8  9
                    [ _, 1, 1, 1, _, _, _, _, _, _], # 0
                    [ 1, 2, 2, 1, _, _, _, _, _, _], # 1
                    [ 1, 2, 3, 2, _, _, 1, _, _, _], # 2
                    [ _, 2, 4, 3, 1, 1, 1, _, _, _], # 3
                    [ _, 1, 2, 2, 2, 1, 1, 1, _, _], # 4
                    [ _, _, 1, 1, 2, 2, 2, 2, 1, _], # 5
                    [ _, _, _, _, 1, 2, 3, 2, 1, _], # 6
                    [ _, _, _, _, _, 1, 2, 2, 1, _], # 7
                    [ _, _, _, _, _, _, 1, 1, _, _], # 8
                    [ _, _, _, _, _, _, _, _, _, _]  # 9
                    # 0  1  2  3  4  5  6  7  8  9
                ],
                [   # 0  1  2  3  4  5  6  7  8  9
                    [ _, _, _, _, _, _, _, _, _, _], # 0
                    [ _, 1, _, _, _, _, _, _, _, _], # 1
                    [ _, 1, 1, _, _, _, _, _, _, _], # 2
                    [ _, 1, 1, 1, _, _, _, _, _, _], # 3
                    [ _, _, _, 1, _, _, 1, 1, _, _], # 4
                    [ _, _, _, _, _, 1, 2, 1, _, _], # 5
                    [ _, _, _, _, _, 1, 2, 2, 1, _], # 6
                    [ _, 1, 1, _, _, _, 1, 1, 1, _], # 7
                    [ 1, 2, 2, 1, _, _, _, _, _, _], # 8
                    [ _, 1, 1, _, _, _, _, _, _, _]  # 9
                    # 0  1  2  3  4  5  6  7  8  9
                ]
            ], dtype=np.float32), 0, 2)

        c.regions_xy_2d = [[
                {(2, 0), (1, 1), (2, 1), (3, 1), (2, 2)},
                {(6, 5), (7, 5), (8, 5), (6, 6), (7, 6), (8, 6),
                    (6, 7), (7, 7), (8, 7)}
            ],[
                {(6, 7), (7, 7), (5, 6), (6, 6), (7, 6), (4, 5), (5, 5),
                    (6, 5), (7, 5), (2, 4), (3, 4), (4, 4), (1, 3), (2, 3),
                    (3, 3), (1, 2), (2, 2), (3, 2), (1, 1), (2, 1)}
            ],[
                {(1, 8), (2, 8)},
                {(6, 5), (6, 6), (7, 6)}
            ]]

        c.threshold = 1.5

    def assert_regions(s, xys_res, xys_sol):
        xys_res = sorted([sorted(i) for i in xys_res])
        xys_sol = sorted([sorted(i) for i in xys_sol])
        for xy_res, xy_sol in zip(xys_res, xys_sol):
            s.assertEqual(xy_res, xy_sol, "pixels don't match")

#------------------------------------------------------------------------------

class IdentifyRegions2D_Base(IdentifyRegions_Base):

    @classmethod
    def setUpClass(c):
        super().setUpClass()
        c.regions_xy = c.regions_xy_2d

class IdentifyRegions2D(IdentifyRegions2D_Base):

    @classmethod
    def setUpClass(c):
        super().setUpClass()

    def test_1_8c(s):
        nx, ny = s.fld_raw.shape[:2]
        const = Constants.default(nx=nx, ny=ny,connectivity=8)
        regions = find_features_2d_threshold(
                s.fld_raw[:, :, 0],
                lower = s.threshold,
                constants = const,
            )
        s.assertEqual(len(regions), 2, "should find exactly two regions")
        regions_px = [[(i, j) for i, j in r.pixels] for r in regions]
        s.assert_regions(regions_px, s.regions_xy[0])

    def test_2_8c(s):
        nx, ny = s.fld_raw.shape[:2]
        const = Constants.default(nx=nx, ny=ny, connectivity=8)
        regions = find_features_2d_threshold(
                s.fld_raw[:, :, 1],
                lower = s.threshold,
                constants = const,
            )
        s.assertEqual(len(regions), 1, "should find only one regions")
        regions_px = [[(i, j) for i, j in r.pixels] for r in regions]
        s.assert_regions(regions_px, s.regions_xy[1])

    def test_3_8c(s):
        nx, ny = s.fld_raw.shape[:2]
        const = Constants.default(nx=nx, ny=ny, connectivity=8)
        regions = find_features_2d_threshold(
                s.fld_raw[:, :, 2],
                lower = s.threshold,
                constants = const,
            )
        s.assertEqual(len(regions), 2, "should find exactly two regions")
        regions_px = [[(i, j) for i, j in r.pixels] for r in regions]
        s.assert_regions(regions_px, s.regions_xy[2])

class IdentifyRegions2DSeeded(IdentifyRegions2D_Base):
    """Seed-based identification algorithm"""

    @classmethod
    def setUpClass(c):
        super().setUpClass()

    def run_test(s, seed):
        features = find_features_2d_threshold_seeded(
                s.fld_raw[:, :, 0], lower=s.threshold, random_seed=seed)
        features_px = [sorted([(i, j) for i, j in f.pixels]) for f in features]
        s.assertEqual(len(features_px), 2, "should find exactly two features")
        s.assert_regions(features_px, s.regions_xy[0])

        features = find_features_2d_threshold_seeded(
                s.fld_raw[:, :, 1], lower=s.threshold, random_seed=seed)
        features_px = [sorted([(i, j) for i, j in f.pixels]) for f in features]
        s.assertEqual(len(features_px), 1, "should find only one features")
        s.assert_regions(features_px, s.regions_xy[1])

        features = find_features_2d_threshold_seeded(
                s.fld_raw[:, :, 2], lower=s.threshold, random_seed=seed)
        features_px = [sorted([(i, j) for i, j in f.pixels]) for f in features]
        s.assertEqual(len(features_px), 2, "should find exactly two features")
        s.assert_regions(features_px, s.regions_xy[2])

    def test_seed0(s):
        s.run_test(0)

    def test_seed1(s):
        s.run_test(1)

    def test_seed2(s):
        s.run_test(2)

class IdentifyRegions2DSeeds(IdentifyRegions2D_Base):

    @classmethod
    def setUpClass(c):
        super().setUpClass()

        c.regions_xy = c.regions_xy_2d

        c.seed_points_xy = np.array([
                [(3, 1), (7, 6)],
                [(2, 3), (6, 6)]
            ], dtype=np.int32)

    def test_lvl0_region1(s):
        seeds = s.seed_points_xy[0, 0, None]
        features = find_features_2d_threshold_seeded(
                s.fld_raw[:, :, 0], lower=s.threshold, seeds=seeds)
        features_px = [sorted([(i, j) for i, j in f.pixels]) for f in features]
        s.assertEqual(len(features_px), 1, "should only find one region")
        s.assert_regions(features_px, [s.regions_xy[0][0]])

    def test_lvl0_region2(s):
        seeds = s.seed_points_xy[0, 1, None]
        features = find_features_2d_threshold_seeded(
                s.fld_raw[:, :, 0], lower=s.threshold, seeds=seeds)
        features_px = [sorted([(i, j) for i, j in f.pixels]) for f in features]
        s.assertEqual(len(features_px), 1, "should only find one region")
        s.assert_regions(features_px, [s.regions_xy[0][1]])

    def test_lvl0(s):
        seeds = s.seed_points_xy[0, :]
        features = find_features_2d_threshold_seeded(
                s.fld_raw[:, :, 0], lower=s.threshold, seeds=seeds)
        features_px = [sorted([(i, j) for i, j in f.pixels]) for f in features]
        s.assertEqual(len(features_px), 2, "should find exactly two regions")
        s.assert_regions(features_px, s.regions_xy[0])

    @unittest.skip("TODO")
    def test_lvl1(s):
        """Multiple seed points in one region."""
        seeds = s.seed_points_xy[1, :]
        features, seeds_px = find_features_2d_threshold_seeded(
                    s.fld_raw[:, :, 1], lower=s.threshold, seeds=seeds)
        features_px = [sorted([(i, j) for i, j in f.pixels]) for f in features]
        s.assertEqual(len(features_px), 1, "should find only 1 region")
        s.assertEqual(len(seeds_px), 1, "should find seeds for only 1 region")
        s.assertEqual(len(seeds_px[0]), 2, "should find 2 seeds for region")
        sol = set([(i, j) for i, j in s.seed_points_xy[1, :]])
        s.assertSetEqual(set(seeds_px[0]), sol)
        s.assert_regions(features_px, s.regions_xy[1])

    @unittest.skip("TODO")
    def test_lvl2(s):
        pass

#==============================================================================

if __name__ == "__main__":

    import logging as log
    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
    log.getLogger().setLevel(log.DEBUG)

    unittest.main()
