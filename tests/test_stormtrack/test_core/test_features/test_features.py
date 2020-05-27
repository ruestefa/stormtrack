#!/usr/bin/env python3

# Standard library
import logging as log
import unittest
import sys
from unittest import TestCase

# Third-party
import numpy as np

# First-party
from stormtrack.core.constants import default_constants
from stormtrack.core.identification import Feature
from stormtrack.core.identification import features_find_neighbors
from stormtrack.core.identification import features_grow
from stormtrack.core.identification import find_minima_2d
from stormtrack.core.identification import merge_adjacent_features

# Local
from ...utils import TestFeatures_Base


# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)


# TODO properly implement and test center and extrema


class Feature_FindNeighbors(TestCase):
    """Find all neighbors in a list of Features."""

    def setUp(s):

        _ = np.nan
        # fmt: off
        fld = np.array(
            [  # 0 1 2 3 4  5 6 7 8 9
                [_,_,_,_,2, 2,_,_,_,_], # 7
                [_,_,_,2,2, 2,2,_,_,_], # 6
                [_,0,0,2,2, _,_,_,_,_], # 5

                [0,0,0,_,_, _,_,3,_,_], # 4
                [0,0,_,1,1, _,3,3,3,_], # 3
                [0,_,_,1,1, 1,_,3,3,_], # 2
                [_,_,_,_,1, 1,_,3,3,_], # 1
                [_,_,_,_,1, _,_,_,_,_], # 0
            ]  # 0 1 2 3 4  5 6 7 8 9
        ).T[:, ::-1]
        # fmt: on
        s.nx, s.ny = fld.shape
        features_pixels = [np.dstack(np.where(fld == i))[0] for i in range(4)]

        def arr(lst, dt=np.int32):
            return np.asarray(lst, dtype=dt)

        s.features = [
            Feature(
                values=np.ones(len(pixels), dtype=np.float32), pixels=arr(pixels), id_=i
            )
            for i, pixels in enumerate(features_pixels)
        ]

    def assertNeighbors(s, feature, sol):
        res = feature.neighbors
        s.assertSetEqual(set(res), set(sol), "id {}".format(feature.id))

    def test_4c(s):
        for feature in s.features:
            s.assertEqual(len(feature.neighbors), 0)
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=4)
        features_find_neighbors(s.features, const)
        s.assertNeighbors(s.features[0], [s.features[2]])
        s.assertNeighbors(s.features[1], [])
        s.assertNeighbors(s.features[2], [s.features[0]])
        s.assertNeighbors(s.features[3], [])

    def test_8c(s):
        for feature in s.features:
            s.assertEqual(len(feature.neighbors), 0)
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        features_find_neighbors(s.features, const)
        s.assertNeighbors(s.features[0], [s.features[2], s.features[1]])
        s.assertNeighbors(s.features[1], [s.features[0], s.features[3]])
        s.assertNeighbors(s.features[2], [s.features[0]])
        s.assertNeighbors(s.features[3], [s.features[1]])


class MergeFeatures(TestFeatures_Base):
    def setUp(s):

        _ = np.nan
        # fmt: off
        s.obj_ids_in_4c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
                [_,_,0,0,1, _,_,_,_,_, _,9,9,_,_, _,_,_,_,_], #  3
                [_,0,0,1,1, 2,2,_,_,_, 9,9,9,9,_, _,_,8,8,_], #  2
                [_,_,1,1,1, 2,2,2,_,_, _,9,9,_,_, _,8,8,8,_], #  1
                [_,_,1,1,2, 2,2,2,_,_, _,_,_,_,_, _,8,8,_,_], # 10

                [_,_,3,2,2, 2,2,_,_,_, _,_,_,_,_, _,8,_,_,_], #  9
                [_,_,3,3,3, 3,_,4,4,_, _,_,_,_,_, 7,_,_,_,_], #  8
                [_,_,_,3,3, _,4,4,4,5, _,_,_,_,7, 7,_,_,_,_], #  7
                [_,_,_,3,_, _,4,4,4,5, 5,_,_,_,7, 7,_,_,_,_], #  6
                [_,_,_,_,_, _,_,4,5,5, 5,5,_,_,7, 7,_,_,_,_], #  5

                [_,_,_,_,_, _,_,_,_,_, _,_,_,6,7, _,_,_,_,_], #  4
                [_,_,_,_,_, _,_,_,_,_, _,6,6,6,_, _,_,_,_,_], #  3
                [_,_,_,_,_, _,_,_,_,_, 6,6,6,6,_, _,_,_,_,_], #  2
                [_,_,_,_,_, _,_,_,_,_, 6,6,6,_,_, _,_,_,_,_], #  1
                [_,_,_,_,_, _,_,_,_,_, _,6,_,_,_, _,_,_,_,_], #  0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
        ).T[:, ::-1]
        # fmt: on
        s.obj_ids_in_8c = s.obj_ids_in_4c
        s.fld = np.where(np.isnan(s.obj_ids_in_4c), 0, 1).astype(np.float32)
        s.inds_features_in_4c = [[i] for i in range(10)]
        s.inds_features_in_8c = [[i] for i in range(10)]

        # fmt: off
        s.inds_neighbors_in_4c = [
            [1], [0, 2, 3], [1, 3], [1, 2], [5], [4], [7], [6], [], [],
        ]
        s.inds_neighbors_in_8c = [
            [1], [0, 2, 3], [1, 3, 4], [1, 2, 4], [2, 3, 5], [4], [7], [6, 8], [7], [],
        ]
        # fmt: on

        s.inds_features_out_4c = [[0, 1, 2, 3], [4, 5], [6, 7], [8], [9]]
        s.inds_features_out_8c = [[0, 1, 2, 3, 4, 5], [6, 7, 8], [9]]

        s.inds_neighbors_out_4c = [[], [], [], [], []]
        s.inds_neighbors_out_8c = [[], [], []]

        # fmt: off
        s.shells_out_4c = [
            [
                np.array(  # [0, 1, 2, 3]
                    [
                        ( 2, 13), ( 3, 13), ( 4, 13), ( 4, 12), ( 5, 12), ( 6, 12),
                        ( 6, 11), ( 7, 11), ( 7, 10), ( 6, 10), ( 6,  9), ( 5,  9),
                        ( 5,  8), ( 4,  8), ( 4,  7), ( 3,  7), ( 3,  6), ( 3,  7),
                        ( 3,  8), ( 2,  8), ( 2,  9), ( 2, 10), ( 2, 11), ( 2, 12),
                        ( 1, 12), ( 2, 12), ( 2, 13),
                    ]
                ),
            ],
            [
                np.array(  # [4, 5]
                    [
                        ( 7,  8), ( 8,  8), ( 8,  7), ( 9,  7), ( 9,  6), (10,  6),
                        (10,  5), (11,  5), (10,  5), ( 9,  5), ( 8,  5), ( 7,  5),
                        ( 7,  6), ( 6,  6), ( 6,  7), ( 7,  7), ( 7,  8),
                    ]
                ),
            ],
            [
                np.array(  # [6, 7]
                    [
                        (15,  8), (15,  7), (15,  6), (15,  5), (14,  5), (14,  4),
                        (13,  4), (13,  3), (13,  2), (12,  2), (12,  1), (11,  1),
                        (11,  0), (11,  1), (10,  1), (10,  2), (11,  2), (11,  3),
                        (12,  3), (13,  3), (13,  4), (14,  4), (14,  5), (14,  6),
                        (14,  7), (15,  7), (15,  8),
                    ]
                ),
            ],
            [
                np.array(  # [8]
                    [
                        (17, 12), (18, 12), (18, 11), (17, 11), (17, 10), (16, 10),
                        (16,  9), (16, 10), (16, 11), (17, 11), (17, 12),
                    ]
                ),
            ],
            [
                np.array(  # [9]
                    [
                        (11, 13), (12, 13), (12, 12), (13, 12), (12, 12), (12, 11),
                        (11, 11), (11, 12), (10, 12), (11, 12), (11, 13),
                    ]
                ),
            ],
        ]
        # fmt: on
        s.holes_out_4c = [[] for _ in range(5)]
        # fmt: off
        s.shells_out_8c = [
            [
                np.array(  # [0, 1, 2, 3, 4, 5]
                    [
                        ( 2, 13), ( 3, 13), ( 4, 13), ( 5, 12), ( 6, 12), ( 7, 11),
                        ( 7, 10), ( 6,  9), ( 7,  8), ( 8,  8), ( 9,  7), (10,  6),
                        (11,  5), (10,  5), ( 9,  5), ( 8,  5), ( 7,  5), ( 6,  6),
                        ( 6,  7), ( 5,  8), ( 4,  7), ( 3,  6), ( 3,  7), ( 2,  8),
                        ( 2,  9), ( 2, 10), ( 2, 11), ( 1, 12), ( 2, 13),
                    ]
                ),
            ],
            [
                np.array(  # [6, 7, 8]
                    [
                        (17, 12), (18, 12), (18, 11), (17, 10), (16,  9), (15,  8),
                        (15,  7), (15,  6), (15,  5), (14,  4), (13,  3), (13,  2),
                        (12,  1), (11,  0), (10,  1), (10,  2), (11,  3), (12,  3),
                        (13,  4), (14,  5), (14,  6), (14,  7), (15,  8), (16,  9),
                        (16, 10), (16, 11), (17, 12),
                    ]
                ),
            ],
            [
                np.array(  # [9]
                    [
                        (11, 13), (12, 13), (13, 12), (12, 11), (11, 11), (10, 12),
                        (11, 13),
                    ]
                ),
            ],
        ]
        # fmt: on
        s.holes_out_8c = [[[(6, 9), (7, 8), (6, 7), (5, 8), (6, 9)]], [], []]

        s.setUpFeatures("in_4c", "in_8c", "out_4c", "out_8c")

    def test_given_neighbors_4c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=4)
        features = merge_adjacent_features(
            s.features_in_4c,
            base_id=0,
            find_neighbors=False,
            ignore_missing_neighbors=False,
            constants=const,
        )
        s.assertFeaturesEqual(features, s.features_out_4c, check_shared_pixels=False)

    def test_given_neighbors_8c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        features = merge_adjacent_features(
            s.features_in_8c,
            base_id=0,
            find_neighbors=False,
            ignore_missing_neighbors=False,
            constants=const,
        )
        s.assertFeaturesEqual(features, s.features_out_8c, check_shared_pixels=False)

    def test_unknown_neighbors_4c(s):
        for feature in s.features_in_8c:
            feature.neighbors = []
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=4)
        features = merge_adjacent_features(
            s.features_in_4c,
            base_id=0,
            find_neighbors=True,
            ignore_missing_neighbors=False,
            constants=const,
        )
        s.assertFeaturesEqual(features, s.features_out_4c, check_shared_pixels=False)

    def test_unknown_neighbors_8c(s):
        for feature in s.features_in_8c:
            feature.neighbors = []
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        features = merge_adjacent_features(
            s.features_in_8c,
            base_id=0,
            find_neighbors=True,
            ignore_missing_neighbors=False,
            constants=const,
        )
        s.assertFeaturesEqual(features, s.features_out_8c, check_shared_pixels=False)


class GrowFeatures(TestFeatures_Base):
    def setUp(s):

        _ = np.nan
        # fmt: off
        s.obj_ids_in_4c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_], # 9
                [_,_,0,0,1, _,_,_,_,_, _,7,7,_,_], # 8
                [_,0,0,1,1, 2,2,_,_,_, 7,7,7,7,_], # 7
                [_,_,1,1,1, 2,2,2,_,_, _,7,7,_,_], # 6
                [_,_,1,1,2, 2,2,2,_,_, _,_,_,_,_], # 5

                [_,_,3,2,2, 2,2,_,_,_, _,_,_,_,_], # 4
                [_,_,3,3,3, 3,_,4,4,_, _,_,_,_,_], # 3
                [_,_,_,3,3, _,4,4,4,5, _,_,_,_,6], # 2
                [_,_,_,3,_, _,4,4,4,5, 5,_,_,_,6], # 1
                [_,_,_,_,_, _,_,4,5,5, 5,5,_,_,6], # 0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
        ).T[:, ::-1]
        # fmt: on
        s.obj_ids_in_8c = s.obj_ids_in_4c
        s.fld = np.where(np.isnan(s.obj_ids_in_4c), 0, 1).astype(np.float32)
        s.inds_features_in_4c = [[i] for i in range(8)]
        s.inds_features_in_8c = [[i] for i in range(8)]
        # fmt: off
        s.inds_neighbors_in_4c = [
            [1],        # 0
            [0, 2, 3],  # 1
            [1, 3],     # 2
            [1, 2],     # 3
            [5],        # 4
            [4],        # 5
            [],         # 6
            [],         # 7
        ]
        s.inds_neighbors_in_8c = [
            [1],        # 0
            [0, 2, 3],  # 1
            [1, 3, 4],  # 2
            [1, 2, 4],  # 3
            [2, 3, 5],  # 4
            [4],        # 5
            [],         # 6
            [],         # 7
        ]
        # fmt: on

        # Grow by 1 (4-connectivity)
        _ = np.nan
        # fmt: off
        s.obj_ids_out_g1_4c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
                [_,_,0,0,1, _,_,_,_,_, _,7,7,_,_], # 9
                [_,0,0,0,1, 2,2,_,_,_, 7,7,7,7,_], # 8
                [0,0,0,1,1, 2,2,2,_,7, 7,7,7,7,7], # 7
                [_,1,1,1,1, 2,2,2,2,_, 7,7,7,7,_], # 6
                [_,1,1,1,2, 2,2,2,2,_, _,7,7,_,_], # 5

                [_,3,3,2,2, 2,2,2,4,_, _,_,_,_,_], # 4
                [_,3,3,3,3, 3,4,4,4,4, _,_,_,_,6], # 3
                [_,_,3,3,3, 3,4,4,4,5, 5,_,_,6,6], # 2
                [_,_,3,3,3, 4,4,4,4,5, 5,5,_,6,6], # 1
                [_,_,_,3,_, _,4,4,5,5, 5,5,5,6,6], # 0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
        ).T[:, ::-1]
        # fmt: on
        s.fld = np.where(np.isnan(s.obj_ids_in_4c), 0, 1).astype(np.float32)
        s.inds_features_out_g1_4c = [[i] for i in range(8)]
        # fmt: off
        s.inds_neighbors_out_g1_4c = [
            [1],        # 0
            [0, 2, 3],  # 1
            [1, 3, 4],  # 2
            [1, 2, 4],  # 3
            [2, 3, 5],  # 4
            [4, 6],     # 5
            [5],        # 6
            [],         # 7
        ]
        # fmt: on

        # Grow by 2 (4-connectivity)
        _ = np.nan
        # fmt: off
        s.obj_ids_out_g2_4c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
                [_,0,0,0,1, 2,2,_,_,_, 7,7,7,7,_], # 9
                [0,0,0,0,1, 2,2,2,_,7, 7,7,7,7,7], # 8
                [0,0,0,1,1, 2,2,2,2,7, 7,7,7,7,7], # 7
                [1,1,1,1,1, 2,2,2,2,7, 7,7,7,7,7], # 6
                [1,1,1,1,2, 2,2,2,2,2, 7,7,7,7,_], # 5

                [3,3,3,2,2, 2,2,2,4,4, _,7,7,_,6], # 4
                [3,3,3,3,3, 3,4,4,4,4, 5,_,_,6,6], # 3
                [_,3,3,3,3, 3,4,4,4,5, 5,5,6,6,6], # 2
                [_,3,3,3,3, 4,4,4,4,5, 5,5,5,6,6], # 1
                [_,_,3,3,3, 4,4,4,5,5, 5,5,5,6,6], # 0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
        ).T[:, ::-1]
        # fmt: on
        s.fld = np.where(np.isnan(s.obj_ids_in_4c), 0, 1).astype(np.float32)
        s.inds_features_out_g2_4c = [[i] for i in range(8)]
        # fmt: off
        s.inds_neighbors_out_g2_4c = [
            [1],            # 0
            [0, 2, 3],      # 1
            [1, 3, 4, 7],   # 2
            [1, 2, 4],      # 3
            [2, 3, 5],      # 4
            [4, 6],         # 5
            [5],            # 6
            [2],            # 7
        ]
        # fmt: on

        # Grow by 3 (4-connectivity)
        _ = np.nan
        # fmt: off
        s.obj_ids_out_g3_4c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
                [0,0,0,0,1, 2,2,2,_,7, 7,7,7,7,7], # 9
                [0,0,0,0,1, 2,2,2,2,7, 7,7,7,7,7], # 8
                [0,0,0,1,1, 2,2,2,2,7, 7,7,7,7,7], # 7
                [1,1,1,1,1, 2,2,2,2,7, 7,7,7,7,7], # 6
                [1,1,1,1,2, 2,2,2,2,2, 7,7,7,7,7], # 5

                [3,3,3,2,2, 2,2,2,4,4, 7,7,7,7,6], # 4
                [3,3,3,3,3, 3,4,4,4,4, 5,5,6,6,6], # 3
                [3,3,3,3,3, 3,4,4,4,5, 5,5,6,6,6], # 2
                [3,3,3,3,3, 4,4,4,4,5, 5,5,5,6,6], # 1
                [_,3,3,3,3, 4,4,4,5,5, 5,5,5,6,6], # 0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
        ).T[:, ::-1]
        # fmt: on
        s.fld = np.where(np.isnan(s.obj_ids_in_4c), 0, 1).astype(np.float32)
        s.inds_features_out_g3_4c = [[i] for i in range(8)]
        # fmt: off
        s.inds_neighbors_out_g3_4c = [
            [1],            # 0
            [0, 2, 3],      # 1
            [1, 3, 4, 7],   # 2
            [1, 2, 4],      # 3
            [2, 3, 5, 7],   # 4
            [4, 6, 7],      # 5
            [5, 7],         # 6
            [2, 4, 5, 6],   # 7
        ]
        # fmt: on

        # Grow by 1 (8-connectivity)
        _ = np.nan
        # fmt: off
        s.obj_ids_out_g1_8c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
                [_,0,0,0,1, 1,_,_,_,_, 7,7,7,7,_], # 9
                [0,0,0,0,1, 2,2,2,_,7, 7,7,7,7,7], # 8
                [0,0,0,1,1, 2,2,2,2,7, 7,7,7,7,7], # 7
                [0,1,1,1,1, 2,2,2,2,7, 7,7,7,7,7], # 6
                [_,1,1,1,2, 2,2,2,2,_, 7,7,7,7,_], # 5

                [_,3,3,2,2, 2,2,2,4,4, _,_,_,_,_], # 4
                [_,3,3,3,3, 3,4,4,4,4, 5,_,_,6,6], # 3
                [_,3,3,3,3, 3,4,4,4,5, 5,5,_,6,6], # 2
                [_,_,3,3,3, 4,4,4,4,5, 5,5,5,6,6], # 1
                [_,_,3,3,3, 4,4,4,5,5, 5,5,5,6,6], # 0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
        ).T[:, ::-1]
        # fmt: on
        s.fld = np.where(np.isnan(s.obj_ids_in_8c), 0, 1).astype(np.float32)
        s.inds_features_out_g1_8c = [[i] for i in range(8)]
        # fmt: off
        s.inds_neighbors_out_g1_8c = [
            [1],            # 0
            [0, 2, 3],      # 1
            [1, 3, 4, 7],   # 2
            [1, 2, 4],      # 3
            [2, 3, 5, 7],   # 4
            [4, 6],         # 5
            [5],            # 6
            [2, 4],         # 7
        ]
        # fmt: on

        # Grow by 2 (8-connectivity)
        _ = np.nan
        # fmt: off
        s.obj_ids_out_g2_8c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
                [0,0,0,0,1, 1,2,2,7,7, 7,7,7,7,7], # 9
                [0,0,0,0,1, 2,2,2,2,7, 7,7,7,7,7], # 8
                [0,0,0,1,1, 2,2,2,2,7, 7,7,7,7,7], # 7
                [0,1,1,1,1, 2,2,2,2,7, 7,7,7,7,7], # 6
                [1,1,1,1,2, 2,2,2,2,7, 7,7,7,7,7], # 5

                [3,3,3,2,2, 2,2,2,4,4, 7,7,7,7,6], # 4
                [3,3,3,3,3, 3,4,4,4,4, 5,5,6,6,6], # 3
                [3,3,3,3,3, 3,4,4,4,5, 5,5,5,6,6], # 2
                [3,3,3,3,3, 4,4,4,4,5, 5,5,5,6,6], # 1
                [_,3,3,3,3, 4,4,4,5,5, 5,5,5,6,6], # 0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
        ).T[:, ::-1]
        # fmt: on
        s.fld = np.where(np.isnan(s.obj_ids_in_8c), 0, 1).astype(np.float32)
        s.inds_features_out_g2_8c = [[i] for i in range(8)]
        # fmt: off
        s.inds_neighbors_out_g2_8c = [
            [1],            # 0
            [0, 2, 3],      # 1
            [1, 3, 4, 7],   # 2
            [1, 2, 4],      # 3
            [2, 3, 5, 7],   # 4
            [4, 6, 7],      # 5
            [5, 7],         # 6
            [2, 4, 5, 6],   # 7
        ]
        # fmt: on

        # Grow by 3 (8-connectivity)
        _ = np.nan
        # fmt: off
        s.obj_ids_out_g3_8c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
                [0,0,0,0,1, 1,2,2,7,7, 7,7,7,7,7], # 9
                [0,0,0,0,1, 2,2,2,2,7, 7,7,7,7,7], # 8
                [0,0,0,1,1, 2,2,2,2,7, 7,7,7,7,7], # 7
                [0,1,1,1,1, 2,2,2,2,7, 7,7,7,7,7], # 6
                [1,1,1,1,2, 2,2,2,2,7, 7,7,7,7,7], # 5

                [3,3,3,2,2, 2,2,2,4,4, 7,7,7,7,6], # 4
                [3,3,3,3,3, 3,4,4,4,4, 5,5,6,6,6], # 3
                [3,3,3,3,3, 3,4,4,4,5, 5,5,5,6,6], # 2
                [3,3,3,3,3, 4,4,4,4,5, 5,5,5,6,6], # 1
                [3,3,3,3,3, 4,4,4,5,5, 5,5,5,6,6], # 0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4
        ).T[:, ::-1]
        # fmt: on
        s.fld = np.where(np.isnan(s.obj_ids_in_8c), 0, 1).astype(np.float32)
        s.inds_features_out_g3_8c = [[i] for i in range(8)]
        # fmt: off
        s.inds_neighbors_out_g3_8c = [
            [1],            # 0
            [0, 2, 3],      # 1
            [1, 3, 4, 7],   # 2
            [1, 2, 4],      # 3
            [2, 3, 5, 7],   # 4
            [4, 6, 7],      # 5
            [5, 7],         # 6
            [2, 4, 5, 6],   # 7
        ]
        # fmt: on

        s.setUpFeatures(
            "in_4c",
            "in_8c",
            "out_g1_4c",
            "out_g1_8c",
            "out_g2_4c",
            "out_g2_8c",
            "out_g3_4c",
            "out_g3_8c",
        )

    def test_grow1_4c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=4)
        features_grown = features_grow(1, s.features_in_4c, const)
        s.assertFeaturesEqual(
            features_grown, s.features_out_g1_4c, check_shared_pixels=False
        )

    def test_grow1_8c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        features_grown = features_grow(1, s.features_in_8c, const)
        s.assertFeaturesEqual(
            features_grown, s.features_out_g1_8c, check_shared_pixels=False
        )

    def test_grow2_4c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=4)
        features_grown = features_grow(2, s.features_in_4c, const)
        s.assertFeaturesEqual(
            features_grown, s.features_out_g2_4c, check_shared_pixels=False
        )

    def test_grow2_8c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        features_grown = features_grow(2, s.features_in_8c, const)
        s.assertFeaturesEqual(
            features_grown, s.features_out_g2_8c, check_shared_pixels=False
        )

    def test_grow3_4c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=4)
        features_grown = features_grow(3, s.features_in_4c, const)
        s.assertFeaturesEqual(
            features_grown, s.features_out_g3_4c, check_shared_pixels=False
        )

    def test_grow3_8c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        features_grown = features_grow(3, s.features_in_8c, const)
        s.assertFeaturesEqual(
            features_grown, s.features_out_g3_8c, check_shared_pixels=False
        )


class TestFindMinimaNeighborhoodSize(TestCase):
    """Test identification for different neighborhood sizes.

    The neighborhood size corresponds to the number of neighboring pixels
    which are considered in the value comparison.

    """

    def setUp(s):
        _, X = 5, 4
        # fmt: off
        s.fld = np.array(
            [  # 0 1 2 3 4  5 6 7 8 9
                [_,_,_,_,_, _,_,_,_,_], # 9
                [_,X,_,_,_, _,_,_,_,_], # 8
                [_,_,_,_,_, _,_,_,_,_], # 7
                [_,_,_,_,_, _,_,X,_,_], # 6
                [_,_,_,_,X, _,_,_,_,_], # 5

                [_,_,_,_,_, _,_,_,_,_], # 4
                [_,_,_,X,_, _,_,_,_,_], # 3
                [_,_,_,_,X, _,_,_,_,_], # 2
                [_,_,_,_,_, _,_,_,_,_], # 1
                [_,_,_,_,_, _,_,_,_,_], # 0
            ],
            dtype=np.float32,
        )
        # fmt: on

    def assertPoints(s, res, sol):
        res = set([(i, j) for i, j in res])
        sol = set([(i, j) for i, j in sol])
        s.assertSetEqual(res, sol)

    def test_n4(s):
        minima = find_minima_2d(s.fld, 4)
        s.assertEqual(len(minima), 5, "should find exactly 5 minima")
        sol = [(1, 1), (3, 7), (4, 4), (6, 3), (7, 4)]
        s.assertPoints(minima, sol)

    def test_n8(s):
        minima = find_minima_2d(s.fld, 8)
        s.assertEqual(len(minima), 3, "should find exactly 3 minima")
        sol = [(1, 1), (3, 7), (4, 4)]
        s.assertPoints(minima, sol)

    def test_n12(s):
        minima = find_minima_2d(s.fld, 12)
        s.assertEqual(len(minima), 2, "should find exactly 2 minima")
        sol = [(3, 7), (4, 4)]
        s.assertPoints(minima, sol)

    def test_n20(s):
        minima = find_minima_2d(s.fld, 20)
        s.assertEqual(len(minima), 1, "should find exactly 1 minimum")
        sol = [(3, 7)]
        s.assertPoints(minima, sol)


if __name__ == "__main__":
    unittest.main()
