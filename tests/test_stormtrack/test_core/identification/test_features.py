#!/usr/bin/env python3

# Standard library
import unittest
import sys
from unittest import TestCase

# Third-party
import numpy as np

# First-party
from stormtrack.core.identification import Feature
from stormtrack.core.identification import feature_split_regiongrow
from stormtrack.core.identification import features_find_neighbors
from stormtrack.core.identification import features_grow
from stormtrack.core.identification import find_minima_2d
from stormtrack.core.identification import merge_adjacent_features
from stormtrack.core.identification import pixels_find_boundaries
from stormtrack.core.identification import split_regiongrow_levels
from stormtrack.core.typedefs import default_constants

# Local
from ...utils import assertBoundaries
from ...utils import TestFeatures_Base


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


def plot_field(outfile, fld, lvl=None, cmap=None):

    import matplotlib as mpl

    if mpl.get_backend != "Agg":
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # p = ax.contourf(fld, levels=lvl, cmap=cmap)
    p = ax.imshow(fld)
    fig.colorbar(p)
    # plt.show()
    fig.savefig(outfile)
    plt.close()


class TestFeaturesSeeds_Base(TestFeatures_Base):
    def setUpFeatures(s, *names):
        super().setUpFeatures(*names)

        # Initialize seeds
        if hasattr(s, "inds_seeds"):
            imax = lambda arr: int(np.nanmax(arr))
            # SR_TODO cleanup wrt connectivity
            obj_ids = getattr(s, "obj_ids_{}".format("in_8c"))
            pixels_ids_raw = [
                s.select_pixels(obj_ids, [i]) for i in range(imax(obj_ids) + 1)
            ]
            s.seeds = [
                s.create_feature(
                    pixels=pixels_ids_raw[ind],
                    id_=s.fid + i,
                    shells=[[[]]],
                    holes=[[[]]],
                )
                for i, ind in enumerate(s.inds_seeds)
            ]
            s.fid += len(s.seeds)


_ = 0
# fmt: off
fld_raw = np.array(
    [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  9
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  8
        [_,_,_,_,_, _,_,_,_,_, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, _,_,_,_,_], #  7
        [_,_,_,_,_, _,_,_,_,_, 1,2,2,2,2, 2,2,2,2,2, 2,1,1,1,1, 1,1,1,1,1, 1,_,_,_,_], #  6
        [_,_,_,_,_, _,_,_,_,_, 1,2,2,3,3, 3,3,3,3,3, 2,2,1,1,1, 1,1,1,1,1, 1,1,_,_,_], # 15

        [_,_,_,_,_, _,_,1,1,1, 1,2,2,2,2, 2,2,2,2,2, 2,2,1,1,1, 1,1,1,1,1, 1,1,_,_,_], #  4
        [_,_,_,_,_, 1,1,1,2,2, 2,2,2,2,2, 2,1,1,1,1, 2,2,1,1,1, 1,1,1,2,1, 1,1,1,_,_], #  3
        [_,_,_,1,1, 1,2,2,2,2, 2,2,2,2,2, 1,1,1,1,1, 1,2,1,1,1, 1,1,1,2,1, 1,1,1,_,_], #  2
        [_,_,1,1,1, 2,2,2,3,3, 3,3,3,2,2, 1,1,_,_,_, 1,1,1,1,1, 1,1,1,2,2, 1,1,1,_,_], #  1
        [_,1,1,1,2, 2,2,2,2,2, 2,2,2,2,2, 1,1,_,_,_, _,_,1,1,1, 1,1,1,2,2, 1,1,1,_,_], # 10

        [_,1,1,1,2, 2,2,2,2,2, 2,2,2,2,2, 2,1,1,_,_, _,_,_,1,1, 1,1,1,1,1, 1,1,1,_,_], #  9
        [_,1,1,2,2, 2,3,3,3,3, 3,3,3,2,2, 2,1,1,_,_, _,_,_,_,_, 1,1,1,1,1, 1,1,_,_,_], #  8
        [_,1,1,2,2, 2,3,3,3,3, 3,3,3,2,2, 2,1,1,_,_, _,_,_,_,_, _,_,1,1,1, 1,_,_,_,_], #  7
        [_,1,1,1,2, 2,2,2,2,2, 2,2,2,2,2, 1,1,1,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  6
        [_,1,1,1,1, 2,2,2,2,2, 2,2,2,2,2, 1,1,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  5

        [_,_,1,1,1, 1,2,2,2,2, 2,2,2,2,1, 1,1,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
        [_,_,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
        [_,_,_,1,1, 1,1,1,1,1, 1,1,1,1,1, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  2
        [_,_,_,_,_, _,_,1,1,1, 1,1,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  1
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
    ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
).T[:, ::-1]
# fmt: on

_ = np.nan
# fmt: off
obj_ids_ge1_from_ge3 = np.array(
    [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  9
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  8
        [_,_,_,_,_, _,_,_,_,_, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,5,5, _,_,_,_,_], #  7
        [_,_,_,_,_, _,_,_,_,_, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,5,5, 5,_,_,_,_], #  6
        [_,_,_,_,_, _,_,_,_,_, 5,5,5,2,2, 2,2,2,2,2, 5,5,5,5,5, 5,5,5,5,5, 5,5,_,_,_], # 15

        [_,_,_,_,_, _,_,4,4,4, 4,5,5,5,5, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,5,5, 5,5,_,_,_], #  4
        [_,_,_,_,_, 4,4,4,4,4, 4,4,4,5,5, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,_,_], #  3
        [_,_,_,3,3, 4,4,4,4,4, 4,4,4,4,4, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,_,_], #  2
        [_,_,3,3,3, 3,4,4,1,1, 1,1,1,4,4, 4,5,_,_,_, 5,5,5,5,5, 5,5,5,5,5, 5,5,5,_,_], #  1
        [_,3,3,3,3, 3,3,4,4,4, 4,4,4,4,4, 4,4,_,_,_, _,_,5,5,5, 5,5,5,5,5, 5,5,5,_,_], # 10

        [_,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,_,_, _,_,_,5,5, 5,5,5,5,5, 5,5,5,_,_], #  9
        [_,3,3,3,3, 3,0,0,0,0, 0,0,0,3,3, 3,3,3,_,_, _,_,_,_,_, 5,5,5,5,5, 5,5,_,_,_], #  8
        [_,3,3,3,3, 3,0,0,0,0, 0,0,0,3,3, 3,3,3,_,_, _,_,_,_,_, _,_,5,5,5, 5,_,_,_,_], #  7
        [_,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  6
        [_,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  5

        [_,_,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
        [_,_,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
        [_,_,_,3,3, 3,3,3,3,3, 3,3,3,3,3, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  2
        [_,_,_,_,_, _,_,3,3,3, 3,3,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  1
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
    ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
).T[:, ::-1]
# fmt: on

_, X = np.nan, 10
# fmt: off
obj_ids_split_levels_23_4c = np.array(
    [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  9
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  8
        [_,_,_,_,_, _,_,_,_,_, 6,6,6,6,6, 6,6,6,6,6, 6,6,6,6,6, 6,9,9,9,9, _,_,_,_,_], #  7
        [_,_,_,_,_, _,_,_,_,_, 6,7,7,7,7, 7,7,7,7,7, 7,6,6,6,6, 6,9,9,9,9, 9,_,_,_,_], #  6
        [_,_,_,_,_, _,_,_,_,_, 6,7,7,8,8, 8,8,8,8,8, 7,7,6,6,6, 6,9,9,9,9, 9,9,_,_,_], # 15

        [_,_,_,_,_, _,_,3,3,3, 3,7,7,7,7, 7,7,7,7,7, 7,7,6,6,6, 9,9,9,9,9, 9,9,_,_,_], #  4
        [_,_,_,_,_, 3,3,3,3,3, 3,3,3,7,7, 7,6,6,6,6, 7,7,6,6,6, 9,9,9,X,9, 9,9,9,_,_], #  3
        [_,_,_,3,3, 3,3,3,3,3, 3,3,3,3,3, 6,6,6,6,6, 6,7,6,6,6, 9,9,9,X,9, 9,9,9,_,_], #  2
        [_,_,3,3,3, 3,3,3,5,5, 5,5,5,3,3, 3,6,_,_,_, 6,6,6,6,6, 9,9,9,X,X, 9,9,9,_,_], #  1
        [_,0,0,0,1, 1,1,3,3,3, 3,3,3,3,3, 3,3,_,_,_, _,_,6,6,9, 9,9,9,X,X, 9,9,9,_,_], # 10

        [_,0,0,0,1, 1,1,1,1,1, 1,1,1,1,1, 1,0,0,_,_, _,_,_,6,9, 9,9,9,9,9, 9,9,9,_,_], #  9
        [_,0,0,1,1, 1,2,2,2,2, 2,2,2,1,1, 1,0,0,_,_, _,_,_,_,_, 9,9,9,9,9, 9,9,_,_,_], #  8
        [_,0,0,1,1, 1,2,2,2,2, 2,2,2,1,1, 1,0,0,_,_, _,_,_,_,_, _,_,9,9,9, 9,_,_,_,_], #  7
        [_,0,0,0,1, 1,1,1,1,1, 1,1,1,1,1, 0,0,0,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  6
        [_,0,0,0,0, 1,1,1,1,1, 1,1,1,1,1, 0,0,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  5

        [_,_,0,0,0, 0,1,1,1,1, 1,1,1,1,0, 0,0,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
        [_,_,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
        [_,_,_,0,0, 0,0,0,0,0, 0,0,0,0,0, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  2
        [_,_,_,_,_, _,_,0,0,0, 0,0,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  1
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
    ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
).T[:, ::-1]
# fmt: on

_, X = np.nan, 10
# fmt: off
obj_ids_split_levels_23_8c = np.array(
    [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  9
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  8
        [_,_,_,_,_, _,_,_,_,_, 6,6,6,6,6, 6,6,6,6,6, 6,6,6,6,6, 6,9,9,9,9, _,_,_,_,_], #  7
        [_,_,_,_,_, _,_,_,_,_, 6,7,7,7,7, 7,7,7,7,7, 7,6,6,6,6, 9,9,9,9,9, 9,_,_,_,_], #  6
        [_,_,_,_,_, _,_,_,_,_, 6,7,7,8,8, 8,8,8,8,8, 7,7,6,6,6, 9,9,9,9,9, 9,9,_,_,_], # 15

        [_,_,_,_,_, _,_,3,3,3, 3,7,7,7,7, 7,7,7,7,7, 7,7,6,6,6, 9,9,9,9,9, 9,9,_,_,_], #  4
        [_,_,_,_,_, 3,3,3,3,3, 3,3,3,7,7, 7,6,6,6,6, 7,7,6,6,6, 9,9,9,X,9, 9,9,9,_,_], #  3
        [_,_,_,0,0, 3,3,3,3,3, 3,3,3,3,3, 6,6,6,6,6, 6,7,6,6,6, 9,9,9,X,9, 9,9,9,_,_], #  2
        [_,_,0,0,0, 1,3,3,5,5, 5,5,5,3,3, 3,6,_,_,_, 6,6,6,6,6, 9,9,9,X,X, 9,9,9,_,_], #  1
        [_,0,0,0,1, 1,1,3,3,3, 3,3,3,3,3, 3,3,_,_,_, _,_,6,6,6, 9,9,9,X,X, 9,9,9,_,_], # 10

        [_,0,0,0,1, 1,1,1,1,1, 1,1,1,1,1, 1,0,0,_,_, _,_,_,6,6, 9,9,9,9,9, 9,9,9,_,_], #  9
        [_,0,0,1,1, 1,2,2,2,2, 2,2,2,1,1, 1,0,0,_,_, _,_,_,_,_, 9,9,9,9,9, 9,9,_,_,_], #  8
        [_,0,0,1,1, 1,2,2,2,2, 2,2,2,1,1, 1,0,0,_,_, _,_,_,_,_, _,_,9,9,9, 9,_,_,_,_], #  7
        [_,0,0,0,1, 1,1,1,1,1, 1,1,1,1,1, 0,0,0,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  6
        [_,0,0,0,0, 1,1,1,1,1, 1,1,1,1,1, 0,0,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  5

        [_,_,0,0,0, 0,1,1,1,1, 1,1,1,1,0, 0,0,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
        [_,_,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
        [_,_,_,0,0, 0,0,0,0,0, 0,0,0,0,0, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  2
        [_,_,_,_,_, _,_,0,0,0, 0,0,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  1
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
    ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
).T[:, ::-1]
# fmt: on

_ = np.nan
# fmt: off
obj_ids_ge1_from_ge3_minsize6 = np.array(
    [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  9
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  8
        [_,_,_,_,_, _,_,_,_,_, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, _,_,_,_,_], #  7
        [_,_,_,_,_, _,_,_,_,_, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,_,_,_,_], #  6
        [_,_,_,_,_, _,_,_,_,_, 3,3,3,1,1, 1,1,1,1,1, 3,3,3,3,3, 3,3,3,3,3, 3,3,_,_,_], # 15

        [_,_,_,_,_, _,_,2,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,_,_,_], #  4
        [_,_,_,_,_, 2,2,2,2,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,_,_], #  3
        [_,_,_,2,2, 2,2,2,2,2, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,_,_], #  2
        [_,_,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,3,_,_,_, 3,3,3,3,3, 3,3,3,3,3, 3,3,3,_,_], #  1
        [_,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,2,_,_,_, _,_,3,3,3, 3,3,3,3,3, 3,3,3,_,_], # 10

        [_,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,2,2,_,_, _,_,_,3,3, 3,3,3,3,3, 3,3,3,_,_], #  9
        [_,2,2,2,2, 2,0,0,0,0, 0,0,0,2,2, 2,2,2,_,_, _,_,_,_,_, 3,3,3,3,3, 3,3,_,_,_], #  8
        [_,2,2,2,2, 2,0,0,0,0, 0,0,0,2,2, 2,2,2,_,_, _,_,_,_,_, _,_,3,3,3, 3,_,_,_,_], #  7
        [_,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,2,2,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  6
        [_,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,2,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  5

        [_,_,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,2,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
        [_,_,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
        [_,_,_,2,2, 2,2,2,2,2, 2,2,2,2,2, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  2
        [_,_,_,_,_, _,_,2,2,2, 2,2,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  1
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
    ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
).T[:, ::-1]
# fmt: on


class SplitRegiongrow_Basic_GE1FromGE3(TestFeaturesSeeds_Base):

    make_plot = False

    def setUp(s):
        s.fld = fld_raw
        s.obj_ids_in_8c = obj_ids_ge1_from_ge3
        s.inds_features_in_8c = [list(range(6))]
        s.inds_features_out_8c = [[0, 3], [1, 4], [2, 5]]
        s.inds_neighbors_out_8c = [[1], [0, 2], [1]]
        s.inds_seeds = list(range(3))
        s.setUpFeatures("in_8c", "out_8c")

    def test_8c(s):
        """Split the ge-1 feature using the ge-3 features as seeds."""
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        subfeatures = feature_split_regiongrow(
            s.features_in_8c[0], s.seeds, constants=const
        )
        s.assertFeaturesEqual(
            subfeatures,
            s.features_out_8c,
            check_boundaries=False,
            check_shared_pixels=False,
        )


_ = np.nan
# fmt: off
obj_ids_ge1_from_ge2_raw = np.array(
    [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  9
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  8
        [_,_,_,_,_, _,_,_,_,_, 2,2,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,3,3,3,3, _,_,_,_,_], #  7
        [_,_,_,_,_, _,_,_,_,_, 2,0,0,0,0, 0,0,0,0,0, 0,2,2,2,2, 3,3,3,3,3, 3,_,_,_,_], #  6
        [_,_,_,_,_, _,_,_,_,_, 2,0,0,0,0, 0,0,0,0,0, 0,0,2,2,2, 3,3,3,3,3, 3,3,_,_,_], # 15

        [_,_,_,_,_, _,_,2,2,2, 2,0,0,0,0, 0,0,0,0,0, 0,0,2,2,2, 3,3,3,3,3, 3,3,_,_,_], #  4
        [_,_,_,_,_, 2,2,2,0,0, 0,0,0,0,0, 0,2,2,2,2, 0,0,2,2,2, 3,3,3,1,3, 3,3,3,_,_], #  3
        [_,_,_,2,2, 2,0,0,0,0, 0,0,0,0,0, 2,2,2,2,2, 2,2,2,2,2, 3,3,3,1,3, 3,3,3,_,_], #  2
        [_,_,2,2,2, 0,0,0,0,0, 0,0,0,0,0, 2,2,_,_,_, 2,2,2,2,2, 3,3,3,1,1, 3,3,3,_,_], #  1
        [_,2,2,2,0, 0,0,0,0,0, 0,0,0,0,0, 2,2,_,_,_, _,_,2,2,2, 3,3,3,1,1, 3,3,3,_,_], # 10

        [_,2,2,2,0, 0,0,0,0,0, 0,0,0,0,0, 0,2,2,_,_, _,_,_,2,3, 3,3,3,3,3, 3,3,3,_,_], #  9
        [_,2,2,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,2,2,_,_, _,_,_,_,_, 3,3,3,3,3, 3,3,_,_,_], #  8
        [_,2,2,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,2,2,_,_, _,_,_,_,_, _,_,3,3,3, 3,_,_,_,_], #  7
        [_,2,2,2,0, 0,0,0,0,0, 0,0,0,0,0, 2,2,2,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  6
        [_,2,2,2,2, 0,0,0,0,0, 0,0,0,0,0, 2,2,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  5

        [_,_,2,2,2, 2,0,0,0,0, 0,0,0,0,2, 2,2,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
        [_,_,2,2,2, 2,2,2,2,2, 2,2,2,2,2, 2,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
        [_,_,_,2,2, 2,2,2,2,2, 2,2,2,2,2, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  2
        [_,_,_,_,_, _,_,2,2,2, 2,2,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  1
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
    ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
).T[:, ::-1]
# fmt: on


class SplitRegiongrow_Basic_GE1FromGE2Raw(TestFeaturesSeeds_Base):
    """Split the GE1 region using raw GE2 seeds (no pre-splitting)."""

    make_plot = False

    def setUp(s):
        s.fld = fld_raw
        s.obj_ids_in_8c = obj_ids_ge1_from_ge2_raw
        s.inds_features_in_8c = [list(range(4))]
        s.inds_seeds = [0, 1]
        s.inds_features_out_8c = [[0, 2], [1, 3]]
        s.inds_neighbors_out_8c = [[1], [0]]
        s.setUpFeatures("in_8c", "out_8c")

    def test_8c(s):
        """Split the ge-1 feature using the ge-2 features as seeds."""
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        subfeatures = feature_split_regiongrow(
            s.features_in_8c[0], s.seeds, constants=const
        )
        s.assertFeaturesEqual(
            subfeatures,
            s.features_out_8c,
            check_boundaries=False,
            check_shared_pixels=False,
        )


_ = np.nan
# fmt: off
obj_ids_ge1_from_ge2_from_ge3 = np.array(
    [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  9
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  8
        [_,_,_,_,_, _,_,_,_,_, 6,6,6,6,6, 6,6,6,6,6, 6,6,6,6,6, 6,7,7,7,7, _,_,_,_,_], #  7
        [_,_,_,_,_, _,_,_,_,_, 6,2,2,2,2, 2,2,2,2,2, 2,6,6,6,6, 7,7,7,7,7, 7,_,_,_,_], #  6
        [_,_,_,_,_, _,_,_,_,_, 6,2,2,2,2, 2,2,2,2,2, 2,2,6,6,6, 7,7,7,7,7, 7,7,_,_,_], # 15

        [_,_,_,_,_, _,_,5,5,5, 5,2,2,2,2, 2,2,2,2,2, 2,2,6,6,6, 7,7,7,7,7, 7,7,_,_,_], #  4
        [_,_,_,_,_, 5,5,5,1,1, 1,1,1,2,2, 2,6,6,6,6, 2,2,6,6,6, 7,7,7,3,7, 7,7,7,_,_], #  3
        [_,_,_,4,4, 5,1,1,1,1, 1,1,1,1,1, 6,6,6,6,6, 6,2,6,6,6, 7,7,7,3,7, 7,7,7,_,_], #  2
        [_,_,4,4,4, 0,1,1,1,1, 1,1,1,1,1, 5,4,_,_,_, 6,6,6,6,6, 7,7,7,3,3, 7,7,7,_,_], #  1
        [_,4,4,4,0, 0,0,1,1,1, 1,1,1,1,1, 4,4,_,_,_, _,_,6,6,6, 7,7,7,3,3, 7,7,7,_,_], # 10

        [_,4,4,4,0, 0,0,0,0,0, 0,0,0,0,0, 0,4,4,_,_, _,_,_,6,6, 7,7,7,7,7, 7,7,7,_,_], #  9
        [_,4,4,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,4,4,_,_, _,_,_,_,_, 7,7,7,7,7, 7,7,_,_,_], #  8
        [_,4,4,0,0, 0,0,0,0,0, 0,0,0,0,0, 0,4,4,_,_, _,_,_,_,_, _,_,7,7,7, 7,_,_,_,_], #  7
        [_,4,4,4,0, 0,0,0,0,0, 0,0,0,0,0, 4,4,4,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  6
        [_,4,4,4,4, 0,0,0,0,0, 0,0,0,0,0, 4,4,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  5

        [_,_,4,4,4, 4,0,0,0,0, 0,0,0,0,4, 4,4,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
        [_,_,4,4,4, 4,4,4,4,4, 4,4,4,4,4, 4,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
        [_,_,_,4,4, 4,4,4,4,4, 4,4,4,4,4, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  2
        [_,_,_,_,_, _,_,4,4,4, 4,4,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  1
        [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
    ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9 20 1 2 3 4 25 6 7 8 9 30 1 2 3 4
).T[:, ::-1]
# fmt: on


class SplitRegiongrow_Basic_GE1FromGE2FromGE3(TestFeaturesSeeds_Base):
    """Split the GE1 region using GE2 seeds, pre-split from (3)."""

    make_plot = False

    def setUp(s):
        s.fld = fld_raw
        s.obj_ids_in_8c = obj_ids_ge1_from_ge2_from_ge3
        s.inds_features_in_8c = [list(range(8))]
        s.inds_seeds = list(range(4))
        s.inds_features_out_8c = [[0, 4], [1, 5], [2, 6], [3, 7]]
        s.inds_neighbors_out_8c = [[1, 2], [0, 2], [0, 1, 3], [2]]
        s.setUpFeatures("in_8c", "out_8c")

    def test_8c(s):
        """Split the ge-1 feature using the ge-2 features as seeds."""
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        subfeatures = feature_split_regiongrow(
            s.features_in_8c[0], s.seeds, constants=const
        )
        s.assertFeaturesEqual(
            subfeatures,
            s.features_out_8c,
            check_boundaries=False,
            check_shared_pixels=False,
        )


class SplitRegiongrow_Levels23(TestFeatures_Base):
    """Split the GE1 region iteratively with levels [2, 3]."""

    def setUp(s):

        s.levels = [2, 3]
        s.fld = fld_raw

        s.obj_ids_in_4c = obj_ids_split_levels_23_4c
        s.inds_features_in_4c = [list(range(11))]
        s.inds_features_out_4c = [[0, 1, 2], [3, 5], [6, 7, 8], [9, 10]]
        s.inds_neighbors_out_4c = [[1], [0, 2], [1, 3], [2]]

        s.obj_ids_in_8c = obj_ids_split_levels_23_8c
        s.inds_features_in_8c = [list(range(11))]
        s.inds_features_out_8c = [[0, 1, 2], [3, 5], [6, 7, 8], [9, 10]]
        s.inds_neighbors_out_8c = [[1], [0, 2], [1, 3], [2]]

        # fmt: off
        s.shells_out_4c = [
            [
                np.array(  # [0, 1, 2]
                    [
                        ( 1, 10), ( 2, 10), ( 3, 10), ( 4, 10), ( 5, 10), ( 6, 10),
                        ( 6,  9), ( 7,  9), ( 8,  9), ( 9,  9), (10,  9), (11,  9),
                        (12,  9), (13,  9), (14,  9), (15,  9), (16,  9), (17,  9),
                        (17,  8), (17,  7), (17,  6), (16,  6), (16,  5), (16,  4),
                        (15,  4), (15,  3), (14,  3), (14,  2), (13,  2), (12,  2),
                        (11,  2), (11,  1), (10,  1), ( 9,  1), ( 8,  1), ( 7,  1),
                        ( 7,  2), ( 6,  2), ( 5,  2), ( 4,  2), ( 3,  2), ( 3,  3),
                        ( 2,  3), ( 2,  4), ( 2,  5), ( 1,  5), ( 1,  6), ( 1,  7),
                        ( 1,  8), ( 1,  9), ( 1, 10),
                    ]
                ),
            ],
            [
                np.array(  # [3, 5]
                    [
                        ( 7, 14), ( 8, 14), ( 9, 14), (10, 14), (10, 13), (11, 13),
                        (12, 13), (12, 12), (13, 12), (14, 12), (14, 11), (15, 11),
                        (15, 10), (16, 10), (15, 10), (14, 10), (13, 10), (12, 10),
                        (11, 10), (10, 10), ( 9, 10), ( 8, 10), ( 7, 10), ( 7, 11),
                        ( 6, 11), ( 5, 11), ( 4, 11), ( 3, 11), ( 2, 11), ( 3, 11),
                        ( 3, 12), ( 4, 12), ( 5, 12), ( 5, 13), ( 6, 13), ( 7, 13),
                        ( 7, 14),
                    ]
                ),
            ],
            [
                np.array(  # [6, 7, 8]
                    [
                        (10, 17), (11, 17), (12, 17), (13, 17), (14, 17), (15, 17),
                        (16, 17), (17, 17), (18, 17), (19, 17), (20, 17), (21, 17),
                        (22, 17), (23, 17), (24, 17), (25, 17), (25, 16), (25, 15),
                        (24, 15), (24, 14), (24, 13), (24, 12), (24, 11), (23, 11),
                        (23, 10), (23,  9), (23, 10), (22, 10), (22, 11), (21, 11),
                        (20, 11), (20, 12), (19, 12), (18, 12), (17, 12), (16, 12),
                        (16, 11), (16, 12), (15, 12), (15, 13), (14, 13), (13, 13),
                        (13, 14), (12, 14), (11, 14), (11, 15), (10, 15), (10, 16),
                        (10, 17),
                    ]
                ),
            ],
            [
                np.array(  # [9, 10]
                    [
                        (26, 17), (27, 17), (28, 17), (29, 17), (29, 16), (30, 16),
                        (30, 15), (31, 15), (31, 14), (31, 13), (32, 13), (32, 12),
                        (32, 11), (32, 10), (32,  9), (31,  9), (31,  8), (30,  8),
                        (30,  7), (29,  7), (28,  7), (27,  7), (27,  8), (26,  8),
                        (25,  8), (25,  9), (24,  9), (24, 10), (25, 10), (25, 11),
                        (25, 12), (25, 13), (25, 14), (26, 14), (26, 15), (26, 16),
                        (26, 17),
                    ]
                ),
            ],
        ]
        # fmt: on

        # fmt: off
        s.shells_out_8c = [
            [
                np.array(  # [0, 1, 2]
                    [
                        ( 3, 12), ( 4, 12), ( 5, 11), ( 6, 10), ( 7,  9), ( 8,  9),
                        ( 9,  9), (10,  9), (11,  9), (12,  9), (13,  9), (14,  9),
                        (15,  9), (16,  9), (17,  9), (17,  8), (17,  7), (17,  6),
                        (16,  5), (16,  4), (15,  3), (14,  2), (13,  2), (12,  2),
                        (11,  1), (10,  1), ( 9,  1), ( 8,  1), ( 7,  1), ( 6,  2),
                        ( 5,  2), ( 4,  2), ( 3,  2), ( 2,  3), ( 2,  4), ( 1,  5),
                        ( 1,  6), ( 1,  7), ( 1,  8), ( 1,  9), ( 1, 10), ( 2, 11),
                        ( 3, 12),
                    ]
                ),
            ],
            [
                np.array(  # [3, 5]
                    [
                        ( 7, 14), ( 8, 14), ( 9, 14), (10, 14), (11, 13), (12, 13),
                        (13, 12), (14, 12), (15, 11), (16, 10), (15, 10), (14, 10),
                        (13, 10), (12, 10), (11, 10), (10, 10), ( 9, 10), ( 8, 10),
                        ( 7, 10), ( 6, 11), ( 5, 12), ( 5, 13), ( 6, 13), ( 7, 14),
                    ]
                ),
            ],
            [
                np.array(  # [6, 7, 8]
                    [
                        (10, 17), (11, 17), (12, 17), (13, 17), (14, 17), (15, 17),
                        (16, 17), (17, 17), (18, 17), (19, 17), (20, 17), (21, 17),
                        (22, 17), (23, 17), (24, 17), (25, 17), (24, 16), (24, 15),
                        (24, 14), (24, 13), (24, 12), (24, 11), (24, 10), (24,  9),
                        (23,  9), (22, 10), (21, 11), (20, 11), (19, 12), (18, 12),
                        (17, 12), (16, 11), (15, 12), (14, 13), (13, 13), (12, 14),
                        (11, 14), (10, 15), (10, 16), (10, 17),
                    ]
                ),
            ],
            [
                np.array(  # [9, 10]
                    [
                        (26, 17), (27, 17), (28, 17), (29, 17), (30, 16), (31, 15),
                        (31, 14), (32, 13), (32, 12), (32, 11), (32, 10), (32,  9),
                        (31,  8), (30,  7), (29,  7), (28,  7), (27,  7), (26,  8),
                        (25,  8), (25,  9), (25, 10), (25, 11), (25, 12), (25, 13),
                        (25, 14), (25, 15), (25, 16), (26, 17),
                    ]
                ),
            ],
        ]
        # fmt: on

        s.setUpFeatures("in_4c", "in_8c", "out_4c", "out_8c")

    def test_4c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=4)
        subfeatures = split_regiongrow_levels(
            s.features_in_4c, s.levels, constants=const
        )
        s.assertFeaturesEqual(
            subfeatures, s.features_out_4c, check_shared_pixels=False, sort_by_size=True
        )

    def test_8c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        subfeatures = split_regiongrow_levels(
            s.features_in_8c, s.levels, constants=const
        )
        s.assertFeaturesEqual(
            subfeatures, s.features_out_8c, check_shared_pixels=False, sort_by_size=True
        )


class SplitRegiongrow_Levels23_MinStrength3(TestFeatures_Base):
    """Split the GE1 region iteratively with [2, 3] and min strength 3."""

    make_plot = False

    def setUp(s):
        s.fld = fld_raw
        s.obj_ids_in_8c = obj_ids_ge1_from_ge3
        s.inds_features_in_8c = [list(range(11))]
        s.levels = [2, 3]
        s.min_strength = 3
        s.inds_features_out_8c = [[0, 3], [1, 4], [2, 5]]
        s.inds_neighbors_out_8c = [[1], [0, 2], [1]]
        s.setUpFeatures("in_8c", "out_8c")

    def test_8c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        subfeatures = split_regiongrow_levels(
            s.features_in_8c,
            s.levels,
            seed_min_strength=s.min_strength,
            constants=const,
        )
        s.assertFeaturesEqual(
            subfeatures,
            s.features_out_8c,
            check_boundaries=False,
            check_shared_pixels=False,
        )


class SplitRegiongrow_Levels3_MinSize6(TestFeatures_Base):
    """Split the GE1 region iteratively with [2, 3] and min size 6."""

    make_plot = False

    def setUp(s):
        s.fld = fld_raw
        s.obj_ids_in_8c = obj_ids_ge1_from_ge3_minsize6
        s.inds_features_in_8c = [list(range(11))]
        s.levels = [3]
        s.min_size = 6
        s.inds_features_out_8c = [[0, 2], [1, 3]]
        s.inds_neighbors_out_8c = [[1], [0]]
        s.setUpFeatures("in_8c", "out_8c")

    def test_8c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        subfeatures = split_regiongrow_levels(
            s.features_in_8c, s.levels, seed_minsize=s.min_size, constants=const
        )
        s.assertFeaturesEqual(
            subfeatures,
            s.features_out_8c,
            check_boundaries=False,
            check_shared_pixels=False,
        )


class SplitRegiongrow_SeedRegion_PartialOverlap(TestFeatures_Base):
    """Seed region only partially overlaps with region to split."""

    def setUp(s):
        _ = 0
        # fmt: off
        fld = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
                [_,_,_,_,6, 6,_,_,_,_, _,_,_,_,_, 5,5,5,_,_], #  2
                [_,_,_,_,6, 6,_,_,_,_, _,_,_,_,5, 5,5,5,_,_], #  1
                [_,_,_,_,6, 6,_,_,_,_, _,_,_,5,5, 5,5,_,_,_], # 10

                [_,_,_,2,4, 4,2,2,1,1, _,_,5,5,5, 5,_,_,_,_], #  9
                [_,_,_,2,4, 4,2,1,1,1, 1,1,5,5,5, _,_,_,_,_], #  8
                [_,_,_,2,2, 2,2,1,1,3, 3,3,5,5,_, _,_,_,_,_], #  7
                [_,_,_,2,2, 2,1,1,3,3, 3,3,5,5,_, _,_,_,_,_], #  6
                [_,_,_,2,2, 1,1,3,3,3, 3,5,_,_,_, _,_,_,_,_], #  5

                [_,_,_,_,1, 1,1,3,3,3, 1,_,_,_,_, _,_,_,_,_], #  4
                [_,_,_,_,_, 1,1,1,1,1, _,_,_,_,_, _,_,_,_,_], #  3
                [_,_,_,_,_, 1,1,1,1,1, _,_,_,_,_, _,_,_,_,_], #  2
                [_,_,_,_,_, _,1,1,1,_, _,_,_,_,_, _,_,_,_,_], #  1
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
            ], # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
            np.int32
        ).T[:, ::-1]
        # fmt: on

        s.nx, s.ny = fld.shape

        xy_feature = np.array(np.where((fld > 0) & (fld <= 4))).astype(np.int32)

        xy_seed_l1 = np.array(np.where(fld == 4)).astype(np.int32)
        xy_seed_r1 = np.array(np.where(fld == 3)).astype(np.int32)

        xy_seed_l2 = np.array(np.where((fld == 4) | (fld == 6))).astype(np.int32)
        xy_seed_r2 = np.array(np.where((fld == 3) | (fld == 5))).astype(np.int32)

        xy_split_l = np.array(np.where((fld == 2) | (fld == 4))).astype(np.int32)
        xy_split_r = np.array(np.where((fld == 1) | (fld == 3))).astype(np.int32)

        s.feature = Feature(pixels=xy_feature.T, id_=0)

        s.seed_l1 = Feature(pixels=xy_seed_l1.T, id_=1)
        s.seed_r1 = Feature(pixels=xy_seed_r1.T, id_=2)

        s.seed_l2 = Feature(pixels=xy_seed_l2.T, id_=3)
        s.seed_r2 = Feature(pixels=xy_seed_r2.T, id_=4)

        s.split_l = Feature(pixels=xy_split_l.T, id_=5)
        s.split_r = Feature(pixels=xy_split_r.T, id_=6)

        s.splits = [s.split_l, s.split_r]

    def assertSplitsEqual(s, res, sol):

        res = sorted(res, key=lambda f: f.n)
        sol = sorted(sol, key=lambda f: f.n)

        res_ns = [f.n for f in res]
        sol_ns = [f.n for f in sol]
        s.assertEqual(res_ns, sol_ns)

        res_px = [sorted(f.pixels.tolist()) for f in res]
        sol_px = [sorted(f.pixels.tolist()) for f in sol]
        s.assertEqual(res_px, sol_px)

    def test_full_overlap_s8(s):
        seeds = [s.seed_l1, s.seed_r1]
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        splits = feature_split_regiongrow(s.feature, seeds, constants=const)
        s.assertSplitsEqual(splits, s.splits)

    def test_partial_overlap_s8(s):
        seeds = [s.seed_l2, s.seed_r2]
        const = default_constants(nx=s.nx, ny=s.ny, connectivity=8)
        splits = feature_split_regiongrow(s.feature, seeds, constants=const)
        s.assertSplitsEqual(splits, s.splits)


# SR_TODO
# Add a test with neighbor in a hole instead of at the shell!
# Not the most likely case, but it might still happen in real data!


class SplitRegiongrow_Neighbors(TestFeatures_Base):
    """Split to regions which initially touch and check neighbors."""

    def setUp(s):

        _ = np.nan

        # fmt: off
        s.fld = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
                [_,_,_,_,_, 1,1,1,1,1, _,_,_,_,_, _,_,_,_,_], #  2
                [_,_,_,_,1, 1,1,1,1,1, 1,_,_,_,_, _,_,_,_,_], #  1
                [_,_,_,1,1, 1,1,1,1,1, 1,1,1,1,_, _,_,_,_,_], # 10

                [_,_,_,1,1, 1,2,2,1,1, 1,1,1,1,1, 1,_,_,_,_], #  9
                [_,_,_,1,1, 1,1,1,1,1, 1,1,2,1,1, 1,1,_,_,_], #  8
                [_,_,_,1,1, 1,1,1,1,1, 1,1,2,1,1, 1,1,1,_,_], #  7
                [_,_,_,1,1, 1,2,2,1,1, 1,1,2,1,1, 2,2,1,_,_], #  6
                [_,_,_,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,_,_], #  5

                [_,_,_,_,_, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,_,_], #  4
                [_,_,_,_,_, _,_,_,_,1, 1,1,2,1,1, 1,1,_,_,_], #  3
                [_,_,_,_,_, _,_,_,_,_, 1,1,1,1,1, 1,1,_,_,_], #  2
                [_,_,_,_,_, _,_,_,_,_, _,1,1,1,1, 1,_,_,_,_], #  1
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
        ).T[:, ::-1]
        # fmt: on

        # fmt: off
        s.obj_ids_in_4c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
                [_,_,_,_,_, 0,0,0,0,0, _,_,_,_,_, _,_,_,_,_], #  2
                [_,_,_,_,0, 0,0,0,0,0, 2,_,_,_,_, _,_,_,_,_], #  1
                [_,_,_,0,0, 0,0,0,0,0, 2,2,2,2,_, _,_,_,_,_], # 10

                [_,_,_,0,0, 0,9,9,0,0, 2,2,2,2,2, 4,_,_,_,_], #  9
                [_,_,_,0,0, 0,0,0,0,0, 2,2,7,2,2, 4,4,_,_,_], #  8
                [_,_,_,1,1, 1,1,1,1,1, 2,2,7,2,4, 4,4,4,_,_], #  7
                [_,_,_,1,1, 1,8,8,1,1, 2,2,7,2,4, 5,5,4,_,_], #  6
                [_,_,_,1,1, 1,1,1,1,1, 2,2,2,2,4, 4,4,4,_,_], #  5

                [_,_,_,_,_, 1,1,1,1,1, 3,3,3,3,4, 4,4,4,_,_], #  4
                [_,_,_,_,_, _,_,_,_,1, 3,3,6,3,3, 4,4,_,_,_], #  3
                [_,_,_,_,_, _,_,_,_,_, 3,3,3,3,3, 4,4,_,_,_], #  2
                [_,_,_,_,_, _,_,_,_,_, _,3,3,3,3, 4,_,_,_,_], #  1
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
        ).T[:, ::-1]
        # fmt: on

        # fmt: off
        s.obj_ids_in_8c = np.array(
            [ #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  3
                [_,_,_,_,_, 0,0,0,0,0, _,_,_,_,_, _,_,_,_,_], #  2
                [_,_,_,_,0, 0,0,0,0,0, 2,_,_,_,_, _,_,_,_,_], #  1
                [_,_,_,0,0, 0,0,0,0,0, 2,2,2,2,_, _,_,_,_,_], # 10

                [_,_,_,0,0, 0,9,9,0,0, 2,2,2,2,2, 2,_,_,_,_], #  9
                [_,_,_,0,0, 0,0,0,0,0, 2,2,7,2,2, 4,4,_,_,_], #  8
                [_,_,_,1,1, 1,1,1,1,1, 2,2,7,2,4, 4,4,4,_,_], #  7
                [_,_,_,1,1, 1,8,8,1,1, 2,2,7,2,4, 5,5,4,_,_], #  6
                [_,_,_,1,1, 1,1,1,1,1, 2,2,2,2,4, 4,4,4,_,_], #  5

                [_,_,_,_,_, 1,1,1,1,1, 3,3,3,3,4, 4,4,4,_,_], #  4
                [_,_,_,_,_, _,_,_,_,1, 3,3,6,3,3, 4,4,_,_,_], #  3
                [_,_,_,_,_, _,_,_,_,_, 3,3,3,3,3, 3,4,_,_,_], #  2
                [_,_,_,_,_, _,_,_,_,_, _,3,3,3,3, 3,_,_,_,_], #  1
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
            ] #  0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
        ).T[:, ::-1]
        # fmt: on

        # fmt: off
        s.inds_features_in_4c = [
            [0, 1, 8, 9],
            [2, 3, 4, 5, 6, 7],
        ]
        s.inds_features_in_8c = s.inds_features_in_4c
        s.inds_features_out_4c = [
            [0, 9],
            [1, 8],
            [2, 7],
            [3, 6],
            [4, 5],
        ]
        s.inds_features_out_8c = s.inds_features_out_4c
        s.inds_neighbors_out_4c = [
            [1, 2],
            [0, 2, 3],
            [0, 1, 3, 4],
            [1, 2, 4],
            [2, 3],
        ]
        s.inds_neighbors_out_8c = s.inds_neighbors_out_4c
        # fmt: on

        #
        # Thoughts about shared boundary pixels:
        # - situation a bit ambiguous w/ 8-connectivity
        #
        # - one way: pixel "shared" when it "sees" the other region
        #   -> pixels at triple/quadrupel points see multiple regions
        #   -> sum of "shared boundary pixels" over all neighbors bigger than
        #      the number of boundary pixels
        #
        # - other way: assign each boundary pixel to only one neighbor (or background)
        #   -> ambiguities arise at edges:
        #       - neighbors only connected diagonally at edges have no shared pixels
        #           -> that's not really a problem, though
        #       - edge pixels with two direct neighbors not assignable unambiguously
        #           -> however the same tie-breakers can be used as for region growing
        #
        # => cleanest: implement both (different methods/method argument)
        #   -> use method argument with unique assignment as default
        #
        # - note also that for 4-connectivity, corner pixels have no neighbors,
        #   as they are only directly connected to pixels of the same feature
        #

        # Boundary pixels which "see" respective neighbor feature
        # 8-connectivity: pixels may see multiple neighobrs at once
        # 4-connectivity: neighbors always unique
        # fmt: off
        s.shared_boundary_pixels_out_8c = [
            [
                [  # 0 -> 1
                    ( 9,  8), ( 8,  8), ( 7,  8), ( 6,  8), ( 5,  8), ( 4,  8),
                    ( 3,  8),
                ],
                [  # 0 -> 2
                    ( 9, 12), ( 9, 11), ( 9, 10), ( 9,  9), ( 9,  8),
                ],
                [  # 0 -> BG (background)
                    ( 3,  8), ( 3,  9), ( 3, 10), ( 4, 11), ( 5, 12), ( 6, 12),
                    ( 7, 12), ( 8, 12), ( 9, 12), ( 9, 11),
                ],
                [  # 0 -> IN (interior)
                ],
            ],
            [
                [  # 1 -> 0
                    ( 3,  7), ( 4,  7), ( 5,  7), ( 6,  7), ( 7,  7), ( 8,  7),
                    ( 9,  7),
                ],
                [  # 1 -> 2
                    ( 9,  7), ( 9,  6), ( 9,  5), ( 9,  4),
                ],
                [  # 1 -> 3
                    ( 9,  5), ( 9,  4), ( 9,  3),
                ],
                [  # 1 -> BG (background)
                    ( 9,  4), ( 9,  3), ( 8,  4), ( 7,  4), ( 6,  4), ( 5,  4),
                    ( 4,  5), ( 3,  5), ( 3,  6), ( 3,  7),
                ],
                [  # 1 -> IN (interior)
                ],
            ],
            [
                [  # 2 -> 0
                    (10,  7), (10,  8), (10,  9), (10, 10), (10, 11),
                ],
                [  # 2 -> 1
                    (10,  5), (10,  6), (10,  7), (10,  8),
                ],
                [  # 2 -> 3
                    (13,  5), (12,  5), (11,  5), (10,  5),
                ],
                [  # 2 -> 4
                    (14,  9), (15,  9), (14,  8), (13,  7), (13,  6), (13,  5),
                ],
                [  # 2 -> BG (background)
                    (10, 10), (10, 11), (11, 10), (12, 10), (13, 10), (14,  9),
                    (15,  9),
                ],
                [  # 2 -> IN (interior)
                ],
            ], [
                [  # 3 -> 1
                    (10,  2), (10,  3), (10,  4),
                ],
                [  # 3 -> 2
                    (10,  4), (11,  4), (12,  4), (13,  4),
                ],
                [  # 3 -> 4
                    (13,  4), (14,  3), (15,  2), (15,  1),
                ],
                [  # 3 -> BG (background)
                    (15,  2), (15,  1), (14,  1), (13,  1), (12,  1), (11,  1),
                    (10,  2), (10,  3),
                ],
                [  # 3 -> IN (interior)
                ],
            ],
            [
                [  # 4 -> 2
                    (14,  4), (14,  5), (14,  6), (14,  7), (15,  8), (16,  8),
                ],
                [  # 4 -> 3
                    (16,  3), (16,  2), (15,  3), (14,  4), (14,  5),
                ],
                [  # 4 -> BG (background)
                    (15,  8), (16,  3), (16,  8), (17,  7), (17,  6), (17,  5),
                    (17,  4), (16,  2),
                ],
                [  # 4 -> IN (interior)
                ],
            ],
        ]
        s.shared_boundary_pixels_out_4c = [
            [
                [  # 0 -> 1
                    ( 9,  8), ( 8,  8), ( 7,  8), ( 6,  8), ( 5,  8), ( 4,  8),
                    ( 3,  8),
                ],
                [  # 0 -> 2
                    ( 9, 11), ( 9, 10), ( 9,  9), ( 9,  8),
                ],
                [  # 0 -> BG (background)
                    ( 3,  8), ( 3,  9), ( 3, 10), ( 4, 11), ( 5, 12), ( 6, 12),
                    ( 7, 12), ( 8, 12), ( 9, 12),
                ],
                [  # 0 -> IN (interior)
                    ( 4, 10), ( 5, 11),
                ],
            ],
            [
                [  # 1 -> 0
                    ( 3,  7), ( 4,  7), ( 5,  7), ( 6,  7), ( 7,  7), ( 8,  7),
                    ( 9,  7),
                ],
                [  # 1 -> 2
                    ( 9,  7), ( 9,  6), ( 9,  5),
                ],
                [  # 1 -> 3
                    ( 9,  4), ( 9,  3),
                ],
                [  # 1 -> BG (background)
                    ( 9,  3), ( 8,  4), ( 7,  4), ( 6,  4), ( 5,  4), (4,  5),
                    ( 3,  5), ( 3,  6), ( 3,  7),
                ],
                [ # 1 -> IN (interior)
                    ( 5,  5),
                ],
            ],
            [
                [  # 2 -> 0
                    (10,  8), (10,  9), (10, 10), (10, 11),
                ],
                [  # 2 -> 1
                    (10,  5), (10,  6), (10,  7),
                ],
                [  # 2 -> 3
                    (13,  5), (12,  5), (11,  5), (10,  5),
                ],
                [  # 2 -> 4
                    (14,  9), (14,  8), (13,  7), (13,  6), (13,  5),
                ],
                [  # 2 -> BG (background)
                    (10,  11), (11, 10), (12, 10), (13, 10), (14,  9),
                ],
                [  # 2 -> IN (interior)
                    (13,  9), (13,  8),
                ],
            ],
            [
                [ # 3 -> 1
                    (10,  3), (10,  4),
                ],
                [  # 3 -> 2
                    (10,  4), (11,  4), (12,  4), (13,  4),
                ],
                [  # 3 -> 4
                    (13,  4), (14,  3), (14,  2), (14,  1),
                ],
                [  # 3 -> BG (background)
                    (14,  1), (13,  1), (12,  1), (11,  1), (10,  2),
                ],
                [  # 3 -> IN (interior)
                    (13,  3), (11,  2),
                ],
            ],
            [
                [  # 4 -> 2
                    (15,  9), (15,  8), (14,  7), (14,  6), (14,  5),
                ],
                [  # 4 -> 3
                    (14,  4), (15,  3), (15,  2), (15,  1),
                ],
                [  # 4 -> BG (background)
                    (15,  9), (16,  8), (17,  7), (17,  6), (17,  5), (17,  4),
                    (16,  3), (16,  2), (15,  1),
                ],
                [  # 4 -> IN (interior)
                    (16,  7), (16,  4), (15,  4), (15,  7),
                ],
            ],
        ]
        # fmt: on
        # Boundary pixels uniquely assigned to a neighbor
        # The same tie-breaking rules apply as for region growing
        # fmt: off
        s.shared_boundary_pixels_unique_out_8c = [
            [
                [  # 0 -> 1
                    ( 8,  8), ( 7,  8), ( 6,  8), ( 5,  8), ( 4,  8),
                ],
                [  # 0 -> 2
                    ( 9, 11), ( 9, 10), ( 9,  9), ( 9,  8),
                ],
                [  # 0 -> BG (background)
                    ( 3,  8), ( 3,  9), ( 3, 10), ( 4, 11), ( 5, 12), ( 6, 12),
                    ( 7, 12), ( 8, 12), ( 9, 12),
                ],
                [  # 0 -> IN (interior)
                ],
            ],
            [
                [  # 1 -> 0
                    ( 4,  7), ( 5,  7), ( 6,  7), ( 7,  7), ( 8,  7),
                ],
                [  # 1 -> 2
                    ( 9,  7), ( 9,  6), ( 9,  5),
                ],
                [  # 1 -> 3
                    ( 9,  4),
                ],
                [  # 1 -> BG (background)
                    ( 9,  3), ( 8,  4), ( 7,  4), ( 6,  4), ( 5,  4), ( 4,  5),
                    ( 3,  5), ( 3,  6), ( 3,  7),
                ],
                [ # 1 -> IN (interior)
                ],
            ],
            [
                [  # 2 -> 0
                    (10,  8), (10,  9), (10, 10),
                ],
                [  # 2 -> 1
                    (10,  5), (10,  6), (10,  7),
                ],
                [  # 2 -> 3
                    (12,  5), (11,  5),
                ],
                [  # 2 -> 4
                    (14,  8), (13,  7), (13,  6), (13,  5),
                ],
                [  # 2 -> BG (background)
                    (10, 11), (11, 10), (12, 10), (13, 10), (14,  9), (15,  9),
                ],
                [  # 2 -> IN (interior)
                ],
            ],
            [
                [  # 3 -> 1
                    (10,  3), (10,  4),
                ],
                [  # 3 -> 2
                    (11,  4), (12,  4), (13,  4),
                ],
                [  # 3 -> 4
                    (14,  3), (15,  2),
                ],
                [  # 3 -> BG (background)
                    (15,  1), (14,  1), (13,  1), (12,  1), (11,  1), (10,  2),
                ],
                [  # 3 -> IN (interior)
                ],
            ],
            [
                [  # 4 -> 2
                    (14,  5), (14,  6), (14,  7), (15,  8),
                ],
                [  # 4 -> 3
                    (15,  3), (14,  4),
                ],
                [  # 4 -> BG (background)
                    (16,  8), (17,  7), (17,  6), (17,  5), (17,  4), (16,  3),
                    (16,  2),
                ],
                [  # 4 -> IN (interior)
                ],
            ],
        ]
        s.shared_boundary_pixels_unique_out_4c = [
            [
                [  # 0 -> 1
                    ( 8,  8), ( 7,  8), ( 6,  8), ( 5,  8), ( 4,  8),
                ],
                [  # 0 -> 2
                    ( 9, 11), ( 9, 10), ( 9,  9), ( 9,  8),
                ],
                [  # 0 -> BG (background)
                    ( 3,  8), ( 3,  9), ( 3, 10), ( 4, 11), ( 5, 12), ( 6, 12),
                    ( 7, 12), ( 8, 12), ( 9, 12),
                ],
                [  # 0 -> IN (interior)
                    ( 4, 10), ( 5, 11),
                ],
            ],
            [
                [  # 1 -> 0
                    ( 4,  7), ( 5,  7), ( 6,  7), ( 7,  7), ( 8,  7),
                ],
                [  # 1 -> 2
                    ( 9,  7), ( 9,  6), ( 9,  5),
                ],
                [  # 1 -> 3
                    ( 9,  4),
                ],
                [  # 1 -> BG (background)
                    ( 9,  3), ( 8,  4), ( 7,  4), ( 6,  4), ( 5,  4), ( 4,  5),
                    ( 3,  5), ( 3,  6), ( 3,  7),
                ],
                [  # 1 -> IN (interior)
                    ( 5,  5),
                ],
            ],
            [
                [  # 2 -> 0
                    (10,  8), (10,  9), (10, 10),
                ],
                [  # 2 -> 1
                    (10,  5), (10,  6), (10,  7),
                ],
                [  # 2 -> 3
                    (12,  5), (11,  5),
                ],
                [  # 2 -> 4
                    (14,  8), (13,  7), (13,  6), (13,  5),
                ],
                [  # 2 -> BG (background)
                    (10, 11), (11, 10), (12, 10), (13, 10), (14,  9),
                ],
                [  # 2 -> IN (interior)
                    (13,  9), (13,  8),
                ],
            ],
            [
                [  # 3 -> 1
                    (10,  3), (10,  4),
                ],
                [  # 3 -> 2
                    (11,  4), (12,  4), (13,  4),
                ],
                [  # 3 -> 4
                    (14,  3), (14,  2),
                ],
                [  # 3 -> BG (background)
                    (14,  1), (13,  1), (12,  1), (11,  1), (10,  2),
                ],
                [  # 3 -> IN (interior)
                    (13,  3), (11,  2),
                ],
            ],
            [
                [  # 4 -> 2
                    (14,  5), (14,  6), (14,  7), (15,  8),
                ],
                [  # 4 -> 3
                    (15,  2), (15,  3), (14,  4),
                ],
                [  # 4 -> BG (background)
                    (15,  9), (16,  8), (17,  7), (17,  6), (17,  5), (17,  4),(16,  3),
                    (16,  2), (15,  1),
                ],
                [  # 4 -> IN (interior)
                    (16,  7), (16,  4), (15,  4), (15,  7),
                ],
            ],
        ]
        # fmt: on

        # fmt: off
        s.shells_out_4c = [
            [
                np.array(  # [0, 9]
                    [
                        ( 5, 12), ( 6, 12), ( 7, 12), ( 8, 12), ( 9, 12), ( 9, 11),
                        ( 9, 10), ( 9,  9), ( 9,  8), ( 8,  8), ( 7,  8), ( 6,  8),
                        ( 5,  8), ( 4,  8), ( 3,  8), ( 3,  9), ( 3, 10), ( 4, 10),
                        ( 4, 11), ( 5, 11), ( 5, 12),
                    ]
                ),
            ],
            [
                np.array(  # [1, 8]
                    [
                        ( 3,  7), ( 4,  7), ( 5,  7), ( 6,  7), ( 7,  7), ( 8,  7),
                        ( 9,  7), ( 9,  6), ( 9,  5), ( 9,  4), ( 9,  3), ( 9,  4),
                        ( 8,  4), ( 7,  4), ( 6,  4), ( 5,  4), ( 5,  5), ( 4,  5),
                        ( 3,  5), ( 3,  6), ( 3,  7),
                    ]
                ),
            ],
            [
                np.array(  # [2, 7]
                    [
                        (10, 11), (10, 10), (11, 10), (12, 10), (13, 10), (13,  9),
                        (14,  9), (14,  8), (13,  8), (13,  7), (13,  6), (13,  5),
                        (12,  5), (11,  5), (10,  5), (10,  6), (10,  7), (10,  8),
                        (10,  9), (10, 10), (10, 11),
                    ]
                ),
            ],
            [
                np.array(  # [3, 6]
                    [
                        (10,  4), (11,  4), (12,  4), (13,  4), (13,  3), (14,  3),
                        (14,  2), (14,  1), (13,  1), (12,  1), (11,  1), (11,  2),
                        (10,  2), (10,  3), (10,  4),
                    ]
                ),
            ],
            [
                np.array(  # [4, 5]
                    [
                        (15,  9), (15,  8), (16,  8), (16,  7), (17,  7), (17,  6),
                        (17,  5), (17,  4), (16,  4), (16,  3), (16,  2), (15,  2),
                        (15,  1), (15,  2), (15,  3), (15,  4), (14,  4), (14,  5),
                        (14,  6), (14,  7), (15,  7), (15,  8), (15,  9),
                    ]
                ),
            ],
        ]
        s.shells_out_8c = [
            [
                np.array(  # [0, 9]
                    [
                        ( 5, 12), ( 6, 12), ( 7, 12), ( 8, 12), ( 9, 12), ( 9, 11),
                        ( 9, 10), ( 9,  9), ( 9,  8), ( 8,  8), ( 7,  8), ( 6,  8),
                        ( 5,  8), ( 4,  8), ( 3,  8), ( 3,  9), ( 3, 10), ( 4, 11),
                        ( 5, 12),
                    ]
                ),
            ],
            [
                np.array(  # [1, 8]
                    [
                        ( 3,  7), ( 4,  7), ( 5,  7), ( 6,  7), ( 7,  7), ( 8,  7),
                        ( 9,  7), ( 9,  6), ( 9,  5), ( 9,  4), ( 9,  3), ( 8,  4),
                        ( 7,  4), ( 6,  4), ( 5,  4), ( 4,  5), ( 3,  5), ( 3,  6),
                        ( 3,  7),
                    ]
                ),
            ],
            [
                np.array(  # [2, 7]
                    [
                        (10, 11), (11, 10), (12, 10), (13, 10), (14,  9), (15,  9),
                        (14,  8), (13,  7), (13,  6), (13,  5), (12,  5), (11,  5),
                        (10,  5), (10,  6), (10,  7), (10,  8), (10,  9), (10, 10),
                        (10, 11),
                    ]
                ),
            ],
            [
                np.array(  # [3, 6]
                    [
                        (10,  4), (11,  4), (12,  4), (13,  4), (14,  3), (15,  2),
                        (15,  1), (14,  1), (13,  1), (12,  1), (11,  1), (10,  2),
                        (10,  3), (10,  4),
                    ]
                ),
            ],
            [
                np.array(  # [4, 5]
                    [
                        (15,  8), (16,  8), (17,  7), (17,  6), (17,  5), (17,  4),
                        (16,  3), (16,  2), (15,  3), (14,  4), (14,  5), (14,  6),
                        (14,  7), (15,  8),
                    ]
                ),
            ],
        ]
        # fmt: on
        s.holes_out_4c = [[], [], [], [], []]
        s.holes_out_8c = [[], [], [], [], []]
        s.setUpFeatures("in_4c", "in_8c", "out_4c", "out_8c")

        s.levels = [2]
        s.min_size = 0

    def test_4c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=4)
        subfeatures = split_regiongrow_levels(
            s.features_in_4c, s.levels, seed_minsize=s.min_size, constants=const
        )
        s.assertFeaturesEqual(subfeatures, s.features_out_4c, sort_by_size=True)

    def test_8c(s):
        const = default_constants(nx=s.nxy[0], ny=s.nxy[1], connectivity=8)
        subfeatures = split_regiongrow_levels(
            s.features_in_8c, s.levels, seed_minsize=s.min_size, constants=const
        )
        s.assertFeaturesEqual(subfeatures, s.features_out_8c, sort_by_size=True)


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
            [ #  0 1 2 3 4  5 6
                [_,X,_,_,_, _,_], # 8
                [_,X,_,_,_, X,_], # 7
                [_,X,X,X,X, X,X], # 6
                [_,X,X,X,X, X,X], # 5

                [_,X,X,_,_, X,X], # 4
                [_,X,X,X,X, X,X], # 3
                [_,X,X,X,X, X,_], # 2
                [_,_,X,X,X, _,_], # 1
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
    import logging as log

    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
    log.getLogger().setLevel(log.DEBUG)
    unittest.main()
