#!/usr/bin/env python

# Standard library
import json
import logging as log
import unittest
import sys
from pprint import pprint as pp
from unittest import TestCase

# First-party
import stormtrack.extra.io_misc as io
from stormtrack.extra.cyclone_id.cyclone import Cyclone
from stormtrack.extra.cyclone_id.cyclone import CycloneIOReaderJson
from stormtrack.extra.cyclone_id.cyclone import CycloneIOWriterBinary
from stormtrack.extra.cyclone_id.cyclone import CycloneIOWriterJson
from stormtrack.extra.cyclone_id.cyclone import Depression

# Local
from ...testing_utilities import ContourSimple
from ...testing_utilities import PointSimple
from ...testing_utilities import contours_are_sorted
from ...testing_utilities import create_nested_circular_contours as cncc
from ...testing_utilities import shuffle_contours_assert


# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)


# DCC: (3c/1c)3c
CONT_PATH_LIST_DCC = [
    # -> (3c)
    [0, 1.0, [[6.0, 7.0], [7.0, 7.0], [7.0, 8.0], [6.0, 8.0], [6.0, 7.0]]],
    [
        1,
        2.0,
        [
            [6.0, 6.0],
            [7.0, 6.0],
            [8.0, 7.0],
            [8.0, 8.0],
            [7.0, 9.0],
            [6.0, 9.0],
            [5.0, 8.0],
            [5.0, 7.0],
            [6.0, 6.0],
        ],
    ],
    [
        2,
        3.0,
        [
            [6.0, 5.0],
            [7.0, 5.0],
            [9.0, 7.0],
            [9.0, 8.0],
            [7.0, 10.0],
            [6.0, 10.0],
            [4.0, 8.0],
            [4.0, 7.0],
            [6.0, 5.0],
        ],
    ],
    # -> [1c]
    [
        3,
        3.0,
        [
            [11.0, 6.0],
            [12.0, 7.0],
            [12.0, 8.0],
            [11.0, 9.0],
            [10.0, 8.0],
            [10.0, 7.0],
            [11.0, 6.0],
        ],
    ],
    # -> 3c
    [
        4,
        4.0,
        [
            [6.0, 3.0],
            [10.0, 3.0],
            [13.0, 6.0],
            [13.0, 9.0],
            [10.0, 12.0],
            [6.0, 12.0],
            [3.0, 9.0],
            [3.0, 6.0],
            [6.0, 3.0],
        ],
    ],
    [
        5,
        5.0,
        [
            [5.0, 2.0],
            [11.0, 2.0],
            [14.0, 5.0],
            [14.0, 10.0],
            [11.0, 13.0],
            [5.0, 13.0],
            [2.0, 10.0],
            [2.0, 5.0],
            [5.0, 2.0],
        ],
    ],
    [
        6,
        6.0,
        [
            [4.0, 1.0],
            [12.0, 1.0],
            [15.0, 4.0],
            [15.0, 11.0],
            [12.0, 14.0],
            [4.0, 14.0],
            [1.0, 11.0],
            [1.0, 4.0],
            [4.0, 1.0],
        ],
    ],
]
# SCC: 2c
CONT_PATH_LIST_SCC = [
    [7, 3.0, [[-6.0, 6.0], [-3.0, 6.0], [-3.0, 9.0], [-6.0, 9.0], [-6.0, 6.0]]],
    [
        8,
        4.0,
        [
            [-6.0, 4.0],
            [-3.0, 4.0],
            [-1.0, 6.0],
            [-1.0, 9.0],
            [-3.0, 11.0],
            [-6.0, 11.0],
            [-8.0, 9.0],
            [-8.0, 6.0],
            [-6.0, 4.0],
        ],
    ],
]
# NO-SCC: 1c (not deep enough)
CONT_PATH_LIST_NO_SCC = [
    [9, 4.0, [[-9.0, 3.0], [-7.0, 4.0], [-8.0, 5.0], [-9.0, 4.0], [-9.0, 3.0]]]
]
# Depression: 2 SCCs with single endlosing contour (doesn't qualify as DCC)
CONT_PATH_LIST_DEP = (
    CONT_PATH_LIST_SCC
    + CONT_PATH_LIST_NO_SCC
    + [
        [
            10,
            5.0,
            [
                [-9.0, 2.0],
                [-4.0, 2.0],
                [0.0, 6.0],
                [0.0, 9.0],
                [-3.0, 12.0],
                [-6.0, 12.0],
                [-10.0, 8.0],
                [-10.0, 3.0],
                [-9.0, 2.0],
            ],
        ]
    ]
)
CONT_PATH_LIST = CONT_PATH_LIST_DCC + CONT_PATH_LIST_DEP


# Minima (id,level,coords) (one for each innermost contour)
MIN_COORD_LIST_DCC = [[0, 0.5, [6.5, 7.5]], [1, 2.5, [11.0, 7.5]]]
MIN_COORD_LIST_SCC = [[2, 2.5, [-4.5, 7.5]]]
MIN_COORD_LIST_NO_SCC = [[3, 3.5, [-8.0, 4.0]]]
MIN_COORD_LIST_DEP = MIN_COORD_LIST_SCC + MIN_COORD_LIST_NO_SCC
MIN_COORD_LIST = MIN_COORD_LIST_DCC + MIN_COORD_LIST_DEP


# Create point objects
MIN_LIST = [PointSimple(x, y, z, id=i) for i, z, (x, y) in MIN_COORD_LIST]

# Create contour objects
CONT_LIST = [ContourSimple(path, level=lvl, id=i) for i, lvl, path in CONT_PATH_LIST]

# Create depression objects
DEPR_LIST = sorted(Depression.create(CONT_LIST, minima=MIN_LIST, id0=0))

# Create cyclone objects
CYCL_LIST = sorted(Cyclone.create(DEPR_LIST, min_depth=1.0, thresh_dcc=0.5, id0=0))


# Random copy/paste from a run with some randomly adapted params
# (i.e. not necessarily consistent with the contours or anything)
CONFIG_WRITTEN_SECTIONS = ["IDENTIFY"]
CONFIG = {
    "GENERAL": {
        "topofile-path": "/foo/bar/cyclone_tracking/tracking",
        "image-format": "PNG",
        "save-images": True,
        "topo-field-name": "HSURF",
        "topofile": "LMCONSTANTS",
        "infile-list": ["lffd2001012806.nc"],
        "infile-path": "/foo/bar/cyclone_tracking/tracking",
        "output-path": "/foo/bar/cyclone_tracking/tracking/output.dev",
        "input-field-name": "PMSL",
    },
    "IDENTIFY": {
        "extrema-identification-size": 9,
        "contour-length-max": -1.0,
        "contour-interval": 0.5,
        "force-contours-closed": False,
        "contour-length-min": 100.0,
        "min-contour-depth": 1.0,
        "max-minima-per-cyclone": 3,
        "topo-cutoff-level": 1600.0,
        "size-boundary-zone": 4,
        "smoothing-sigma": 6.6,
    },
    "INTERNAL": {
        "dump-ini": False,
        "dump-ini-default": False,
        "dump-conf-default": False,
        "debug": False,
        "verbose": True,
        "print-conf": False,
        "dump-conf": False,
    },
}


# io.plot_contours("test_write_string_contours.png",
#       CONT_LIST,MIN_LIST,labels=lambda x:x.lvl)


class TestWriteComponents(TestCase):
    """Write the resp. objects (e.g. Contour) to JSON and verify the result.

    To check the dump, parse the resulting JSON string and reconstruct the
    nested list structure initially used to initialize the objects (or a
    similar list structure).
    """

    def setUp(s):
        s.writer = CycloneIOWriterJson()

    def test_config(s):
        s.writer.add_config(CONFIG)
        jstr = s.writer.write_string()
        jdat = json.loads(jstr)["CONFIG"]
        res = {name: jdat[name] for name in CONFIG_WRITTEN_SECTIONS}
        sol = {k: v for k, v in CONFIG.items() if k in CONFIG_WRITTEN_SECTIONS}
        for key, val in res.items():
            s.assertDictEqual(val, sol[key])

    def test_minima(s):
        s.writer.add_points("MINIMA", MIN_LIST)
        jstr = s.writer.write_string()
        jdat = json.loads(jstr)["POINTS_MINIMA"]
        res = [[p["id"], p["level"], [p["lon"], p["lat"]]] for p in jdat]
        s.assertCountEqual(res, MIN_COORD_LIST)

    def test_contours(s):
        s.writer.add_contours(CONT_LIST)
        jstr = s.writer.write_string()
        jdat = json.loads(jstr)["CONTOURS"]
        res = [[c["id"], c["level"], c["path"]] for c in jdat]
        s.assertCountEqual(res, CONT_PATH_LIST)

    def test_depressions(s):
        s.writer.add_depressions(DEPR_LIST)
        jstr = s.writer.write_string()
        jdat = json.loads(jstr)["DEPRESSIONS"]

        # Extract data of the two depressions
        s.assertEqual(len(jdat), 2)
        dep1, dep2 = jdat

        s.assert_depression(CONT_PATH_LIST_DEP, MIN_COORD_LIST_DEP, dep1)
        s.assert_depression(CONT_PATH_LIST_DCC, MIN_COORD_LIST_DCC, dep2)

    def test_cyclones(s):
        s.writer.add_cyclones(CYCL_LIST)
        jstr = s.writer.write_string()
        jdat = json.loads(jstr)["CYCLONES"]

        # Extract data of the three cyclones
        s.assertEqual(len(jdat["DCC"]), 1)
        s.assertEqual(len(jdat["SCC"]), 1)
        s.assertEqual(len(jdat["TCC"]), 0)
        dcc = jdat["DCC"][0]
        scc = jdat["SCC"][0]

        s.assert_cyclone(CONT_PATH_LIST_DCC, MIN_COORD_LIST_DCC, dcc)
        s.assert_cyclone(CONT_PATH_LIST_SCC, MIN_COORD_LIST_SCC, scc)

    def assert_depression(s, contours, minima, data):

        # Number of contours
        sol_ncont = len(contours)
        res_ncont = data["n_contours"]
        s.assertEqual(res_ncont, sol_ncont)

        # Contour IDs
        sol_cont = [c[0] for c in contours]
        res_cont = data["contours_id"]
        s.assertCountEqual(res_cont, sol_cont)

        # Number of minima
        sol_nmin = len(minima)
        res_nmin = len(data["minima_id"])
        s.assertEqual(res_nmin, sol_nmin)

        # Minima IDs
        sol_min = [m[0] for m in minima]
        res_min = data["minima_id"]
        s.assertCountEqual(res_min, sol_min)

    def assert_cyclone(s, cont, minima, depr):

        s.assert_depression(cont, minima, depr)

        # Minimal and maximal depth
        def assert_depth(type, cont, minima, depr):
            fct = max if type == "min" else (min if type == "max" else None)
            res_depth = depr[type + "_depth"]
            sol_depth = max([c[1] for c in cont]) - fct([m[1] for m in minima])
            s.assertEqual(res_depth, sol_depth)

        assert_depth("min", cont, minima, depr)
        assert_depth("max", cont, minima, depr)


class TestReadComponents(TestCase):
    def setUp(s):
        s.writer = CycloneIOWriterJson()
        s.reader = CycloneIOReaderJson()

    def test_config(s):
        s.writer.add_config(CONFIG)
        jstr = s.writer.write_string()
        res = s.reader.read_string(jstr)["CONFIG"]
        sol = {k: v for k, v in CONFIG.items() if k in CONFIG_WRITTEN_SECTIONS}
        for key, val in res.items():
            s.assertDictEqual(val, sol[key])

    def test_minima(s):
        s.writer.add_points("MINIMA", MIN_LIST)
        jstr = s.writer.write_string()
        res = s.reader.read_string(jstr)["POINTS_MINIMA"]
        s.assertCountEqual(res, MIN_LIST)

    def test_contours(s):
        s.writer.add_contours(CONT_LIST)
        jstr = s.writer.write_string()
        res = s.reader.read_string(jstr)["CONTOURS"]
        s.assertEqual(res, CONT_LIST)

    def test_depressions_core(s):
        """Test the core method "rebuild_depressions"."""
        s.writer.add_depressions(DEPR_LIST)
        jstr = s.writer.write_string()
        jdat = json.loads(jstr)["DEPRESSIONS"]
        res = s.reader.rebuild_depressions(jdat, CONT_LIST, MIN_LIST)
        s.assertEqual(res, DEPR_LIST)

    def test_depressions(s):
        """Test the general method "read_string"."""
        s.writer.add_depressions(DEPR_LIST)
        s.writer.add_contours(CONT_LIST)
        s.writer.add_points("MINIMA", MIN_LIST)
        jstr = s.writer.write_string()
        res = s.reader.read_string(jstr)["DEPRESSIONS"]
        s.assertEqual(res, DEPR_LIST)

    def test_cyclones_core(s):
        """Test the core method "rebuild_cyclones"."""
        s.writer.add_cyclones(CYCL_LIST)
        jstr = s.writer.write_string()
        jdat = json.loads(jstr)["CYCLONES"]
        res = s.reader.rebuild_cyclones(jdat, CONT_LIST, MIN_LIST)
        res.sort()
        s.assertEqual(res, CYCL_LIST)

    def test_cyclones(s):
        """Test the general method "read_string"."""
        s.writer.add_cyclones(CYCL_LIST)
        s.writer.add_contours(CONT_LIST)
        s.writer.add_points("MINIMA", MIN_LIST)
        jstr = s.writer.write_string()
        res = s.reader.read_string(jstr)["CYCLONES"]
        res.sort()
        s.assertEqual(res, CYCL_LIST)


if __name__ == "__main__":
    unittest.main()
