#!/usr/bin/env python3

# Standard library
import json
import logging as log
import os
import pytest
import unittest
import sys
from unittest import TestCase

# Third-party
import numpy as np

# First-party
import stormtrack.extra.utilities_misc as io
from stormtrack.extra.cyclone_id.cyclone import CycloneIOWriterJson
from stormtrack.extra.cyclone_id.identify import identify_features
from stormtrack.extra.utilities_misc import Field2D


# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)


@pytest.mark.skip("missing input data")
class TestIdentifyFeatures(TestCase):
    """Run the identification using the same case but different configs.

    The input fields are read from disk (binary), as is the solution (JSON).
    Assertion is done by comparing result and solution as JSON strings.
    If a test fails, both result and solution are written to JSON files.

    """

    DBG = False
    WORK_DIR = "."
    DATA_DIR = "./test_data"
    FILE_DIR = os.path.dirname(os.path.relpath(__file__))
    TOPO_NAME = "HSURF"

    @classmethod
    def setUpClass(cls):
        file_in = "{p}/test_identify.input.npz".format(p=cls.DATA_DIR)
        cls.read_test_data(file_in)

    @classmethod
    def read_test_data(cls, file_name):
        """Read the input data from file. Return data as Field2D objects."""
        if cls.DBG:
            print("READ TEST DATA FROM FILE: {n}".format(n=file_name))
        data = np.load(file_name)
        cls.slp_raw = data["slp"]
        cls.topo_raw = data["topo"]
        cls.lon = data["lon"]
        cls.lat = data["lat"]
        cls.topo = Field2D(cls.topo_raw, cls.lon, cls.lat, name=cls.TOPO_NAME)

    def setUp(s):

        # Create new SLP field
        s.slp = Field2D(s.slp_raw.copy(), s.lon, s.lat)

        # Default configuration
        s.conf = {
            "timings-identify": False,
            "size-boundary-zone": 5,
            "smoothing-sigma": 7.0,
            "force-contours-closed": False,
            "contour-interval": 0.5,
            "depression-min-contours": 1,
            "contour-length-max": -1.0,
            "bcc-fraction": 0.5,
            "min-cyclone-depth": 1.0,
            "extrema-identification-size": 9,
            "max-minima-per-cyclone": 3,
            "contour-length-min": -1.0,
            "topo-cutoff-level": 1500.0,
            "read-slp-contours": False,
            "read-slp-extrema": False,
            "save-slp-contours": False,
            "save-slp-extrema": False,
        }

    def run_identification(s, conf):
        """Run the cyclone identification. Return as a JSON string."""
        if s.DBG:
            print("RUN IDENTIFICATION")
        features = identify_features(s.slp, s.topo, conf)
        writer = CycloneIOWriterJson()
        writer.add_points("MINIMA", s.slp.minima())
        writer.add_points("MAXIMA", s.slp.maxima())
        writer.add_depressions(features["depressions"])
        writer.add_cyclones(features["cyclones"])
        # SR_TMP<
        depressions, cyclones = features["depressions"], features["cyclones"]
        levels = np.arange(950, 1050, 0.5)
        io.plot_depressions("nodypy_depressions.png", depressions, s.slp, levels)
        io.plot_cyclones("nodypy_cyclones.png", cyclones, s.slp, levels)
        # SR_TMP>
        return writer.write_string().strip()

    def read_solution(s, name):
        """Read the solution from file as a JSON string."""
        file_sol = "{p}/test_identify.{n}.sol.json".format(p=s.FILE_DIR, n=name)
        if s.DBG:
            print("READ SOLUTION FROM FILE: {f}".format(f=file_sol))
        with open(file_sol, "r") as f:
            return f.read().strip()

    def evaluate_result(s, name, jstr_res, jstr_sol):
        """Compare result and solution in form of JSON string.

        If the test fails, write both strings to file for comparison.
        """
        if s.DBG:
            print("EVALUATE RESULTS OF TEST: {n}".format(n=name))
        tag = "test_identify"
        file_res = "{p}/tmp.{t}.{n}.res.json".format(p=s.WORK_DIR, t=tag, n=name)
        file_sol = "{p}/{t}.{n}.sol.json".format(p=s.FILE_DIR, t=tag, n=name)
        err_msg = (
            "JSON strings not equal! Dumped result: {r}\n"
            "To check what's wrong, run:\nvimdiff {r} {s}"
        ).format(r=file_res, s=file_sol)
        try:
            s.assertEqual(jstr_res, jstr_sol, err_msg)
        except AssertionError:
            with open(file_res, "w") as f:
                f.write(jstr_res)
            with open(file_sol, "w") as f:
                f.write(jstr_sol)
            raise

    def run_test(s, test_name):
        if s.DBG:
            print("RUN TEST: {n}".format(n=test_name))
        jstr_res = s.run_identification(s.conf)
        jstr_sol = s.read_solution(test_name)
        s.evaluate_result(test_name, jstr_res, jstr_sol)

    # DEFAULT

    def test_default(s):
        s.run_test("test_default")

    # SMOOTHING

    @pytest.mark.skip("takes far too long")
    def test_sig00(s):
        s.conf["smoothing-sigma"] = 0.0
        s.run_test("test_sig00")

    def test_sig03(s):
        s.conf["smoothing-sigma"] = 3.0
        s.run_test("test_sig03")

    @pytest.mark.skip("fails...")
    def test_sig15(s):
        s.conf["smoothing-sigma"] = 15.0
        s.run_test("test_sig15")

    # BCC FRACTION

    def test_bcc00(s):
        s.conf["bcc-fraction"] = 0.0
        s.run_test("test_bcc00")

    def test_bcc03(s):
        s.conf["bcc-fraction"] = 0.3
        s.run_test("test_bcc03")

    def test_bcc06(s):
        s.conf["bcc-fraction"] = 0.6
        s.run_test("test_bcc06")

    def test_bcc10(s):
        s.conf["bcc-fraction"] = 1.0
        s.run_test("test_bcc10")


if __name__ == "__main__":
    unittest.main()
