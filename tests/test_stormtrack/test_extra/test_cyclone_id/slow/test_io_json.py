#!/usr/bin/env python

# Standard library
import json
import logging as log
import os
import pytest
import unittest
import sys
from pprint import pprint as pp
from unittest import TestCase

# First-party
from stormtrack.extra.cyclone_id.cyclone import CycloneIOReaderJson
from stormtrack.extra.cyclone_id.cyclone import CycloneIOWriterJson
from stormtrack.extra.cyclone_id.cyclone import CycloneIOWriterBinary
from stormtrack.extra.tracking_old.tracking import FeatureTrackIOReaderJson
from stormtrack.extra.tracking_old.tracking import FeatureTrackIOWriterJson


# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)


class TestCycloneIO(TestCase):
    """Test pure JSON input/output."""

    def setUp(s):
        file_dir = os.path.dirname(os.path.relpath(__file__))
        data_dir = "./test_data"
        work_dir = "."

        name_base_json = "test_io_json.output_pure"
        s.file_in_json = "{}/{}.json".format(data_dir, name_base_json)
        s.file_out_json = "{}/tmp.{}.json".format(work_dir, name_base_json)

        name_base_hybrid = "test_io_json.output_hybrid"
        s.in_hybrid_json = "{}/{}.json".format(file_dir, name_base_hybrid)
        s.in_hybrid_bin = "{}/{}.npz".format(data_dir, name_base_hybrid)
        s.out_hybrid_json = "{}/tmp.{}.json".format(work_dir, name_base_hybrid)
        s.out_hybrid_bin = "{}/tmp.{}.npz".format(work_dir, name_base_hybrid)

        s.reader = CycloneIOReaderJson()
        s.writer_json = CycloneIOWriterJson()
        s.writer_bin = CycloneIOWriterBinary()

    @unittest.skip("missing input data")
    @pytest.mark.skip("missing input data")
    def test_pure_json(s):
        """Read and write pure JSON output.

        Read the input file, extract the data, and write it again.
        The resulting output file must be identical to the input file.
        """
        # Read input file and extract data
        data = s.reader.read_file(s.file_in_json)
        head = s.reader.get_header()
        conf = data["CONFIG"]
        mins = data["POINTS_MINIMA"]
        cont = data["CONTOURS"]
        depr = data["DEPRESSIONS"]
        cycl = data["CYCLONES"]

        # Write the data as pure JSON output
        s.writer_json.set_config(**head)
        s.writer_json.add_config(conf)
        s.writer_json.add_points("MINIMA", mins)
        s.writer_json.add_cyclones(cycl)
        s.writer_json.add_depressions(depr)
        s.writer_json.add_contours(cont)
        s.writer_json.write_file(s.file_out_json)

        # Read input and output files into string
        with open(s.file_in_json, "r") as fi:
            sol = fi.read().strip()
        with open(s.file_out_json, "r") as fo:
            res = fo.read().strip()

        # Compare the file contents (string comparison)
        err_msg = (
            "To compare the result to the solution, run:\n" "vimdiff {r} {s}"
        ).format(s=s.file_in_json, r=s.file_out_json)
        s.assertEqual(sol, res, err_msg)

        # Clean up
        os.remove(s.file_out_json)

    @unittest.skip("missing input data")
    @pytest.mark.skip("missing input data")
    def test_hybrid_json_binary(s):
        """Read and write all mixed JSON-binary output.

        In contrast do pure JSON output, the contour paths are written to
        a binary file (largest amount of data).

        Read the pure JSON output file and extract the data. Then write
        mixed JSON-binary output, read it again, and extract the data.
        Verify the test by comparing to the data read from pure JSON.
        """
        # Read initial data from pure JSON file
        data_ini = s.reader.read_file(s.in_hybrid_json)
        conf = data_ini["CONFIG"]
        mins = data_ini["POINTS_MINIMA"]
        cont = data_ini["CONTOURS"]
        depr = data_ini["DEPRESSIONS"]
        cycl = data_ini["CYCLONES"]

        # Write the data as hybrid JSON-binary output
        s.writer_json.add_config(conf)
        s.writer_json.add_points("MINIMA", mins)
        s.writer_json.add_cyclones(cycl)
        s.writer_json.add_depressions(depr)
        s.writer_json.add_contours(cont)
        s.writer_json.set_config(
            contour_path_file=s.out_hybrid_bin, save_paths=False, path_digits=3
        )
        s.writer_json.write_file(s.out_hybrid_json)
        s.writer_bin.write_contour_path_file(s.out_hybrid_bin, cont)

        # Read the binary output and check the data
        data = s.reader.read_file(s.out_hybrid_json)
        s.assertEqual(data, data_ini)

        # Clean up
        os.remove(s.out_hybrid_json)
        os.remove(s.out_hybrid_bin)


class TestTrackIO(TestCase):
    """Test pure JSON input/output."""

    def setUp(s):
        file_dir = os.path.dirname(os.path.relpath(__file__))
        data_dir = "./test_data"
        work_dir = "."

        name_base_json = "test_track_io_json.output_pure"
        s.file_in_json = "{}/{}.json".format(data_dir, name_base_json)
        s.file_out_json = "{}/tmp.{}.json".format(work_dir, name_base_json)

        name_base_hybrid = "test_track_io_json.output_hybrid"
        s.in_hybrid_json = "{}/{}.json".format(file_dir, name_base_hybrid)
        s.in_hybrid_bin = "{}/{}.npz".format(data_dir, name_base_hybrid)
        s.out_hybrid_json = "{}/tmp.{}.json".format(work_dir, name_base_hybrid)
        s.out_hybrid_bin = "{}/tmp.{}.npz".format(work_dir, name_base_hybrid)

        s.reader = FeatureTrackIOReaderJson()
        s.writer_json = FeatureTrackIOWriterJson()
        # s.writer_bin = FeatureTrackIOWriterBinary()

    @unittest.skip("add method rebuild_features (?)")
    @pytest.mark.skip("add method rebuild_features (?)")
    def test_pure_json(s):
        """Read and write pure JSON output.

        Read the input file, extract the data, and write it again.
        The resulting output file must be identical to the input file.
        """
        # Read input file and extract data
        data = s.reader.read_file(s.file_in_json)
        head = s.reader.get_header()
        # conf = data["CONFIG"] #SR_TODO
        tracks = data["TRACKS"]

        # Write the data as pure JSON output
        s.writer_json.set_config(**head)
        # s.writer_json.add_config(conf) #SR_TODO
        s.writer_json.add_tracks(tracks)
        s.writer_json.write_file(s.file_out_json)

        # Read input and output files into string
        with open(s.file_in_json, "r") as fi:
            sol = fi.read().strip()
        with open(s.file_out_json, "r") as fo:
            res = fo.read().strip()

        # Compare the file contents (string comparison)
        err_msg = (
            "To compare the result to the solution, run:\n" "vimdiff {r} {s}"
        ).format(s=s.file_in_json, r=s.file_out_json)
        s.assertEqual(sol, res, err_msg)

        # Clean up
        os.remove(s.file_out_json)

    @unittest.skip("Not Implemented")
    @pytest.mark.skip("Not Implemented")
    def test_hybrid_json_binary(s):
        """Read and write all mixed JSON-binary output.

        In contrast do pure JSON output, the contour paths are written to
        a binary file (largest amount of data).

        Read the pure JSON output file and extract the data. Then write
        mixed JSON-binary output, read it again, and extract the data.
        Verify the test by comparing to the data read from pure JSON.
        """
        # Read initial data from pure JSON file
        data_ini = s.reader.read_file(s.in_hybrid_json)
        conf = data_ini["CONFIG"]
        tracks = data["TRACKS"]

        # Write the data as hybrid JSON-binary output
        s.writer_json.add_config(conf)
        s.writer_json.add_tracks(tracks)
        # ...
        s.writer_bin.write_contour_path_file(s.out_hybrid_bin, cont)

        # Read the binary output and check the data
        data = s.reader.read_file(s.out_hybrid_json)
        s.assertEqual(data, data_ini)

        # Clean up
        os.remove(s.out_hybrid_json)
        os.remove(s.out_hybrid_bin)


if __name__ == "__main__":
    unittest.main()
