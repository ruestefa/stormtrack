#!/usr/bin/env python3

# Standard library
import itertools
import os
import sys
import unittest
from unittest import TestCase

# Third-party
import numpy as np

# First-party
from stormtrack.core.identification import Feature
from stormtrack.utils.various import import_module


# Define tolerances for area deviations based on setup
# They are defined by increasing specificity and later searched inversely

tol_pct_by_setup = [
    ({}, 1.0),
    ({"method": "dyntools", "delta": 0.5, "clat": 0}, 1.1),
    ({"method": "dyntools", "delta": 0.5, "clat": 10}, 1.1),
    ({"method": "dyntools", "delta": 0.5, "clat": 90}, 1.7),
    ({"method": "dyntools", "delta": 1.0, "clat": 40}, 2.1),
    ({"method": "pyproj"}, 10.0),
    ({"method": "pyproj", "clat": 40}, 8.0),
    ({"method": "pyproj", "clat": 80}, 30.0),
    ({"method": "pyproj", "delta": 0.5, "clat": 60}, 15.0),
    ({"method": "pyproj", "delta": 0.5, "clat": 70}, 20.0),
    ({"method": "pyproj", "delta": 0.5, "clat": 90}, 60.0),
    ({"method": "pyproj", "delta": 1.0, "clat": 0}, 20.0),
]


def get_tol_pct(setup):
    for setup_i, tol_pct in tol_pct_by_setup[::-1]:
        for key, val in setup_i.items():
            if setup[key] != val:
                break
        else:
            return tol_pct
    return 0


# dyntls d0.01 lat00 r800  800  800   0.05%  2012524  2010619   0.09%
# dyntls d0.01 lat40 r800  800  800   0.05%  2012469  2010619   0.09%
# dyntls d0.01 lat80 r800  800  800   0.05%  2012498  2010619   0.09%
# dyntls d0.05 lat00 r800  800  800   0.03%  2011991  2010619   0.07%
# dyntls d0.05 lat40 r800  800  800   0.04%  2012047  2010619   0.07%
# dyntls d0.05 lat80 r800  800  800   0.04%  2012306  2010619   0.08%
# dyntls d0.10 lat00 r800  800  799   0.06%  2008134  2010619   0.12%
# dyntls d0.10 lat40 r800  800  800   0.01%  2010918  2010619   0.01%
# dyntls d0.10 lat80 r800  800  800   0.03%  2011639  2010619   0.05%
# dyntls d0.50 lat00 r800  800  804   0.51%  2031339  2010619   1.03%
# dyntls d0.50 lat10 r800  800  804   0.50%  2030850  2010619   1.01%
# dyntls d0.50 lat20 r800  800  802   0.34%  2024349  2010619   0.68%
# dyntls d0.50 lat30 r800  800  800   0.09%  2014300  2010619   0.18%
# dyntls d0.50 lat40 r800  800  801   0.18%  2017778  2010619   0.36%
# dyntls d0.50 lat50 r800  800  801   0.19%  2018405  2010619   0.39%
# dyntls d0.50 lat60 r800  800  802   0.27%  2021356  2010619   0.53%
# dyntls d0.50 lat70 r800  800  800   0.06%  2013000  2010619   0.12%
# dyntls d0.50 lat80 r800  800  801   0.23%  2019735  2010619   0.45%
# dyntls d0.50 lat90 r800  800  806   0.83%  2043972  2010619   1.66%
# dyntls d1.00 lat00 r800  800  796   0.48%  1991213  2010619   0.97%
# dyntls d1.00 lat40 r800  800  807   1.00%  2051019  2010619   2.01%
# dyntls d1.00 lat80 r800  800  803   0.40%  2026873  2010619   0.81%

# pyproj d0.05 lat00 r800  800  802   0.31%  2023099  2010619   0.62%
# pyproj d0.05 lat40 r800  800  769   3.83%  1859371  2010619   7.52%
# pyproj d0.05 lat80 r800  800  673  15.83%  1424349  2010619  29.16%
# pyproj d0.10 lat00 r800  800  807   0.89%  2046655  2010619   1.79%
# pyproj d0.10 lat40 r800  800  772   3.47%  1873414  2010619   6.82%
# pyproj d0.10 lat80 r800  800  674  15.64%  1430918  2010619  28.83%
# pyproj d0.50 lat00 r800  800  819   2.48%  2111540  2010619   5.02%
# pyproj d0.50 lat10 r800  800  817   2.14%  2097702  2010619   4.33%
# pyproj d0.50 lat20 r800  800  813   1.69%  2078968  2010619   3.40%
# pyproj d0.50 lat30 r800  800  804   0.55%  2032705  2010619   1.10%
# pyproj d0.50 lat40 r800  800  787   1.56%  1948250  2010619   3.10%
# pyproj d0.50 lat50 r800  800  764   4.47%  1835046  2010619   8.73%
# pyproj d0.50 lat60 r800  800  746   6.72%  1749311  2010619  13.00%
# pyproj d0.50 lat70 r800  800  717  10.27%  1618814  2010619  19.49%
# pyproj d0.50 lat80 r800  800  685  14.33%  1475816  2010619  26.60%
# pyproj d0.50 lat90 r800  800  529  33.76%   882282  2010619  56.12%
# pyproj d1.00 lat00 r800  800  871   8.94%  2386068  2010619  18.67%
# pyproj d1.00 lat40 r800  800  815   1.96%  2090214  2010619   3.96%
# pyproj d1.00 lat80 r800  800  699  12.56%  1537328  2010619  23.54%


class Test_Base(TestCase):

    # Method used to compute the areas from the pixels
    # Note: 'grid' seems to be more precise than 'proj'
    # method_comp_area = "proj"
    method_comp_area = "grid"

    # Whether to run the tests (True) or just print the results (False)
    check_results = True
    # check_results=False

    def create_feature(self):
        """Create feature from self.mask."""

        if not hasattr(self, "mask"):
            raise Exception("attribute self.mask missing")

        pixels = np.asarray(np.where(self.mask), np.int32).T

        self.feature = Feature(pixels)

    def comp_feature_area(self):
        if self.method_comp_area == "grid":
            lon, lat = self.lon1d, self.lat1d
        elif self.method_comp_area == "proj":
            lon, lat = self.lon2d, self.lat2d
        else:
            raise ValueError("mode='" + mode + "'")
        return self.feature.area_lonlat(lon, lat, method=self.method_comp_area)

    def print_res_sol(self, area_res, area_sol):
        """Helper method to print result and solution with the error."""
        rad_sol = self.rad_km
        rad_res = np.sqrt(area_res / np.pi)
        err_rad = abs(rad_res - rad_sol) / rad_sol
        err_area = abs(area_res - area_sol) / area_sol
        print(
            "\r{} {:4} {:4} {:7.2%} {:8} {:8} {:7.2%}".format(
                self.__class__.__name__.lstrip("Test_"),
                int(self.rad_km),
                int(rad_res),
                err_rad,
                int(area_res),
                int(area_sol),
                err_area,
            )
        )

    def eval_test(self, area_res, area_sol, tol_pct=None):

        if tol_pct is None:
            tol_pct = get_tol_pct(self.setup)

        if self.check_results:
            rel_err_pct = 100 * abs(area_res - area_sol) / area_sol
            msg = ("area differs by {:.1f}% > {}%: {} km2 != {} km2").format(
                rel_err_pct, tol_pct, area_res, area_sol
            )
            self.assertTrue(rel_err_pct < tol_pct, msg)

        else:
            self.print_res_sol(area_res, area_sol)


class Test_dyntls_d1p00_lat00_r800(Test_Base):

    setup = dict(clat=0, rad=800, delta=1.0, method="dyntools")

    def setUp(s):

        s.clon, s.clat = 0.0, 0.0
        s.rad_km = 800.0
        s.area_km2 = np.pi * s.rad_km ** 2

        s.nlat, s.nlon = 17, 17
        s.lat1d = np.linspace(-8.0, 8.0, s.nlat)
        s.lon1d = np.linspace(-8.0, 8.0, s.nlon)
        s.lat2d, s.lon2d = np.meshgrid(s.lat1d, s.lon1d)

        _, X = 0, 1
        # fmt: off
        s.mask = np.array(
            [
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,X,X,X,_,_,_,_,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,_,_,_,_,X,X,X,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
            ],
            np.bool,
        ).T[:, ::-1]
        # fmt: on

        s.create_feature()

    def test_area(s):
        res = s.comp_feature_area()
        sol = s.area_km2
        s.eval_test(res, sol)


class Test_dyntls_d1p00_lat40_r800(Test_Base):

    setup = dict(clat=40, rad=800, delta=1.0, method="dyntools")

    def setUp(s):

        s.clon, s.clat = 0.0, 40.0
        s.rad_km = 800.0
        s.area_km2 = np.pi * s.rad_km ** 2

        s.nlat, s.nlon = 17, 21
        s.lat1d = np.linspace(32.0, 48.0, s.nlat)
        s.lon1d = np.linspace(-10.0, 10.0, s.nlon)
        s.lat2d, s.lon2d = np.meshgrid(s.lat1d, s.lon1d)

        _, X = 0, 1
        # fmt: off
        s.mask = np.array(
            [
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,X,X,X,X,X,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_],
                [_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,X,X,X,X,X,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
            ],
            np.bool,
        ).T[:, ::-1]
        # fmt: on

        s.create_feature()

    def test_area(s):
        res = s.comp_feature_area()
        sol = s.area_km2
        s.eval_test(res, sol)


class Test_dyntls_d1p00_lat80_r800(Test_Base):

    setup = dict(clat=80, rad=800, delta=1.0, method="dyntools")

    def setUp(s):

        s.clon, s.clat = 0.0, 80.0
        s.rad_km = 800.0
        s.area_km2 = np.pi * s.rad_km ** 2

        s.nlat, s.nlon = 17, 95
        s.lat1d = np.linspace(72.0, 88.0, s.nlat)
        s.lon1d = np.linspace(-47.0, 47.0, s.nlon)
        s.lat2d, s.lon2d = np.meshgrid(s.lat1d, s.lon1d)

        _, X = 0, 1
        # fmt: off
        s.mask = np.array(
            [
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
            ],
            np.bool
        ).T[:, ::-1]
        # fmt: on

        s.create_feature()

    def test_area(s):
        res = s.comp_feature_area()
        sol = s.area_km2
        s.eval_test(res, sol)


class Test_pyproj_d1p00_lat00_r800(Test_Base):

    setup = dict(clat=0, rad=800, delta=1.0, method="pyproj")

    def setUp(s):

        s.clon, s.clat = 0.0, 0.0
        s.rad_km = 800.0
        s.area_km2 = np.pi * s.rad_km ** 2

        s.nlat, s.nlon = 17, 17
        s.lat1d = np.linspace(-8.0, 8.0, s.nlat)
        s.lon1d = np.linspace(-8.0, 8.0, s.nlon)
        s.lat2d, s.lon2d = np.meshgrid(s.lat1d, s.lon1d)

        _, X = 0, 1
        # fmt: off
        s.mask = np.array(
            [
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
            ],
            np.bool,
        ).T[:, ::-1]
        # fmt: on

        s.create_feature()

    def test_area(s):
        res = s.comp_feature_area()
        sol = s.area_km2
        s.eval_test(res, sol)


class Test_pyproj_d1p00_lat40_r800(Test_Base):

    setup = dict(clat=40, rad=800, delta=1.0, method="pyproj")

    def setUp(s):

        s.clon, s.clat = 0.0, 40.0
        s.rad_km = 800.0
        s.area_km2 = np.pi * s.rad_km ** 2

        s.nlat, s.nlon = 17, 21
        s.lat1d = np.linspace(32.0, 48.0, s.nlat)
        s.lon1d = np.linspace(-10.0, 10.0, s.nlon)
        s.lat2d, s.lon2d = np.meshgrid(s.lat1d, s.lon1d)

        _, X = 0, 1
        # fmt: off
        s.mask = np.array(
            [
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,_,_,_,_,X,X,X,X,X,X,X,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
            ],
            np.bool,
        ).T[:, ::-1]
        # fmt: on

        s.create_feature()

    def test_area(s):
        res = s.comp_feature_area()
        sol = s.area_km2
        s.eval_test(res, sol)


class Test_pyproj_d1p00_lat80_r800(Test_Base):

    setup = dict(clat=80, rad=800, delta=1.0, method="pyproj")

    def setUp(s):

        s.clon, s.clat = 0.0, 80.0
        s.rad_km = 800.0
        s.area_km2 = np.pi * s.rad_km ** 2

        s.nlat, s.nlon = 17, 75
        s.lat1d = np.linspace(72.0, 88.0, s.nlat)
        s.lon1d = np.linspace(-37.0, 37.0, s.nlon)
        s.lat2d, s.lon2d = np.meshgrid(s.lat1d, s.lon1d)

        _, X = 0, 1
        # fmt: off
        s.mask = np.array(
            [
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_],
                [_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_],
                [_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_],
                [_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_],
                [_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
                [_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_],
            ],
            np.bool,
        ).T[:, ::-1]
        # fmt: on

        s.create_feature()

    def test_area(s):
        res = s.comp_feature_area()
        sol = s.area_km2
        s.eval_test(res, sol)


# Automatic tests based on text files


data_path = f"{os.path.dirname(os.path.abspath(__file__))}/data"
data_file_fmt = (
    data_path + "/circle_on_globe_clat-{clat:02}_rad-{rad}_delta-{delta}_{method}.py"
)


def create_test_class(name, setup):
    def method_setUp(s):
        infile = s.data_file_fmt.format(**s.setup)
        mod = import_module(infile)
        for var in [
            "clon",
            "clat",
            "rad_km",
            "area_km2",
            "nlat",
            "nlon",
            "lat1d",
            "lon1d",
            "lat2d",
            "lon2d",
            "mask",
        ]:
            setattr(s, var, getattr(mod, var))
        s.create_feature()

    def method_test_area(s):
        res = s.comp_feature_area()
        sol = s.area_km2
        s.eval_test(res, sol)

    attributes = {
        "data_file_fmt": data_file_fmt,
        "setup": setup,
    }
    methods = {"setUp": method_setUp, "test_area": method_test_area}

    bases = (Test_Base,)
    dict_ = {**methods, **attributes}

    return type(name, bases, dict_)


clats = np.arange(10) * 10
rads = [800]
deltas = [0.5, 0.1, 0.05]
methods = ["dyntools", "pyproj"]

cls_name_fmt = "Test_{method}_d{delta_str}_lat{clat:02}_r{rad}"

for clat, rad, delta, method in itertools.product(clats, rads, deltas, methods):

    # Define test setup
    setup = dict(clat=clat, rad=rad, delta=delta, method=method)

    # Skip setup if no infile exists
    infile = data_file_fmt.format(path=data_path, **setup)
    if not os.path.isfile(infile):
        continue

    # Define test class name
    delta_str = "{:4.2f}".format(delta).replace(".", "p")
    cls_name = cls_name_fmt.format(delta_str=delta_str, **setup).replace(
        "_dyntools_", "_dyntls_"
    )

    # Create test class and add it to current module
    globals()[cls_name] = create_test_class(cls_name, setup)


if __name__ == "__main__":
    import logging as log

    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
    log.getLogger().setLevel(log.DEBUG)
    unittest.main()
