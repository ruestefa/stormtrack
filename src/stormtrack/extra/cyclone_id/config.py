#!/usr/bin/env python

# Standard library
import argparse
import ast
import json
import logging as log
import os
import sys
from configparser import SafeConfigParser


__all__ = []


# SR_TODO: Make this more compatible with optparse (cf. options_internal)
def options():
    return {
        "GENERAL": {
            # Input
            "infile-path": {
                "description": None,
                "type": "path",
                "default": ".",
                "metavar": "PATH",
            },
            "infile-list": {
                "description": None,
                "type": "list/str",
                "default": None,
                "metavar": "LIST",
            },
            "input-field-name": {
                "description": None,
                "type": "str",
                "default": "PMSL",
                "metavar": "NAME",
            },
            "input-field-level": {
                "description": "level (in case of 3D input field)",
                "type": "int",
                "default": None,
                "metavar": "LVL",
            },
            "topofile-path": {
                "description": None,
                "type": "path",
                "default": ".",
                "metavar": "PATH",
            },
            "topofile": {
                "description": None,
                "type": "str",
                "default": "LMCONSTANTS",
                "metavar": "NAME",
            },
            "topo-field-name": {
                "description": None,
                "type": "str",
                "default": "HSURF",
                "metavar": "NAME",
            },
            # Output: data
            "output-path": {
                "description": None,
                "type": "path",
                "default": "./output",
                "metavar": "PATH",
            },
            "save-contour-paths-binary": {
                "description": "Save the paths of contours to a binary file.",
                "type": "bool",
                "default": True,
                "metavar": "BOOL",
            },
            # Output: plots
            "make-plots": {
                "description": None,
                "type": "bool",
                "default": True,
                "metavar": "BOOL",
            },
            "image-format": {
                "description": None,
                "type": "str",
                "default": "PNG",
                "metavar": "TYPE",
            },
            "plots": {
                "description": "List of plots to write.",
                "type": "list/str",
                "default": "depressions, cyclones",
                "metavar": "NAMES",
            },
            # Output: test data
            "write-test-input-file-identify": {
                "description": (
                    "Write the input fields for the identification "
                    "to a numpy binary file (*.npz)."
                ),
                "type": "str",
                "default": "",
                "metavar": "NAME",
            },
            "write-test-output-file-identify": {
                "description": (
                    "Write the output of the identification "
                    "(Depressions and Cyclones) to a JSON file."
                ),
                "type": "str",
                "default": "",
                "metavar": "NAME",
            },
            # Various
            # SR_TODO: Set level of timings (max depth of nested timings)
            "timings": {
                "description": "Measure local timings.",
                "type": "bool",
                "default": True,
                "metavar": "BOOL",
            },
        },
        "IDENTIFY": {
            "timings-identify": {
                "description": (
                    "Measure local timings in identification "
                    "(only if global timings activated)."
                ),
                "type": "bool",
                "default": True,
                "metavar": "BOOL",
            },
            "ids-datetime": {
                "description": "use datetime-based object IDs",
                "type": "bool",
                "default": True,
                "metavar": "BOOL",
            },
            "ids-datetime-digits": {
                "description": "for datetime-based IDs, no. digits of ID",
                "type": "int",
                "default": 5,
                "metavar": "N",
            },
            "contour-length-min": {
                "description": "minimal contour length [km]",
                "type": "float",
                "default": -1.0,
                "metavar": "LEN",
            },
            "contour-length-max": {
                "description": "maximal contour length [km]",
                "type": "float",
                "default": -1.0,
                "metavar": "LEN",
            },
            "contour-interval": {
                "description": "sampling interval for contours (hPa or gpdm)",
                "type": "float",
                "default": 0.5,
                "metavar": "DP",
            },
            "contour-level-start": {
                "description": "start level for contours (hPa or gpdm)",
                "type": "float",
                "default": 920,
                "metavar": "P",
            },
            "contour-level-end": {
                "description": "end level for contours (hPa or gpdm)",
                "type": "float",
                "default": 1050,
                "metavar": "P",
            },
            "depression-min-contours": {
                "description": "minimal number of contours for Depressions.",
                "type": "int",
                "default": 1,
                "metavar": "N",
            },
            "smoothing-sigma": {
                "description": "smoothing factor (sigma value of Gaussian filter)",
                "type": "float",
                "default": 7.0,
                "metavar": "SIG",
            },
            "topo-cutoff-level": {
                "description": "cut-off level for topography [km|hPa]",
                "type": "float",
                "default": 1500.0,
                "metavar": "LVL",
            },
            "extrema-identification-size": {
                "description": (
                    "side length of the neighbourhood for finding "
                    "local minima/maxima"
                ),
                "type": "int",
                "default": 9,
                "metavar": "N",
            },
            "min-cyclone-depth": {
                "description": (
                    "minimal cyclone depth measured from the outer-"
                    "most contour to the deepest minimum [hPa]"
                ),
                "type": "float",
                "default": 1.0,
                "metavar": "DP",
            },
            "max-minima-per-cyclone": {
                "description": "max. no. minima per cyclone",
                "type": "int",
                "default": 3,
                "metavar": "N",
            },
            "size-boundary-zone": {
                "description": (
                    "min. distance in grid points of minima/maxima "
                    "from domain boundary"
                ),
                "type": "int",
                "default": 5,
                "metavar": "N",
            },
            "force-contours-closed": {
                "description": (
                    "if True, only closed contours (never leave "
                    "the domain) are considered for cyclones"
                ),
                "type": "bool",
                "default": False,
                "metavar": "BOOL",
            },
            "bcc-fraction": {
                "description": (
                    "max. fraction of boundary-crossing contours"
                    " (BCCs) per feature (Depression/Cyclone) [0..1]"
                ),
                "type": "float",
                "default": 0.3,
                "metavar": "FRAC",
            },
            "save-slp-contours": {
                "description": (
                    "save SLP contours to file after pre-processing "
                    "and exit; 'slp-contours-file' must be set"
                ),
                "type": "bool",
                "default": False,
                "metavar": "BOOL",
            },
            "read-slp-contours": {
                "description": (
                    "read SLP contours from file instead of computing "
                    "them; 'slp-contours-file' must be set"
                ),
                "type": "bool",
                "default": False,
                "metavar": "BOOL",
            },
            "slp-contours-file": {
                "description": "file to/from which SLP contours are written/read",
                "type": "str",
                "default": "contours.npz",
                "metavar": "FILE",
            },
            "slp-contours-file-path": {
                "description": "directory where 'slp-contours-file' resides",
                "type": "str",
                "default": ".",
                "metavar": "PATH",
            },
            "save-slp-extrema": {
                "description": (
                    "save SLP extrema to file after pre-processing "
                    "and exit; 'slp-extrema-file' must be set"
                ),
                "type": "bool",
                "default": False,
                "metavar": "BOOL",
            },
            "read-slp-extrema": {
                "description": (
                    "read SLP extrema from file instead of computing "
                    "them; 'slp-extrema-file' must be set"
                ),
                "type": "bool",
                "default": False,
                "metavar": "BOOL",
            },
            "slp-extrema-file": {
                "description": "file to/from which SLP extrema are written/read",
                "type": "str",
                "default": "extrema.npz",
                "metavar": "FILE",
            },
            "slp-extrema-file-path": {
                "description": "directory where 'slp-extrema-file' resides",
                "type": "str",
                "default": ".",
                "metavar": "PATH",
            },
        },
    }


def get_config_args(args=sys.argv[1:]):
    """Read configuration from command line arguments.

    Optional arguments:
     - args: Command line arguments. Defaults to sys.argv.

    """
    parser = argparse.ArgumentParser(usage="%(prog)s [OPTIONS]")

    # SR_TODO: Merge this into options()
    options_internal = {
        ("v", "verbose"): {
            "action": "store_true",
            "default": True,
            "help": "show verbose output",
        },
        ("d", "debug"): {
            "action": "store_true",
            "default": False,
            "help": "show debug output",
        },
        (None, "print-conf"): {
            "action": "store_true",
            "default": False,
            "help": "print config in conf dict-format",
        },
        (None, "dump-conf"): {
            "action": "store_true",
            "default": False,
            "help": "print config in conf dict-format and exit",
        },
        (None, "dump-conf-default"): {
            "action": "store_true",
            "default": False,
            "help": "print default config in conf dict-format and exit",
        },
        (None, "dump-ini"): {
            "action": "store_true",
            "default": False,
            "help": "print config in INI file-format and exit",
        },
        (None, "dump-ini-default"): {
            "action": "store_true",
            "default": False,
            "help": "print default config in INI file-format and exit",
        },
    }

    # Define "interal options"
    for (shortname, longname), opt in options_internal.items():
        longflag = "--" + longname
        if shortname is None:
            parser.add_argument(longflag, **opt)
        else:
            shortflag = "-" + shortname if shortname is not None else None
            parser.add_argument(longflag, shortflag, **opt)

    # Define options
    for secname, sec in options().items():
        for optname, opt in sorted(sec.items()):
            log.debug("add option {n}".format(n=optname))
            parser.add_argument(
                "--" + optname,
                type=option_conversion_function(opt["type"]),
                # default=opt["default"],
                help=opt["description"],
                metavar=opt["metavar"],
            )

    # Parse arguments
    # Note: Dashed (-) in arguments are converted to underscores (_).
    args_parsed = parser.parse_args(args)

    # Construct conf dict containing options passed as argument
    conf = {}
    for secname, sec in options().items():
        for optname, opt in sec.items():
            flagname = optname.replace("-", "_")
            if not flagname in args_parsed:
                continue
            val = getattr(args_parsed, flagname)
            if val is None:
                continue
            if not secname in conf:
                conf[secname] = {}
            conf[secname][optname] = val

    # Add "internal options"
    conf["INTERNAL"] = {}
    for optname in (longname for shortname, longname in options_internal):
        flagname = optname.replace("-", "_")
        if flagname in args_parsed:
            conf["INTERNAL"][optname] = getattr(args_parsed, flagname)
        else:
            conf["INTERNAL"][optname] = False  # SR_TMP

    return conf


def merge_configs(conf_list):
    """Merge multiple config dicts into one.

    The dicts to merge are passed as a list, ordered by descending precedence.
    This means that an option present in multiple dicts will get the value
    it has in the dict that come latest in the list.

    The merged dict is returned.

    Arguments:
     - conf_list: List of conf dicts to be merged, ordered by descending
    precedence.
    """
    conf_merged = {}
    for conf in conf_list:
        for secname, sec in conf.items():
            if not secname in conf_merged:
                conf_merged[secname] = sec
            else:
                conf_merged[secname].update(sec)
    return conf_merged


def get_config_default():
    """Construct the conf dictionary with the default values."""
    conf = {}
    for sec, vars in options().items():
        conf[sec] = {k: v["default"] for k, v in vars.items()}
    return conf


def find_inifile(dir=".", priorities=None, priorities_only=True):
    """Find the most suitable INI file in a given directory.

    The default directory is the current working directory.

    By default, the first INI file found is returned.

    Optionally, a list of priorities can be passed, i.e. preferred file names.
    File earlier in this list are preferred over those coming later.

    By default, if a list of priorities is passed, INI files not in this list
    are not considered.

    Return the name incl. path of the most suitable INI file found.
    If none is found, return None.
    """
    nosuffix = lambda s: os.path.splitext(s)
    withsuffix = lambda s: s if s.endswith(".ini") else s + ".ini"
    candidates = [nosuffix(f) for f in os.listdir(dir) if f.endswith(".ini")]
    if priorities is not None:
        if any([nosuffix(f) in candidates for f in priorities]):
            inifile = [f for f in priorities if nosuffix(f) in candidates][0]
            return withsuffix(inifile)
        elif priorities_only:
            return None
    try:
        inifile = candidates[0]
    except IndexError:
        return None
    else:
        return withsuffix(inifile)


def get_config_inifile(filename):
    """Read configuration from INI file and return a config dict.

    Arguments:
     - filename: Name of INI file.
    """
    # SR_TODO: default config
    parser = SafeConfigParser()
    parser.optionxform = str  # to preserve case
    parser.read(filename)
    log.debug("read configuration from file '{n}'".format(n=filename))
    conf = {}
    for sec in parser.sections():
        if sec not in conf:
            conf[sec] = {}
        for opt, val in parser.items(sec):
            try:
                conf[sec][opt] = process_option(sec, opt, val)
            except Exception as e:
                err = "error processing option {}/{}={}:\n{}({})".format(
                    sec, opt, val, e.__class__.__name__, e
                )
                raise Exception(err) from e
    return conf


def process_option(sec, name, val):
    """Check validity of an option and convert it to the correct type.

    The conversion of the option value is done using the conversion function
    specified by the option type.

    Returns the converted option value.

    Arguments:
     - sec: Section name.
     - name: Option name.
     - val: Option value.
    """
    out = None
    if sec in options() and name in options()[sec]:
        fct = option_conversion_function(options()[sec][name]["type"])
        out = fct(val)
        log.debug(
            "read option from section '{s}'\t: {n} = {v}".format(s=sec, n=name, v=val)
        )
    else:
        try:
            # auto-convert str to int etc.
            out = ast.literal_eval(val)
        except ValueError:
            pass
        log.warning(
            "unknown option from section '{s}'\t: {n} = {v}".format(
                s=sec, n=name, v=val
            )
        )

    return out


def option_conversion_function(var_type):
    """..."""
    return {
        "float": procFloat,
        "str": procStr,
        "int": procInt,
        "bool": procBool,
        "path": procPath,
        "list/str": procListStr,
    }[var_type]


def procStr(val):
    return None if val == "None" else str(val)


def procInt(val):
    return None if val == "None" else int(val)


def procFloat(val):
    return None if val == "None" else float(val)


def procBool(val):
    return True if val in ["True", "true"] else False


def procPath(val):
    return os.path.abspath(val)


def procListStr(val):
    return [procStr(s) for s in val.split(",")]


def dump_config_compact(conf):
    """Print the configuration in compact form."""
    for secname, sec in sorted(conf.items()):
        print("{n}:".format(n=secname.upper()))
        for optname, opt in sec.items():
            print("{n}: {v}".format(n=optname, v=opt))
        print("")


def dump_config_ini(conf):
    """Print the configuration in INI file-format."""
    for sec in options():

        # Skip section with leadint underscore (internal config)
        if sec[0] == "_":
            continue

        # Section header
        sys.stdout.write("\n[{sec}]\n\n".format(sec=sec))

        descr_prev = False
        for name, var in sorted(options()[sec].items()):
            descr = var["description"] is not None

            # Emty line if previous entry had a description
            if descr_prev:
                sys.stdout.write("\n")

            # Empty line if both this and previous entry have a description
            if descr_prev and descr:
                sys.stdout.write("\n")

            # Description comment
            if descr:
                sys.stdout.write("# {d}\n".format(d=var["description"]))

            # Variable entry
            sys.stdout.write("{n} = {v}\n".format(n=name, v=conf[sec][name]))

            descr_prev = descr is not None


if __name__ == "__main__":
    pass
