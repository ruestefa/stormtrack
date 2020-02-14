#!/usr/bin/env python3

from __future__ import print_function

import argparse
import datetime
import os
import re
import sys
import warnings

#==============================================================================

def ipython(__globals__, __locals__, __msg__=None, __err__=66):
    """Drop into an iPython shell with all global and local variables.

    To pass the global and local variables, call with globals() and locals()
    as arguments, respectively. Notice how both functions must called in the
    call to ipython.

    To exit the program after leaving the iPython shell, pass an integer,
    which will be returned as the error code.

    To display a message when dropping into the iPython shell, pass a
    string after the error code.

    Examples
    --------
    >>> ipython(globals(), locals(), "I'm here!")

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        import IPython
        print('\n----------\n')
        globals().update(__globals__)
        locals().update(__locals__)
        if __msg__ is not None:
            print("\n{l}\n{m}\n{l}\n".format(l="*" * 60, m=__msg__))
        IPython.embed(display_banner=None)
        if __err__ is not None:
            sys.exit(__err__)

def import_module(filename, setenv=None):
    """Import a Python module from file.

    Parameters
    ----------
        filename: str
            Input file.

        setenv: dict, optional
            Environment variables set before file import, providing a way
            to pass variables to the input files that can be used therein.
            Note that the values must be convertible to str.
    """
    from importlib.machinery import SourceFileLoader

    if setenv is not None:
        for key, val in setenv.items():
            if isinstance(val, bool):
                # Handle "bool(str(False)) == True" issue
                val = int(val)
            os.environ[key] = str(val)

    try:
        fi = SourceFileLoader("module.name", filename)
        return fi.load_module()
    except Exception as e:
        err = "Cannot import file as module: {}".format(filename)
        raise Exception(err)

def extract_args(kwas, prefix):
    """Extract arguments from keyword argument dict by prefix.

    Parameters
    ----------
        kwas: dict
            Dictionary with string keys.

        prefix: str
            Prefix to select keys (must be followed by '__').

    Example
    -------
    >>> parser = argparse.ArgumentParser(...)
    >>> parser.add_argument(..., dest="input__foo")
    >>> parser.add_argument(..., dest="output__bar")
    >>> kwas = vars(parser.parse_args())
    >>> conf_input = extract_args(kwas, "input")
    >>> conf_output = extract_args(kwas, "output")
    >>> conf_input, conf_output
        ({"foo": ...}, {"bar": ...})

    """
    conf = {}
    prefix = "{}__".format(prefix)
    for key, val in kwas.copy().items():
        if key.startswith(prefix):
            del kwas[key]
            conf[key.replace(prefix, '')] = val
    return conf

def print_args(args, *, skip=None, nt=40):
    """Print input arguments obtained by argparse (or any dict).

    Example
    -------
    >>> parser = argparse.ArgumentParser(...)
        ...
    >>> args = parser.parse_args() # or vars(parser.parse_args())
    >>> print_args(args)

    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if nt > 0:
        print("=" * nt)
    nn = 0 if not args else max([len(k) for k in args])
    for name, arg in sorted(args.items()):
        if skip is not None and name in skip:
            arg = "SKIP"
        elif isinstance(arg, (list, tuple)):
            arg = ", ".join([str(a) for a in arg])
        print("{{:{}}} : {{}}".format(nn).format(name.upper(), arg))
    if nt > 0:
        print("=" * nt)

class TimestepGenerator:

    def __init__(self, start, end, stride, format=None, mode="int",
            n_before=0, n_after=0):
        """Create timesteps from range arguments (start, end, stride).

        Parameters
        ----------
            start: int or datetime.datetime
                First timestep.

            end: int or datetime.datetime
                Last timestep (inclusive).

            stride: int or datetime.timedelta
                Timestep interval.

            format: str, optional
                Datetime format string; will be derived from first timestep
                if not passed (default: None).

            mode: str, optional
                Timestep mode ('int', 'str', 'datetime) (default: 'int').
        """

        # Check arguments
        modes = ["int", "str", "datetime"]
        if mode not in modes:
            raise ValueError("Invalid mode {}; must be one of: {}".format(
                    mode, ", ".join(modes)))

        if format is None:
            # Determine format
            sts = str(start)
            if len(sts) == 12:
                format = "%Y%m%d%H%M"
            elif len(sts) == 10:
                format = "%Y%m%d%H"
            elif len(sts) == 8:
                format = "%Y%m%d"
            elif len(sts) == 6:
                format = "%Y%m"
            try:
                datetime.datetime.strptime(sts, format)
            except Exception as e:
                err = "cannot derive timestep format from {}: {}".format(
                        sts, e)
                raise ValueError(err)

        # Determine frequency
        if all(i in format for i in ["%Y", "%m", "%d", "%H", "%M"]):
            frequency = "minute"
        elif all(i in format for i in ["%Y", "%m", "%d", "%H"]):
            frequency = "hourly"
        elif all(i in format for i in ["%Y", "%m", "%d"]):
            frequency = "daily"
        elif all(i in format for i in ["%Y", "%m"]):
            frequency = "monthly"
            if n_before != 0 or n_after != 0:
                raise NotImplementedError("n_before/n_after for monthly")
        else:
            raise ValueError("unknown format: "+format)
        self.frequency = frequency

        # Initialize parameters
        if isinstance(stride, datetime.timedelta):
            self.stride = stride
        else:
            if frequency == "minute":
                self.stride = datetime.timedelta(minutes=stride)
            elif frequency == "hourly":
                self.stride = datetime.timedelta(hours=stride)
            elif frequency == "daily":
                self.stride = datetime.timedelta(days=stride)
            elif frequency == "monthly":
                # Note: there's no 'months' timedelta
                self.stride = stride
        if isinstance(start, datetime.datetime):
            self.start = start
        else:
            self.start = datetime.datetime.strptime(str(start), format)
        if n_before != 0:
            self.start -= n_before*self.stride
        if isinstance(end, datetime.datetime):
            self.end = end
        else:
            self.end = datetime.datetime.strptime(str(end), format)
        if n_after != 0:
            self.end += n_after*self.stride
        self.format = format
        self.mode = mode

        # Another argument check
        if self.end < self.start:
            raise ValueError("{} < {}".format(self.end, self.start))

    def __iter__(self):
        ts = self.start
        while ts <= self.end:

            # Yield current timestep
            if self.mode == "datetime":
                yield ts
            elif self.mode == "str":
                yield ts.strftime(self.format)
            elif self.mode == "int":
                yield int(ts.strftime(self.format))

            # Increment timestep
            if self.frequency != "monthly":
                ts += self.stride
            else:
                ts = self._increment_monthly(ts)

    def _increment_monthly(self, ts):
        """Increment timestep by N months."""
        sts_old = ts.strftime(self.format)
        yyyy = int(sts_old[:4])
        mm = int(sts_old[4:6])
        mm = mm + self.stride
        if mm > 12:
            yyyy += 1
            mm = mm%12
        sts_new = "{:04}{:02}{}".format(yyyy, mm, sts_old[6:])
        ts_new = datetime.datetime.strptime(sts_new, self.format)
        return ts_new

    def __len__(self):
        return len(list(iter(self)))

    def __repr__(self):
        return "TimestepsGenerator({}, {}, {})".format(
                self.start, self.end, self.stride)

    def tolist(self):
        """Return timesteps as list."""
        return list(iter(self))

    @classmethod
    def from_args(cls, list_, ranges, **kwas):
        """Create list of timesteps from either a list, or range arguments.

        Parameters
        ----------
            list_: list
                A list of timesteps; must be None if ranges is not None.

            ranges: list
                A list of (start, end, stride) tuples; must be None if list_
                is not None.

        This method useful, for instance, if a program accepts timesteps as
        either an explicit list, or as range arguments, but not both.
        """
        if list_ and ranges:
            err = "pass timesteps as either list or range, not both"
            raise ValueError(err)

        elif list_:
            # Already a list of timesteps; nothing to do!
            return [ts for ts in list_]

        elif ranges:
            # Create timesteps from range arguments
            timesteps = []
            for start, end, stride in ranges:
                timesteps_i = cls(start, end, stride, **kwas)
                timesteps.extend(timesteps_i)
            return sorted(set(timesteps))

        else:
            err = "no timesteps; pass as either list or range"
            raise ValueError(err)

    @classmethod
    def from_months(cls, yyyymms):
        tss_all = []
        for yyyymm in sorted(yyyymms):
            yyyy, mm = str(yyyymm)[:4], str(yyyymm)[4:]
            ts0 = int("{}0100".format(yyyymm))
            ts1 = int("{:4}{:02}0100".format(
                    int(yyyy)+1 if int(mm) == 12 else int(yyyy),
                    1 if int(mm) == 12 else int(mm)+1,
                ))
            tss = cls.from_args(None, [(ts0, ts1, 1)])[:-1]
            tss_all.extend(tss)
        return tss_all

class TimestepStringFormatter:

    def __init__(self, format_string, return_timesteps=True, freq=None):
        """Create strings by inserting timesteps into a format string.

        Timesteps must have the format YYYYMMDDHH, e.g., 2007102321
        for 21:00, Oct 23, 2007. They are passed to the run method.

        Parameters
        ----------
            format_string: str
                String (optionally) containing date keys ('{YYYY}', '{MM}',
                '{DD}', '{HH}', '{NN}'), which are replaced by the respective
                datetime component (year, month, day, hour, minute).

            return_timesteps: bool, optional
                If False, the formatted strings are returned as a list,
                otherwise in a dictionary with the respective timesteps as
                keys (default: True).

            freq: str
                Enforce a certain frequency ('yearly', 'monthly', 'daily',
                'hourly', 'minute'). (default: None)

        """
        if not format_string:
            raise ValueError("empty format string")
        self._check_format_string(format_string, freq)
        self.ts_keys = ("{YYYY}", "{MM}", "{DD}", "{HH}", "{NN}")
        self._tmpl = self.protect_other_keys(format_string)
        self.return_timesteps = return_timesteps

    def _check_format_string(self, fmt, freq):
        """Check format string for forced frequency mode."""
        if freq is None:
            return

        # Get index corresponding to frequency
        inds = dict(yearly=1, monthly=2, daily=3, hourly=4, minute=5)
        try:
            ind = inds[freq]
        except KeyError:
            err = "invalid freq '{}'; must be one of {}".format(freq,
                    sorted(inds.keys()))
            raise ValueError(err)

        # Check presence of necessary keys (and absence of others)
        keys = ["{YYYY}", "{MM}", "{DD}", "{HH}", "{NN}"]
        if (not all(k in fmt for k in keys[:ind]) or
                any(k in fmt for k in keys[ind:])):
            err = "format string not {} frequency: {}".format(freq, fmt)
            raise ValueError(err)

    def protect_other_keys(self, format_string):
        """Replace non-timestep {KEY}s by {{KEY}}s to protect them."""
        for key in set(re.findall(r'{[^}]+}', format_string)):
            if key not in self.ts_keys:
                format_string = format_string.replace(key, "{"+key+"}")
        return format_string

    def run(self, timesteps, extra_keys=None):
        """Create strings from list of timesteps (see __init__ for details).

        Parameters
        ----------
            timesteps: list
                List of timesteps in format YYYYMMDD[HH[NN]].

            extra_keys: dict, optional
                Extra keys to be replaced in the string in addition to
                the timestep keys, e.g., '{"FOO": "bar"}' to replace all
                occurrences of "{FOO}" by "bar". (default: None)
        """
        self.single_file = False
        if isinstance(timesteps, int):
            self.single_file = True
            timesteps = [timesteps]

        tmpl = self._tmpl

        if extra_keys is not None:
            # Replace extra keys
            for key, val in extra_keys.items():
                while not key.startswith("{{"):
                    key = "{"+key
                while not key.endswith("}}"):
                    key = key+"}"
                if val == key:
                    continue
                while key in tmpl:
                  tmpl = tmpl.replace(key, str(val))

        # None
        if not any(key in tmpl for key in self.ts_keys):
            if self.return_timesteps:
                return {tuple(timesteps): tmpl}
            elif self.single_file:
                return tmpl
            else:
                return [tmpl]

        # Yearly
        elif not any(key in tmpl for key in self.ts_keys[1:]):
            if not all(key in tmpl for key in self.ts_keys[:1]):
                raise NotImplementedError("incomplete key set: "+tmpl)
            return self._run(timesteps, 0, 4)

        # Monthly
        elif not any(key in tmpl for key in self.ts_keys[2:]):
            if not all(key in tmpl for key in self.ts_keys[:2]):
                if self.ts_keys[0] not in tmpl:
                    # Special case: year not defined but month
                    # TODO find general solution
                    return self._run(timesteps, 4, 6)
                else:
                    raise NotImplementedError("incomplete key set: "+tmpl)
            return self._run(timesteps, 0, 6)

        # Daily
        elif not any(key in tmpl for key in self.ts_keys[3:]):
            if not all(key in tmpl for key in self.ts_keys[:3]):
                raise NotImplementedError("incomplete key set: "+tmpl)
            return self._run(timesteps, 0, 8)

        # Hourly
        elif not any(key in tmpl for key in self.ts_keys[4:]):
            if not all(key in tmpl for key in self.ts_keys[:4]):
                raise NotImplementedError("incomplete key set: "+tmpl)
            return self._run(timesteps, 0, 10)

        # Minute
        elif not any(key in tmpl for key in self.ts_keys[5:]):
            if not all(key in tmpl for key in self.ts_keys[:5]):
                raise NotImplementedError("incomplete key set: "+tmpl)
            return self._run(timesteps, 0, 12)

        # None
        if self.single_file:
            return tmpl
        return {tuple(timesteps): tmpl}

    def _run(self, timesteps, i, j):
        files_ts = {}
        for ts in timesteps:
            for tss, file in files_ts.copy().items():
                if str(ts)[i:j] == str(tss[0])[i:j]:
                    del files_ts[tss]
                    files_ts[tss + (ts,)] = self._tmpl.format(
                            YYYY = str(ts)[ 0: 4],
                            MM   = str(ts)[ 4: 6],
                            DD   = str(ts)[ 6: 8],
                            HH   = str(ts)[ 8:10],
                            NN   = str(ts)[10:12],
                        )
                    break
            else:
                files_ts[(ts,)] = self._tmpl.format(
                        YYYY = str(ts)[ 0: 4],
                        MM   = str(ts)[ 4: 6],
                        DD   = str(ts)[ 6: 8],
                        HH   = str(ts)[ 8:10],
                        NN   = str(ts)[10:12],
                    )
        if self.return_timesteps:
            return files_ts
        elif self.single_file:
            return next(iter(files_ts.values()))
        else:
            return sorted(files_ts.values())

#==============================================================================
# Custon Json encoder to write lists on single lines
#==============================================================================
# src: http://stackoverflow.com/a/25935321
import uuid
import json

class NoIndent(object):
    def __init__(self, value):
        self.value = value

class NoIndentEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwas):
        super(NoIndentEncoder, self).__init__(*args, **kwas)
        self.kwas = dict(kwas)
        del self.kwas['indent']
        self._replacement_map = {}

    def default(self, o):
        if isinstance(o, NoIndent):
            key = uuid.uuid4().hex
            self._replacement_map[key] = json.dumps(o.value, **self.kwas)
            return "@@%s@@" % (key,)
        else:
            return super(NoIndentEncoder, self).default(o)

    def encode(self, o):
        result = super(NoIndentEncoder, self).encode(o)
        for k, v in self._replacement_map.iteritems():
            result = result.replace('"@@%s@@"' % (k,), v)
        return result

#==============================================================================
