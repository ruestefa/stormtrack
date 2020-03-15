#!/usr/bin/env python3

# Standard library
import logging as log
import sys
import time
from collections import OrderedDict


__all__ = []


_TIMERS = {}


def create_timer(name):
    if name in _TIMERS:
        err = "Timer with name {n} already exists!".format(n=name)
        raise Exception(err)
    _TIMERS[name] = Timer()
    return _TIMERS[name]


def get_timer(name):
    try:
        return _TIMERS[name]
    except KeyError:
        err = "No writer with name {n}!".format(n=name)
        raise Exception(err)


def delete_timer(name):
    try:
        del _TIMERS[name]
    except KeyError:
        err = "No writer with name {n}!".format(n=name)
        raise Exception(err)


class Timer:
    def __init__(self, level=0):
        self._timers = OrderedDict()
        self._subtimers = OrderedDict()
        self._start = OrderedDict()
        self._timers["total"] = 0.0
        self._level = level

    def add_timer(self, timer, reset_existing=False):
        try:
            self._check_timer_not_existing(timer)
        except TimerExistingError:
            if not reset_existing:
                raise
        self._timers[timer] = 0.0

    def add_timers(self, timers, reset_existing=False):
        for timer in timers:
            self.add_timer(timer, reset_existing=reset_existing)

    def add_subtimer(self, timer, subtimer, reset_existing=False):
        try:
            self._check_subtimer_not_existing(timer, subtimer)
        except TimerExistingError:
            if not reset_existing:
                raise
        else:
            if timer not in self._subtimers:
                self._subtimers[timer] = OrderedDict()
            self._subtimers[timer][subtimer] = Timer(level=self._level + 1)
        self._subtimers[timer][subtimer].reset_timers()
        return self._subtimers[timer][subtimer]

    def add_subtimers(self, timer, subtimers, reset_existing=False):
        for subtimer in subtimers:
            self.add_subtimer(timer, subtimer, reset_existing=reset_existing)

    def get_subtimer(self, timer, subtimer):
        self._check_subtimer_existing(timer, subtimer)
        return self._subtimers[timer][subtimer]

    def get_subtimers(self, timer, subtimers=None):
        if subtimers is None:
            if timer in self._subtimers:
                subtimers = self._subtimers[timer]
            else:
                subtimers = []
        return [self.get_subtimer(timer, subtimer) for subtimer in subtimers]

    def reset_timers(self, timers=None, skip_total=False):
        if timers is None:
            timers = self._timers
        for timer in timers:
            if timer == "total" and skip_total:
                continue
            self.reset_timer(timer)

    def reset_timer(self, timer):
        for subtimer in self.get_subtimers(timer):
            subtimer.reset_timers()
        self._timers[timer] = 0.0

    def start(self, timer):
        log.debug("TIMER START {n} ({t:.4f})".format(n=timer, t=self._timers[timer]))
        self._check_timer_existing(timer)
        self._check_timer_not_running(timer)
        self._start[timer] = time.clock()

    def end(self, timer):
        self._check_timer_existing(timer)
        self._check_timer_running(timer)
        start = self._start.pop(timer)
        end = time.clock()
        incr = end - start
        log.debug(
            "TIMER END {n} ({t:.4f}+{i:.4f})".format(
                n=timer, t=self._timers[timer], i=incr
            )
        )
        self._increment(timer, incr)

    def _increment(self, timer, incr):
        self._timers[timer] += incr
        self._timers["total"] += incr

    def write(self, filename, **kwargs):
        with open(filename, "w") as f:
            if "minimal" not in kwargs:
                kwargs["minimal"] = True
            if "o" in kwargs:
                del kwargs["o"]
            self.print_(o=f, **kwargs)

    def print_(
        self,
        minimal=False,
        o=sys.stdout,
        time_tot=None,
        hlines=True,
        hlines_sub=True,
        vlines=True,
        header=True,
        nested=False,
        root_level=None,
        len_name_max=0,
        len_time_max=0,
    ):

        if root_level is None:
            root_level = self._level
        level = self._level - root_level

        if minimal:
            vlines = False
            hlines = False
            header = False
        if not hlines:
            hlines_sub = False
        if level > 0:
            hlines = False
            header = False

        if time_tot is None:
            time_tot = self._timers["total"]

        len_frac = 3
        len_vl = 1 if vlines else 0
        max_level = self._get_max_level(root_level)

        len_name = self._get_len_name(
            nested=nested, len_max=len_name_max, max_level=max_level, level=level
        )
        len_time = self._get_len_time(len_max=len_time_max, len_frac=len_frac)
        len_pct = 3 + 1 + len_frac
        len_tot = (
            len_vl
            + 1
            + len_name
            + 1
            + len_vl
            + 1
            + len_time
            + 1
            + len_vl
            + 1
            + len_pct
            + 1
            + len_vl
            + 1
            + len_pct
            + 1
            + len_vl
        )

        vline = "|" if vlines else ""
        hline = "-" * len_tot + "\n"
        hline_nesting = (
            vline
            + "-" * (2 + len_name)
            + vline
            + "-" * (2 + len_time)
            + vline
            + "-" * (2 + len_pct)
            + vline
            + "-" * (2 + len_pct)
            + vline
            + "\n"
        )

        str_header = (
            (
                "{l} {{n:<{n}}} "
                "{l} {{t:>{t}}} "
                "{l} {{pr:>{p}}} "
                "{l} {{pt:>{p}}} {l}\n"
            )
            .format(l=vline, n=len_name, t=len_time, p=len_pct)
            .format(n="name", t="time/s", pr="pct_r/%", pt="pct_t/%")
        )

        if hlines:
            o.write(hline)
        if header:
            o.write(str_header)
        if hlines:
            o.write(hline)

        template = (
            "{l} {{n:<{n}}} "
            "{l} {{t:{t}.{f}f}} "
            "{l} {{pr:{p}.{f}f}} "
            "{l} {{pt:{p}.{f}f}} {l}\n"
        ).format(l=vline, n=len_name, f=len_frac, t=len_time, p=len_pct)

        for i, (name, time) in enumerate(self._timers.items()):
            try:
                pctr = 100 * time / self._timers["total"]
            except ZeroDivisionError:
                pctr = 0
            try:
                pctt = 100 * time / time_tot
            except ZeroDivisionError:
                pctt = 0
            ind = "  " * level if nested else ""
            o.write(template.format(n=ind + name, t=time, pr=pctr, pt=pctt))
            if name in self._subtimers:
                if hlines_sub:
                    o.write(hline_nesting)
                for subtimer in self._subtimers[name].values():
                    subtimer.print_(
                        o=o,
                        time_tot=time_tot,
                        header=header,
                        vlines=vlines,
                        hlines=hlines,
                        hlines_sub=hlines_sub,
                        nested=True,
                        root_level=root_level,
                        len_name_max=len_name,
                        len_time_max=len_time,
                    )
                if hlines_sub and i + 1 < len(self._timers):
                    o.write(hline_nesting)

        if hlines:
            o.write(hline)

    def _get_max_level(self, root_level):
        max_level = self._level - root_level
        for subtimers in self._subtimers.values():
            for subtimer in subtimers.values():
                max_level = max([max_level, subtimer._get_max_level(root_level)])
        return max_level

    def _get_len_name(self, *, nested, len_max, max_level, level):
        len_ind = 2 * max_level
        len_ = len_ind + max([len(name) for name in self._timers])
        len_max = max(len_, len_max)
        len_sub = []
        for subtimers in self._subtimers.values():
            for subtimer in subtimers.values():
                len_sub.append(
                    subtimer._get_len_name(
                        nested=nested,
                        len_max=len_max,
                        max_level=max_level,
                        level=level + 1,
                    )
                )
                len_sub
        return max([len_max] + len_sub)

    def _get_len_time(self, len_max=0, len_frac=0):
        len_ = len(str(int(max(self._timers.values())))) + 1 + len_frac
        len_max = max([len_, len_max])
        len_sub = [
            subtimer._get_len_time(len_max=len_max)
            for subtimers in self._subtimers.values()
            for subtimer in subtimers.values()
        ]
        return max([len_max] + len_sub)

    def _check_timer_existing(self, timer):
        if timer not in self._timers:
            err = ("Timer '{t}' not registered! " "Registered timers: {l}").format(
                t=timer, l=", ".join(self._timers)
            )
            raise TimerMissingError(err)

    def _check_timer_not_existing(self, timer):
        if timer in self._timers:
            err = ("Timer '{t}' already registered! " "Registered timers: {l}").format(
                t=timer, l=", ".join(self._timers)
            )
            raise TimerExistingError(err)

    def _check_subtimer_existing(self, timer, subtimer):
        if timer not in self._subtimers or subtimer not in self._subtimers[timer]:
            err = "Subtimer '{t}/{s}' not registered!".format(t=timer, s=subtimer)
            raise TimerMissingError(err)

    def _check_subtimer_not_existing(self, timer, subtimer):
        if timer in self._subtimers and subtimer in self._subtimers[timer]:
            err = "Subtimer '{t}/{s}' already registered!".format(t=timer, s=subtimer)
            raise TimerExistingError(err)

    def _check_timer_running(self, timer):
        if timer not in self._start:
            err = ("Timer '{t}' not running! " "Running timers: {l}").format(
                t=timer, l=", ".join(self._start)
            )
            raise TimerStoppedError(err)

    def _check_timer_not_running(self, timer):
        if timer in self._start:
            err = ("Timer '{t}' already running! " "Running timers: {l}").format(
                t=timer, l=", ".join(self._start)
            )
            raise TimerRunningError(err)


class TimerExistingError(Exception):
    pass


class TimerMissingError(Exception):
    pass


class TimerRunningError(Exception):
    pass


class TimerStoppedError(Exception):
    pass


if __name__ == "__main__":
    pass
