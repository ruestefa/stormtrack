# -*- coding: utf-8 -*-
"""
Top-level package for StormTrack.
"""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.1.0"

# SR_DEV < Shorthand to embed IPython shell
try:
    import IPython  # isort:skip
except ImportError:
    ipy = None
else:
    ipy = IPython.terminal.embed.embed
# SR_DEV >
