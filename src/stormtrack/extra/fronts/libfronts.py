# Local
from ._libfronts import abs2d as _abs2d
from ._libfronts import grad as _grad
from ._libfronts import pottemp
from ._libfronts import clustering
from ._libfronts import equpot
from ._libfronts import filt2d
from ._libfronts import norm2d
from ._libfronts import scal2d as _scal2d


__all__ = [
    "pottemp",
    "equpot",
    "filt2d",
    "grad",
    "abs2d",
    "norm2d",
    "scal2d",
    "clustering",
]


def grad(gradx, grady, fld, vert, xmin, ymin, dx, dy, stride, nan):
    """Wrapper swapping gradx and grady, otherwise wrong result (?!?!?)."""
    _grad(grady, gradx, fld, vert, xmin, ymin, dx, dy, stride, nan)


def abs2d(gradx, grady, absgrad, nan):
    """Wrapper to restore original subroutine-like interface."""
    absgrad_ = _abs2d(gradx, grady, nan)
    assert list(absgrad.shape) + [1] == list(absgrad_.shape)
    absgrad[:, :] = absgrad_[:, :, 0]


def scal2d(xfld1, yfld1, xfld2, yfld2, ofld, nan):
    """Wrapper to restore original subroutine-like interface."""
    ofld_ = _scal2d(xfld1, yfld1, xfld2, yfld2, nan)
    assert list(ofld.shape) + [1] == list(ofld_.shape)
    ofld[:, :] = ofld_[:, :, 0]
