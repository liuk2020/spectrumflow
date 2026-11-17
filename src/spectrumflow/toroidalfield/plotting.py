"""Matplotlib views of toroidal scalar fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .field import ToroidalField, _integer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.contour import QuadContourSet


def plot_field(
    field: ToroidalField,
    *,
    ax: Axes | None = None,
    ntheta: int = 129,
    nzeta: int = 129,
    one_period: bool = True,
    filled: bool = True,
    **kwargs: Any,
) -> tuple[Axes, QuadContourSet]:
    """Plot theta versus zeta (radians); return (axes, contour artist).

    kwargs are passed to ax.contourf/contour. The display grid includes both
    endpoints. Add a labeled colorbar with ax.figure.colorbar(artist, ax=ax).
    No show(), save, colorbar or global style changes are performed here.
    """
    import matplotlib.pyplot as plt

    ntheta, nzeta = _integer(ntheta, "ntheta", 2), _integer(nzeta, "nzeta", 2)
    theta = np.linspace(0, 2 * np.pi, ntheta)
    zeta = np.linspace(0, 2 * np.pi / (field.nfp if one_period else 1), nzeta)
    values = field(theta[:, None], zeta[None, :])
    if ax is None:
        _, ax = plt.subplots()
    draw = ax.contourf if filled else ax.contour
    artist = draw(zeta, theta, values, **kwargs)
    ax.set_xlabel(r"$\zeta$ [rad]")
    ax.set_ylabel(r"$\theta$ [rad]")
    return ax, artist
