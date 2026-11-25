"""Internal Matplotlib implementation for toroidal surface sections."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .surface import Surface, _angles

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D


def _plot_cross_section(
    surface: Surface,
    phi: ArrayLike | None,
    *,
    ax: Axes | None,
    ntheta: int,
    fontsize: float,
    tick_fontsize: float | None,
    usetex: bool,
    figsize: tuple[float, float] | None,
    **kwargs: Any,
) -> tuple[Axes, list[Line2D]]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    if isinstance(ntheta, (bool, np.bool_)) or not isinstance(ntheta, Integral):
        raise TypeError("ntheta must be an integer")
    if ntheta < 3:
        raise ValueError("ntheta must be >= 3")
    phi = np.asarray([0, np.pi / surface.nfp] if phi is None else phi)
    if phi.ndim > 1 or phi.size == 0:
        raise ValueError("phi must be a scalar or nonempty 1D array")
    phi, _ = _angles(np.atleast_1d(phi), 0)
    theta = np.linspace(0, 2 * np.pi, ntheta)
    r, z = surface.cross_section(theta[:, None], phi[None, :])
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    lines = ax.plot(r, z, **kwargs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$R$")
    ax.set_ylabel(r"$Z$")
    tick_fontsize = fontsize if tick_fontsize is None else tick_fontsize
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    for axis in (ax.xaxis, ax.yaxis):
        formatter = axis.get_major_formatter()
        if isinstance(formatter, ScalarFormatter):
            formatter.set_useMathText(True)
        axis.label.set(fontsize=fontsize, usetex=usetex)
        axis.get_offset_text().set(fontsize=tick_fontsize, usetex=usetex)
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            tick.label1.set_usetex(usetex)
            tick.label2.set_usetex(usetex)
    return ax, lines
