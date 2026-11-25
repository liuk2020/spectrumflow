"""Internal Matplotlib implementation for toroidal surface plots."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .surface import Surface, _angles

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _sample_count(count: int, name: str) -> int:
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
        raise TypeError(f"{name} must be an integer")
    if count < 3:
        raise ValueError(f"{name} must be >= 3")
    return int(count)


def _style_axis(axis: Axis, fontsize: float, tick_fontsize: float, usetex: bool) -> None:
    from matplotlib.ticker import ScalarFormatter

    formatter = axis.get_major_formatter()
    if isinstance(formatter, ScalarFormatter):
        formatter.set_useMathText(True)
    axis.label.set(fontsize=fontsize, usetex=usetex)
    axis.get_offset_text().set(fontsize=tick_fontsize, usetex=usetex)
    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
        tick.label1.set_usetex(usetex)
        tick.label2.set_usetex(usetex)


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

    ntheta = _sample_count(ntheta, "ntheta")
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
        _style_axis(axis, fontsize, tick_fontsize, usetex)
    return ax, lines


def _plot_3d(
    surface: Surface,
    *,
    ax: Axes3D | None,
    ntheta: int,
    nzeta: int,
    one_period: bool,
    fontsize: float,
    tick_fontsize: float | None,
    usetex: bool,
    figsize: tuple[float, float] | None,
    **kwargs: Any,
) -> tuple[Axes3D, Poly3DCollection]:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ntheta = _sample_count(ntheta, "ntheta")
    nzeta = _sample_count(nzeta, "nzeta")
    if ax is not None and not isinstance(ax, Axes3D):
        raise TypeError("ax must be a 3D axes")
    theta = np.linspace(0, 2 * np.pi, ntheta)
    zeta = np.linspace(0, 2 * np.pi / (surface.nfp if one_period else 1), nzeta)
    xyz = surface.position(theta[:, None], zeta[None, :])
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "3d"})
    if "rstride" not in kwargs and "cstride" not in kwargs:
        kwargs.setdefault("rcount", ntheta)
        kwargs.setdefault("ccount", nzeta)
    artist = ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2], **kwargs)
    # Match the display box to the data limits so one metre has the same
    # length along all axes, including when overlaying existing geometry.
    ax.set_box_aspect([
        upper - lower
        for lower, upper in (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())
    ])
    ax.set_xlabel(r"$x$", labelpad=12)
    ax.set_ylabel(r"$y$", labelpad=12)
    ax.set_zlabel(r"$z$", labelpad=12)
    tick_fontsize = fontsize if tick_fontsize is None else tick_fontsize
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        _style_axis(axis, fontsize, tick_fontsize, usetex)
    return ax, artist
