"""Internal Matplotlib implementation for ToroidalField.plot()."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .field import ToroidalField, _integer

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.contour import QuadContourSet


def _style_axes(ax: Axes, fontsize: float, tick_fontsize: float, usetex: bool) -> None:
    """Set text properties on artists so they persist until deferred drawing."""
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    for axis in (ax.xaxis, ax.yaxis):
        axis.label.set(fontsize=fontsize, usetex=usetex)
        axis.get_offset_text().set(fontsize=tick_fontsize, usetex=usetex)
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            tick.label1.set_usetex(usetex)
            tick.label2.set_usetex(usetex)


def _plot_field(
    field: ToroidalField,
    *,
    ax: Axes | None = None,
    ntheta: int = 129,
    nzeta: int = 129,
    one_period: bool = True,
    filled: bool = True,
    cmap: str | Colormap | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fontsize: float = 18,
    tick_fontsize: float | None = None,
    usetex: bool = True,
    figsize: tuple[float, float] | None = None,
    colorbar: bool = False,
    colorbar_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[Axes, QuadContourSet]:
    """Render a field; see ToroidalField.plot() for the public interface."""
    import matplotlib.pyplot as plt

    ntheta, nzeta = _integer(ntheta, "ntheta", 2), _integer(nzeta, "nzeta", 2)
    theta = np.linspace(0, 2 * np.pi, ntheta)
    zeta = np.linspace(0, 2 * np.pi / (field.nfp if one_period else 1), nzeta)
    values = field(theta[:, None], zeta[None, :])
    if cmap is None and kwargs.get("colors") is None:
        cmap = "tol:sunset"
    if isinstance(cmap, str) and ":" in cmap:
        from cmap import Colormap
        cmap = Colormap(cmap).to_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    draw = ax.contourf if filled else ax.contour
    artist = draw(zeta, theta, values, cmap=cmap, **kwargs)
    ax.set_xlabel(r"$\zeta$" if xlabel is None else xlabel)
    ax.set_ylabel(r"$\theta$" if ylabel is None else ylabel)
    if one_period:
        xlabels = ["$0$", r"$\pi/N_\mathrm{fp}$", r"$2\pi/N_\mathrm{fp}$"]
    else:
        xlabels = ["$0$", r"$\pi$", r"$2\pi$"]
    ax.set_xticks(np.linspace(0, zeta[-1], 3), labels=xlabels)
    ax.set_yticks([0, np.pi, 2 * np.pi], labels=["$0$", r"$\pi$", r"$2\pi$"])
    tick_fontsize = fontsize if tick_fontsize is None else tick_fontsize
    _style_axes(ax, fontsize, tick_fontsize, usetex)
    if colorbar:
        from matplotlib.ticker import ScalarFormatter

        bar_kwargs = dict(colorbar_kwargs or {})
        bar = ax.figure.colorbar(artist, ax=ax, **bar_kwargs)
        if "format" not in bar_kwargs and isinstance(bar.formatter, ScalarFormatter):
            # Math mode keeps numerical ticks consistent with the angle labels.
            bar.formatter.set_useMathText(True)
            bar.update_ticks()
        _style_axes(bar.ax, fontsize, tick_fontsize, usetex)
    return ax, artist
