"""Matplotlib rendering for scalar radial profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D

    from .radial_profile import RadialProfile


def _plot_profile(
    profile: RadialProfile,
    *,
    ax: Axes | None,
    xlabel: str | None,
    ylabel: str | None,
    **kwargs: Any,
) -> tuple[Axes, Line2D]:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    line, = ax.plot(profile.coordinate, profile.values, **kwargs)
    if xlabel is None:
        xlabel = profile.coordinate_name
        if profile.coordinate_unit:
            xlabel += f" [{profile.coordinate_unit}]"
    if ylabel is None:
        ylabel = profile.name
        if profile.unit:
            ylabel += f" [{profile.unit}]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax, line
