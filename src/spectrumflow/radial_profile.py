"""Scalar physical quantities sampled on an explicit radial coordinate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D


class RadialProfile:
    """A scalar quantity on a strictly increasing, one-dimensional grid.

    ``coordinate`` and ``values`` have the same nonempty shape ``(nr,)``.
    The default coordinate is normalized toroidal flux s; other radial
    labels and units must be supplied explicitly. Arrays are copied, and
    array properties return copies. Each instance stores one quantity.
    """

    def __init__(
        self,
        coordinate: ArrayLike,
        values: ArrayLike,
        *,
        name: str = "",
        unit: str = "",
        coordinate_name: str = "s",
        coordinate_unit: str = "",
        grid: str | None = None,
    ) -> None:
        if np.iscomplexobj(coordinate) or np.iscomplexobj(values):
            raise TypeError("coordinate and values must be real")
        if np.any(np.ma.getmaskarray(coordinate)) or np.any(np.ma.getmaskarray(values)):
            raise ValueError("coordinate and values must not contain missing data")
        radius = np.array(coordinate, dtype=float, copy=True)
        data = np.array(values, dtype=float, copy=True)
        if radius.ndim != 1 or radius.size == 0 or data.shape != radius.shape:
            raise ValueError("coordinate and values must have the same nonempty 1D shape")
        if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(data)):
            raise ValueError("coordinate and values must be finite")
        if np.any(np.diff(radius) <= 0):
            raise ValueError("coordinate must be strictly increasing")
        radius.setflags(write=False)
        data.setflags(write=False)
        self._coordinate, self._values = radius, data
        self.name, self.unit = name, unit
        self.coordinate_name, self.coordinate_unit = coordinate_name, coordinate_unit
        self.grid = grid

    @property
    def coordinate(self) -> NDArray[np.float64]:
        """A copy of the radial coordinates, shape (nr,)."""
        return self._coordinate.copy()

    @property
    def values(self) -> NDArray[np.float64]:
        """A copy of the scalar samples, shape (nr,)."""
        return self._values.copy()

    def evaluate(self, coordinate: ArrayLike) -> NDArray[np.float64]:
        """Linearly interpolate, preserving input shape; reject extrapolation.

        A scalar input returns a zero-dimensional array. A one-point
        profile can only be evaluated at its single coordinate.
        """
        if np.iscomplexobj(coordinate):
            raise TypeError("coordinate must be real")
        if np.any(np.ma.getmaskarray(coordinate)):
            raise ValueError("coordinate must not contain missing data")
        radius = np.asarray(coordinate, dtype=float)
        if not np.all(np.isfinite(radius)):
            raise ValueError("coordinate must be finite")
        if np.any(radius < self._coordinate[0]) or np.any(radius > self._coordinate[-1]):
            raise ValueError("coordinate is outside the sampled interval")
        return np.asarray(np.interp(radius.ravel(), self._coordinate, self._values)).reshape(
            radius.shape
        )

    __call__ = evaluate

    def plot(
        self,
        *,
        ax: Axes | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        **kwargs: Any,
    ) -> tuple[Axes, Line2D]:
        """Plot the stored samples; return (axes, line), without show/save.

        Default labels include the quantity/coordinate names and supplied
        units. Other keyword arguments are passed to ax.plot().
        """
        from .profile_plotting import _plot_profile

        return _plot_profile(self, ax=ax, xlabel=xlabel, ylabel=ylabel, **kwargs)
