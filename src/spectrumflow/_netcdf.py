"""Shared NetCDF snapshots and explicit real-Fourier conversion at IO boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from os import PathLike

import numpy as np
from numpy.typing import NDArray

from .toroidalfield import ToroidalField


@dataclass
class _NetcdfData:
    variables: dict[str, NDArray]
    dimensions: dict[str, tuple[str, ...]]
    sizes: dict[str, int]
    units: dict[str, str]

    @classmethod
    def read(cls, path: str | PathLike[str]) -> _NetcdfData:
        try:
            from netCDF4 import Dataset
        except ImportError as error:
            raise ImportError("NetCDF reading requires spectrumflow[netcdf]") from error

        with Dataset(path) as dataset:
            variables = {
                name: np.asanyarray(variable[...]).copy()
                for name, variable in dataset.variables.items()
            }
            dimensions = {
                name: variable.dimensions for name, variable in dataset.variables.items()
            }
            units = {
                name: str(getattr(variable, "units", ""))
                for name, variable in dataset.variables.items()
            }
            sizes = {name: len(dim) for name, dim in dataset.dimensions.items()}
        return cls(variables, dimensions, sizes, units)

    def array(
        self,
        name: str,
        dimensions: tuple[str, ...],
        *,
        index: int | slice | None = None,
    ) -> NDArray[np.float64]:
        """Read finite real data with explicitly named axes, reordering if needed."""
        raw = self.variables[name]
        dims = self.dimensions[name]
        if len(dims) != len(dimensions) or set(dims) != set(dimensions):
            raise ValueError(f"{name} must have dimensions {dimensions}, got {dims}")
        raw = raw.transpose(tuple(dims.index(dim) for dim in dimensions))
        if index is not None:
            raw = raw[index]
        if np.any(np.ma.getmaskarray(raw)):
            raise ValueError(f"{name} contains missing data")
        if np.iscomplexobj(raw):
            raise TypeError(f"{name} must be real")
        data = np.asarray(raw, dtype=float)
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} must be finite")
        return data

    def integer(self, name: str, minimum: int = 0) -> int:
        value = self.array(name, ()).item()
        if value != np.rint(value) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    def asymmetry(self) -> bool:
        value = self.integer("lasym__logical__")
        if value not in (0, 1):
            raise ValueError("lasym__logical__ must be 0 or 1")
        return bool(value)


class _FourierModes:
    def __init__(self, m: NDArray, xn: NDArray, nfp: int) -> None:
        if m.ndim != 1 or m.size == 0 or xn.shape != m.shape:
            raise ValueError("mode lists must have the same nonempty 1D shape")
        if np.any(m != np.rint(m)) or np.any(xn != np.rint(xn)):
            raise ValueError("mode numbers must be integers")
        if np.any(m < 0) or np.any(xn % nfp != 0):
            raise ValueError("m must be nonnegative and xn must be a multiple of nfp")
        self.m, self.n = m.astype(int), (xn / nfp).astype(int)
        self.nfp = nfp
        # For m=0, opposite n labels belong to the same independent mode.
        keys = [(int(m), abs(int(n)) if m == 0 else int(n))
                for m, n in zip(self.m, self.n)]
        if len(set(keys)) != len(keys):
            raise ValueError("mode lists contain duplicate or conjugate modes")

    def field(self, cosine: NDArray, sine: NDArray) -> ToroidalField:
        modes = {}
        for m, n, a, b in zip(self.m, self.n, cosine, sine):
            m, n = int(m), int(n)
            if m == n == 0:
                if b != 0:
                    raise ValueError("the zero-mode sine coefficient must be zero")
                modes[m, n] = complex(a)
            elif m == 0 and n < 0:
                modes[0, -n] = complex(a, b) / 2
            else:
                modes[m, n] = complex(a, -b) / 2
        return ToroidalField.from_modes(modes, nfp=self.nfp)


def _surface_index(index: int, count: int) -> int:
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
        raise TypeError("surface_index must be an integer")
    if not -count <= index < count:
        raise IndexError("surface_index is outside the radial grid")
    return int(index % count)


def _read_fourier(
    data: _NetcdfData,
    modes: _FourierModes,
    cosine_name: str,
    sine_name: str,
    dimensions: tuple[str, str],
    index: int,
    *,
    lasym: bool,
    sine_symmetric: bool = False,
) -> ToroidalField:
    coefficients = []
    for name, required in (
        (cosine_name, lasym or not sine_symmetric),
        (sine_name, lasym or sine_symmetric),
    ):
        if not required and name not in data.variables:
            coefficients.append(np.zeros(modes.m.size))
            continue
        values = data.array(name, dimensions, index=index)
        if values.shape != modes.m.shape:
            raise ValueError(f"{name} does not match its mode list")
        if not required and np.any(values != 0):
            raise ValueError(f"{name} is nonzero but lasym__logical__ is false")
        coefficients.append(values)
    return modes.field(*coefficients)
