"""Real scalar fields represented by a finite, two-dimensional Fourier series."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import convolve2d


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _real(value: float, name: str) -> float:
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


class ToroidalField:
    r"""A real field with phase ``exp(i * (m*theta - n*nfp*zeta))``.

    ``coefficients`` has shape ``(2*mpol+1, 2*ntor+1)``. Its axes are
    increasing m and n, with inclusive bounds [-mpol, mpol], [-ntor, ntor].
    The central entry is the mean, counted once. Reality requires
    C[-m,-n] = conj(C[m,n]); a nonzero independent mode contributes
    ``2*real(C[m,n]*exp(i*phase))``. Thus ``a*cos(phase)+b*sin(phase)``
    has coefficient ``(a-1j*b)/2``. There is no extra nfp normalization.

    Angles are radians; zeta is the full toroidal parameter angle, with
    period 2*pi/nfp (not necessarily a physical cylindrical angle).
    Input arrays are copied. Conjugate symmetry is checked with rtol=1e-12,
    atol=1e-14, then roundoff is symmetrized. Operations return new fields.
    """

    def __init__(self, coefficients: ArrayLike, nfp: int = 1) -> None:
        self._nfp = _integer(nfp, "nfp", 1)
        data = np.array(coefficients, dtype=complex, copy=True)
        if data.ndim != 2 or any(size % 2 != 1 for size in data.shape):
            raise ValueError("coefficients must be a 2D array with positive odd dimensions")
        if not np.all(np.isfinite(data)):
            raise ValueError("coefficients must be finite")
        partner = data[::-1, ::-1].conj()
        if not np.allclose(data, partner, rtol=1e-12, atol=1e-14):
            raise ValueError("coefficients must have real-field conjugate symmetry")
        self._coefficients = data / 2 + partner / 2
        self._coefficients.setflags(write=False)

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def mpol(self) -> int:
        """Largest retained absolute poloidal mode number (inclusive)."""
        return self._coefficients.shape[0] // 2

    @property
    def ntor(self) -> int:
        """Largest retained absolute toroidal mode number (inclusive)."""
        return self._coefficients.shape[1] // 2

    @property
    def coefficients(self) -> NDArray[np.complex128]:
        """A copy of the full centered spectrum, with axes (m, n)."""
        return self._coefficients.copy()

    @property
    def modes(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """One-dimensional m and n arrays indexing the coefficient axes."""
        return np.arange(-self.mpol, self.mpol + 1), np.arange(-self.ntor, self.ntor + 1)

    @property
    def mean(self) -> float:
        """Uniform angular average over one field period."""
        return float(self._coefficients[self.mpol, self.ntor].real)

    @classmethod
    def constant(
        cls, value: float, *, nfp: int = 1, mpol: int = 0, ntor: int = 0
    ) -> ToroidalField:
        """Construct a constant field, optionally padded with zero modes."""
        return cls.from_modes({(0, 0): _real(value, "value")}, nfp=nfp, mpol=mpol, ntor=ntor)

    @classmethod
    def from_modes(
        cls,
        modes: Mapping[tuple[int, int], complex],
        *,
        nfp: int = 1,
        mpol: int | None = None,
        ntor: int | None = None,
    ) -> ToroidalField:
        """Construct from independent complex coefficients; fill conjugate partners.

        Keys must have m>0, or m=0 and n>=0. Omitted modes are zero.
        Bounds default to the largest supplied modes; explicit bounds may
        pad but may not discard a supplied mode. The zero mode must be real.
        """
        for m, n in modes:
            _integer(m, "m")
            if isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral):
                raise TypeError("n must be an integer")
            if m == 0 and n < 0:
                raise ValueError("independent modes require m>0, or m=0 and n>=0")
        required_m = max((m for m, _ in modes), default=0)
        required_n = max((abs(n) for _, n in modes), default=0)
        mpol = required_m if mpol is None else _integer(mpol, "mpol")
        ntor = required_n if ntor is None else _integer(ntor, "ntor")
        if mpol < required_m or ntor < required_n:
            raise ValueError("resolution cannot discard supplied modes")
        data = np.zeros((2 * mpol + 1, 2 * ntor + 1), dtype=complex)
        for (m, n), value in modes.items():
            value = complex(value)
            if m == n == 0 and value.imag != 0:
                raise ValueError("the zero mode must be real")
            data[mpol + m, ntor + n] = value
            data[mpol - m, ntor - n] = value.conjugate()
        return cls(data, nfp=nfp)

    @classmethod
    def from_samples(cls, values: ArrayLike, *, nfp: int = 1) -> ToroidalField:
        """Interpolate real samples with axes (theta, zeta) using an FFT.

        Samples must have odd shape (Ntheta, Nzeta) on the endpoint-excluded
        grid theta[j]=2*pi*j/Ntheta, zeta[k]=2*pi*k/(nfp*Nzeta).
        Both origins are zero. Even grids are rejected to avoid ambiguous
        Nyquist modes. Input frequencies beyond the grid bandwidth alias;
        the caller must supply adequate sampling of the original field.
        """
        from .sampling import coefficients_from_samples

        return cls(coefficients_from_samples(values), nfp=nfp)

    def coefficient(self, m: int, n: int) -> complex:
        """Return C[m,n]; raise IndexError for a mode outside the retained bounds."""
        for value, name in ((m, "m"), (n, "n")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        if abs(m) > self.mpol or abs(n) > self.ntor:
            raise IndexError("mode is outside the retained bounds")
        return complex(self._coefficients[self.mpol + m, self.ntor + n])

    def evaluate(self, theta: ArrayLike, zeta: ArrayLike) -> NDArray[np.float64]:
        """Evaluate at broadcastable angles; return their broadcast shape (scalar: ()).

        For a tensor grid use theta[:, None], zeta[None, :], giving axes
        (theta, zeta). Equal-length 1D arrays instead evaluate paired points.
        """
        theta, zeta = np.asarray(theta), np.asarray(zeta)
        if np.iscomplexobj(theta) or np.iscomplexobj(zeta):
            raise TypeError("angles must be real")
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(zeta)):
            raise ValueError("angles must be finite")
        theta, zeta = np.broadcast_arrays(theta, zeta)
        m, n = self.modes
        poloidal = np.exp(1j * m[:, None] * theta.ravel())
        toroidal = np.exp(-1j * n[:, None] * self.nfp * zeta.ravel())
        values = np.einsum("mn,mp,np->p", self._coefficients, poloidal, toroidal, optimize=True)
        return values.real.reshape(theta.shape)

    __call__ = evaluate

    def sample(
        self, ntheta: int | None = None, nzeta: int | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return theta, zeta and values on an endpoint-excluded one-period grid.

        Coordinate arrays are 1D; values have shape (ntheta, nzeta). Counts
        default to (2*mpol+1, 2*ntor+1) and may only increase that bandwidth.
        """
        from .sampling import angular_grid

        ntheta = 2 * self.mpol + 1 if ntheta is None else _integer(ntheta, "ntheta", 1)
        nzeta = 2 * self.ntor + 1 if nzeta is None else _integer(nzeta, "nzeta", 1)
        if ntheta <= 2 * self.mpol or nzeta <= 2 * self.ntor:
            raise ValueError("sample counts must resolve all retained modes")
        theta, zeta = angular_grid(ntheta, nzeta, self.nfp)
        return theta, zeta, self(theta[:, None], zeta[None, :])

    def derivative(self, theta_order: int = 0, zeta_order: int = 0) -> ToroidalField:
        """Differentiate in parameter angles, including mixed/higher derivatives."""
        theta_order = _integer(theta_order, "theta_order")
        zeta_order = _integer(zeta_order, "zeta_order")
        m, n = self.modes
        factors = (1j * m[:, None]) ** theta_order * (-1j * n * self.nfp) ** zeta_order
        return ToroidalField(self._coefficients * factors, self.nfp)

    def with_resolution(self, mpol: int, ntor: int) -> ToroidalField:
        """Explicitly truncate and/or zero-pad the spectrum to inclusive bounds."""
        mpol, ntor = _integer(mpol, "mpol"), _integer(ntor, "ntor")
        m, n = min(mpol, self.mpol), min(ntor, self.ntor)
        data = np.zeros((2 * mpol + 1, 2 * ntor + 1), dtype=complex)
        data[mpol - m : mpol + m + 1, ntor - n : ntor + n + 1] = self._coefficients[
            self.mpol - m : self.mpol + m + 1, self.ntor - n : self.ntor + n + 1
        ]
        return ToroidalField(data, self.nfp)

    def normalize(self) -> ToroidalField:
        """Divide by the angular mean; reject a zero mean."""
        return self / self.mean

    def shift(self, theta: float = 0, zeta: float = 0) -> ToroidalField:
        """Return g(t,z)=f(t+theta,z+zeta); shifts are radians."""
        theta, zeta = _real(theta, "theta"), _real(zeta, "zeta")
        m, n = self.modes
        return ToroidalField(
            self._coefficients * np.exp(1j * (m[:, None] * theta - n * self.nfp * zeta)),
            self.nfp,
        )

    def reverse_poloidal_angle(self) -> ToroidalField:
        """Return g(theta,zeta)=f(pi-theta,zeta)."""
        m, _ = self.modes
        return ToroidalField(self._coefficients[::-1, :] * (-1.0) ** m[:, None], self.nfp)

    def _check_nfp(self, other: ToroidalField) -> None:
        if self.nfp != other.nfp:
            raise ValueError("fields must have the same nfp")

    def __add__(self, other: ToroidalField | float) -> ToroidalField:
        """Add fields using the union of mode bounds (zero padding, no resampling)."""
        if isinstance(other, ToroidalField):
            self._check_nfp(other)
            m, n = max(self.mpol, other.mpol), max(self.ntor, other.ntor)
            data = (
                self.with_resolution(m, n)._coefficients
                + other.with_resolution(m, n)._coefficients
            )
        elif isinstance(other, Real):
            data = self.coefficients
            data[self.mpol, self.ntor] += _real(other, "other")
        else:
            return NotImplemented
        return ToroidalField(data, self.nfp)

    __radd__ = __add__

    def __neg__(self) -> ToroidalField:
        return ToroidalField(-self._coefficients, self.nfp)

    def __sub__(self, other: ToroidalField | float) -> ToroidalField:
        if not isinstance(other, (ToroidalField, Real)):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: float) -> ToroidalField:
        return -self + other

    def __mul__(self, other: ToroidalField | float) -> ToroidalField:
        """Multiply with full linear convolution; bounds add, without aliasing."""
        if isinstance(other, ToroidalField):
            self._check_nfp(other)
            data = convolve2d(self._coefficients, other._coefficients, mode="full")
        elif isinstance(other, Real):
            data = self._coefficients * _real(other, "other")
        else:
            return NotImplemented
        return ToroidalField(data, self.nfp)

    __rmul__ = __mul__

    def __truediv__(self, other: float) -> ToroidalField:
        """Divide by a real scalar; field division needs an explicit reciprocal."""
        if not isinstance(other, Real):
            return NotImplemented
        other = _real(other, "other")
        if other == 0:
            raise ZeroDivisionError("cannot divide a field by zero")
        return ToroidalField(self._coefficients / other, self.nfp)

    def __pow__(self, exponent: int) -> ToroidalField:
        """Exact nonnegative integer power, retaining the full product spectrum."""
        exponent = _integer(exponent, "exponent")
        result = ToroidalField.constant(1, nfp=self.nfp)
        factor = self
        while exponent:
            if exponent % 2:
                result = result * factor
            exponent //= 2
            if exponent:
                factor = factor * factor
        return result

    def power(
        self, exponent: float, *, mpol: int, ntor: int, sample_shape: tuple[int, int]
    ) -> ToroidalField:
        """Approximate a real power on an explicit odd grid, then truncate.

        The grid must resolve both input and requested output bounds. Nonlinear
        spectra are generally infinite: increase sample_shape and output bounds
        to check convergence. Invalid real powers and zeros at sampled points
        for negative powers raise ValueError; singularities between nodes are
        not detected. For exact nonnegative integer powers use ``field ** k``.
        """
        exponent = _real(exponent, "exponent")
        mpol, ntor = _integer(mpol, "mpol"), _integer(ntor, "ntor")
        if len(sample_shape) != 2:
            raise ValueError("sample_shape must contain (ntheta, nzeta)")
        ntheta = _integer(sample_shape[0], "ntheta", 1)
        nzeta = _integer(sample_shape[1], "nzeta", 1)
        if ntheta % 2 != 1 or nzeta % 2 != 1:
            raise ValueError("nonlinear sampling requires odd grid dimensions")
        if ntheta <= 2 * mpol or nzeta <= 2 * ntor:
            raise ValueError("sample_shape must resolve the requested output modes")
        _, _, values = self.sample(ntheta, nzeta)
        if exponent < 0 and np.any(values == 0):
            raise ValueError("negative powers require nonzero sampled values")
        if not exponent.is_integer() and np.any(values < 0):
            raise ValueError("fractional powers require nonnegative sampled values")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            samples = np.power(values, exponent)
        return ToroidalField.from_samples(samples, nfp=self.nfp).with_resolution(mpol, ntor)

    def reciprocal(
        self, *, mpol: int, ntor: int, sample_shape: tuple[int, int]
    ) -> ToroidalField:
        """Approximate 1/f with explicit sampling and truncation; see power()."""
        return self.power(-1, mpol=mpol, ntor=ntor, sample_shape=sample_shape)
