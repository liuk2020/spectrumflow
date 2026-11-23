"""Spectral toroidal surfaces in cylindrical and Boozer parameter angles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from numbers import Integral
from os import PathLike

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..toroidalfield import ToroidalField

VectorFields = tuple[ToroidalField, ToroidalField, ToroidalField]


def _angles(theta: ArrayLike, zeta: ArrayLike) -> tuple[NDArray, NDArray]:
    theta, zeta = np.asarray(theta), np.asarray(zeta)
    if np.iscomplexobj(theta) or np.iscomplexobj(zeta):
        raise TypeError("angles must be real")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(zeta)):
        raise ValueError("angles must be finite")
    return np.broadcast_arrays(theta, zeta)


@dataclass(frozen=True, eq=False)
class Surface(ABC):
    r"""A surface with R, Z in metres and physical angle phi = s*zeta + omega.

    Angles are radians; ``s=toroidal_sign`` is +1 or -1. Cartesian coordinates
    use the right-handed convention (R*cos(phi), R*sin(phi), Z). R, Z and
    omega share nfp and parameter angles, but may have different mode bounds;
    Fourier conventions are defined by :class:`ToroidalField`. R must be
    positive and the parametrization regular for a physical toroidal surface.

    Angle inputs broadcast like ToroidalField.evaluate: paired 1D arrays
    give paired points; theta[:, None], zeta[None, :] give a tensor grid.
    Geometry vectors put Cartesian components on the last axis. Instances
    are immutable; with_resolution() returns a new surface.
    """

    r: ToroidalField
    z: ToroidalField
    toroidal_sign: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        for name, value in self._fields.items():
            if not isinstance(value, ToroidalField):
                raise TypeError(f"{name} must be a ToroidalField")
        if any(value.nfp != self.r.nfp for value in self._fields.values()):
            raise ValueError("surface fields must have the same nfp")
        if isinstance(self.toroidal_sign, (bool, np.bool_)) or not isinstance(
            self.toroidal_sign, Integral
        ):
            raise TypeError("toroidal_sign must be an integer")
        if self.toroidal_sign not in (-1, 1):
            raise ValueError("toroidal_sign must be +1 or -1")

    @property
    def _fields(self) -> dict[str, ToroidalField]:
        return {"r": self.r, "z": self.z}

    @property
    @abstractmethod
    def _angle_shift(self) -> ToroidalField:
        """Periodic contribution to the physical cylindrical angle."""

    @property
    def nfp(self) -> int:
        return self.r.nfp

    @property
    def mpol(self) -> int:
        """Largest retained absolute m across all surface fields."""
        return max(value.mpol for value in self._fields.values())

    @property
    def ntor(self) -> int:
        """Largest retained absolute n across all surface fields."""
        return max(value.ntor for value in self._fields.values())

    def with_resolution(self, mpol: int, ntor: int) -> Surface:
        """Explicitly truncate/zero-pad every field to the requested bounds."""
        return replace(self, **{
            name: value.with_resolution(mpol, ntor)
            for name, value in self._fields.items()
        })

    def phi(self, theta: ArrayLike, zeta: ArrayLike) -> NDArray[np.float64]:
        """Physical cylindrical angle, unwrapped, with the broadcast shape."""
        theta, zeta = _angles(theta, zeta)
        return self.toroidal_sign * zeta + self._angle_shift(theta, zeta)

    def rz(
        self, theta: ArrayLike, zeta: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (R, Z) in metres at parameter angles; each has broadcast shape."""
        return self.r(theta, zeta), self.z(theta, zeta)

    def position(self, theta: ArrayLike, zeta: ArrayLike) -> NDArray[np.float64]:
        """Return Cartesian positions in metres with shape (..., 3)."""
        r, z = self.rz(theta, zeta)
        phi = self.phi(theta, zeta)
        return np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=-1)

    def covariant_basis(self) -> tuple[VectorFields, VectorFields]:
        r"""Return (e_theta, e_zeta) as triples of spectral fields, in m/rad.

        Components are in the local orthonormal cylindrical basis
        (e_R, e_phi, e_Z), not the coordinate basis (d/dR, d/dphi, d/dZ):
        e_a = (R_a, R*phi_a, Z_a). Products retain their full spectra.
        """
        shift = self._angle_shift
        return (
            (self.r.derivative(1), self.r * shift.derivative(1), self.z.derivative(1)),
            (
                self.r.derivative(zeta_order=1),
                self.r * (self.toroidal_sign + shift.derivative(zeta_order=1)),
                self.z.derivative(zeta_order=1),
            ),
        )

    def basis(self, theta: ArrayLike, zeta: ArrayLike) -> NDArray[np.float64]:
        """Cartesian covariant basis, shape (..., 2, 3), ordered (theta, zeta)."""
        phi = self.phi(theta, zeta)
        cosine, sine = np.cos(phi), np.sin(phi)
        vectors = []
        for components in self.covariant_basis():
            radial, azimuthal, vertical = (value(theta, zeta) for value in components)
            vectors.append(np.stack((
                radial * cosine - azimuthal * sine,
                radial * sine + azimuthal * cosine,
                vertical,
            ), axis=-1))
        return np.stack(vectors, axis=-2)

    def metric(self) -> tuple[ToroidalField, ToroidalField, ToroidalField]:
        """Return (g_theta_theta, g_theta_zeta, g_zeta_zeta) in m²/rad².

        g_ab = e_a dot e_b. Full spectral convolution and zero-padding retain
        every product mode, with no resampling, truncation or aliasing.
        """
        theta, zeta = self.covariant_basis()

        def dot(a: VectorFields, b: VectorFields) -> ToroidalField:
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        return dot(theta, theta), dot(theta, zeta), dot(zeta, zeta)

    def metric_tensor(self, theta: ArrayLike, zeta: ArrayLike) -> NDArray[np.float64]:
        """Evaluate g_ab; shape (..., 2, 2), with axes ordered (theta, zeta)."""
        basis = self.basis(theta, zeta)
        return np.einsum("...ai,...bi->...ab", basis, basis)

    def area_volume(self, ntheta: int = 129, nzeta: int = 129) -> tuple[float, float]:
        """Integrate full-torus (area [m²], enclosed volume [m³]).

        Uses endpoint-excluded trapezoidal quadrature over one field period,
        multiplied by nfp. Increase both counts to check convergence: area
        is generally not band-limited. Volume is the absolute oriented flux
        integral x dot (e_theta cross e_zeta)/3, for a closed surface without
        self-intersections; it is independent of the parameter orientation.
        """
        for count, name, modes in (
            (ntheta, "ntheta", self.mpol), (nzeta, "nzeta", self.ntor)
        ):
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
                raise TypeError(f"{name} must be an integer")
            if count <= 2 * modes:
                raise ValueError(f"{name} must resolve all retained modes")
        theta = 2 * np.pi * np.arange(ntheta)[:, None] / ntheta
        zeta = 2 * np.pi * np.arange(nzeta)[None, :] / (self.nfp * nzeta)
        # The scalar integrands are invariant under rotation of the local
        # cylindrical frame; no Cartesian rotation is needed for quadrature.
        e_theta, e_zeta = (
            np.stack([value(theta, zeta) for value in vector], axis=-1)
            for vector in self.covariant_basis()
        )
        normal = np.cross(e_theta, e_zeta)
        r, z = self.rz(theta, zeta)
        area = 4 * np.pi**2 * np.mean(np.linalg.norm(normal, axis=-1))
        volume = 4 * np.pi**2 * abs(np.mean(r * normal[..., 0] + z * normal[..., 2])) / 3
        return float(area), float(volume)

    def save(self, path: str | PathLike[str]) -> None:
        """Write a versioned compressed NPZ archive to this exact path."""
        from .surface_io import save_surface

        save_surface(self, path)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Surface:
        """Load NPZ without pickle; Surface.load() detects the coordinate type.

        Calling a concrete subclass rejects archives of the other type.
        """
        from .surface_io import load_surface

        surface = load_surface(path)
        if not isinstance(surface, cls):
            raise ValueError(f"archive does not contain a {cls.__name__}")
        return surface


@dataclass(frozen=True, eq=False)
class CylindricalSurface(Surface):
    """R(theta,zeta), Z(theta,zeta) with physical phi=toroidal_sign*zeta.

    The default sign is +1, so zeta is the physical cylindrical angle.
    A sign of -1 reverses its direction; R and Z are still evaluated at the
    supplied parameter zeta, with no implicit reversal of their spectra.
    """

    @property
    def _angle_shift(self) -> ToroidalField:
        return ToroidalField.constant(0, nfp=self.nfp)


@dataclass(frozen=True, eq=False)
class BoozerSurface(Surface):
    """R, Z, omega in Boozer angles, with phi=toroidal_sign*zeta+omega.

    omega is a periodic angle in radians and may have its own mode bounds.
    An opposite omega sign is expressed by negating that field. Geometry
    alone does not establish that supplied angles are magnetic coordinates;
    their Boozer meaning must come from the equilibrium data.
    """

    omega: ToroidalField

    @property
    def _fields(self) -> dict[str, ToroidalField]:
        return {"r": self.r, "z": self.z, "omega": self.omega}

    @property
    def _angle_shift(self) -> ToroidalField:
        return self.omega
