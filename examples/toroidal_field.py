"""Small analytic example; run after installing spectrumflow[plot]."""

import matplotlib.pyplot as plt
import numpy as np

from spectrumflow import ToroidalField


def main() -> None:
    nfp = 3

    def original(theta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        return 2 + 0.8*np.cos(2*theta - nfp*zeta) + 0.4*np.sin(theta + nfp*zeta)

    # FFT samples cover one field period and exclude the periodic endpoints.
    theta_sample = np.linspace(0, 2*np.pi, 17, endpoint=False)
    zeta_sample = np.linspace(0, 2*np.pi/nfp, 19, endpoint=False)
    samples = original(theta_sample[:, None], zeta_sample[None, :])
    field1 = ToroidalField.from_samples(samples, nfp=nfp)
    field_2 = ToroidalField.from_modes(
        {(0, 0): 2, (2, 1): 0.4, (1, -1): -0.2j}, nfp=nfp
    )
    theta_check = np.linspace(0, 2*np.pi, 23, endpoint=False)[:, None]
    zeta_check = np.linspace(0, 2*np.pi, 29, endpoint=False)[None, :]
    error = np.max(abs(field1(theta_check, zeta_check) - original(theta_check, zeta_check)))
    print(f"FFT reconstruction error: {error:.3e}")
    print(f"df/dzeta at (0.2, 0.1): {field_2.derivative(zeta_order=1)(0.2, 0.1):.6f}")

    fig, ax = plt.subplots()
    _, contours = field1.plot(
        ax=ax, one_period=False, filled=False,
    )

    # Overlay the mode-based field only within its first toroidal period.
    field_2.plot(
        ax=ax, filled=True, fontsize=18, one_period=True, colorbar=True,
        levels=contours.levels,
    )

    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
