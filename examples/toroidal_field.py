"""Small analytic example; run after installing spectrumflow[plot]."""

import matplotlib.pyplot as plt
import numpy as np

from spectrumflow import ToroidalField
from spectrumflow.toroidalfield import plot_field


def main() -> None:
    # f = 2 + 0.8*cos(2*theta - 3*zeta) + 0.4*sin(theta + 3*zeta).
    field = ToroidalField.from_modes(
        {(0, 0): 2, (2, 1): 0.4, (1, -1): -0.2j}, nfp=3
    )
    theta, zeta, values = field.sample(17, 19)
    restored = ToroidalField.from_samples(values, nfp=field.nfp)
    error = np.max(abs(restored(theta[:, None], zeta[None, :]) - values))
    print(f"FFT reconstruction error: {error:.3e}")
    print(f"df/dzeta at (0.2, 0.1): {field.derivative(zeta_order=1)(0.2, 0.1):.6f}")
    ax, artist = plot_field(field, levels=30)
    ax.figure.colorbar(artist, ax=ax, label="f [a.u.]")
    ax.figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
