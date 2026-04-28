"""Linear algebra helpers."""
from __future__ import annotations

import numpy as np


def weighted_svd(
    Z: np.ndarray,
    row_weights: np.ndarray | None = None,
    col_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute weighted SVD for initialising bilinear mortality models.

    Computes the SVD of the scaled matrix  W^{1/2} Z V^{1/2}  and returns
    the leading singular vectors unscaled back to the original space.

    Parameters
    ----------
    Z:
        Matrix to decompose, shape (m, n).
    row_weights:
        Non-negative row weights, shape (m,).  Defaults to uniform.
    col_weights:
        Non-negative column weights, shape (n,).  Defaults to uniform.

    Returns
    -------
    U, s, Vt:
        Leading singular vectors and values.  U has shape (m, k), s (k,),
        Vt (k, n).
    """
    m, n = Z.shape
    rw = np.sqrt(row_weights) if row_weights is not None else np.ones(m)
    cw = np.sqrt(col_weights) if col_weights is not None else np.ones(n)

    # Guard against zeros
    rw = np.where(rw > 0, rw, 0.0)
    cw = np.where(cw > 0, cw, 0.0)

    Zw = (Z * rw[:, None]) * cw[None, :]
    U, s, Vt = np.linalg.svd(Zw, full_matrices=False)

    # Unscale
    rw_inv = np.where(rw > 0, 1.0 / np.where(rw > 0, rw, 1.0), 0.0)
    cw_inv = np.where(cw > 0, 1.0 / np.where(cw > 0, cw, 1.0), 0.0)
    U = U * rw_inv[:, None]
    Vt = Vt * cw_inv[None, :]

    return U, s, Vt
