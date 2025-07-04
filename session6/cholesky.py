import numpy as np

def cholesky_update(L, x, sign='+'):
    """Cholesky rank-one update or downdate.
    L: lower-triangular Cholesky factor (L L^T = A)
    x: vector for update/downdate
    sign: '+' for update, '-' for downdate
    Returns: updated Cholesky factor
    """
    p = L.shape[0]
    x = np.asarray(x).copy()

    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array, got shape {x.shape}")
    if len(x) != p:
        raise ValueError(f"Length of x ({len(x)}) must match L.shape[0] ({p})")

    for k in range(p):
        Lkk = L[k, k]
        if np.abs(Lkk) < 1e-10:
            raise ValueError(f"Near-zero diagonal detected at L[{k},{k}]={Lkk}")

        if sign == '+':
            r2 = Lkk**2 + x[k]**2
        else:
            r2 = Lkk**2 - x[k]**2
            if r2 <= 0:
                raise ValueError(f"Negative value under sqrt in downdate at index {k}")

        r = np.sqrt(r2)
        c = r / Lkk
        s = x[k] / Lkk
        L[k, k] = r

        if k + 1 < p:
            if sign == '+':
                L[k+1:, k] = (L[k+1:, k] + s * x[k+1:]) / c
                x[k+1:] = c * x[k+1:] - s * L[k+1:, k]
            else:
                L[k+1:, k] = (L[k+1:, k] - s * x[k+1:]) / c
                x[k+1:] = c * x[k+1:] - s * L[k+1:, k]

    return L