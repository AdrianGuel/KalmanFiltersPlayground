import numpy as np
from scipy.linalg import cholesky, qr
import matplotlib.pyplot as plt

def simulate_dynamics(x, u, dt, m):
    pos, vel, b = x
    a = (-b * vel + u) / m
    return np.array([
        pos + vel * dt,
        vel + a * dt,
        b  # fricción constante
    ])

def measurement_function(x):
    return np.array([x[1]])  # sólo medimos velocidad

def generate_sigma_points(x, S, alpha, beta, kappa):
    n = len(x)
    lambda_ = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lambda_)
    A = gamma * S
    X = np.column_stack([x, *(x[:, None] + A.T), *(x[:, None] - A.T)])
    return X

def compute_weights(n, alpha, beta, kappa):
    lambda_ = alpha**2 * (n + kappa) - n
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wc = np.copy(Wm)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)
    return Wm, Wc

def cholesky_update(L, x, sign=1):
    x = x.copy()
    n = len(x)
    for k in range(n):
        r = np.sqrt(L[k, k]**2 + sign * x[k]**2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < n:
            L[k+1:, k] = (L[k+1:, k] + sign * s * x[k+1:]) / c
            x[k+1:] = c * x[k+1:] - s * L[k+1:, k]
    return L

def sr_ukf(f, h, x0, S0, Q, R, us, zs, dt, m, alpha=1, beta=2, kappa=0):
    n = len(x0)
    x = x0.copy()
    S = S0.copy()
    N = len(us)
    xs = np.zeros((N, n))
    Wm, Wc = compute_weights(n, alpha, beta, kappa)
    sqrtWc = np.sqrt(np.abs(Wc))

    for k in range(N):
        X = generate_sigma_points(x, S, alpha, beta, kappa)
        X_pred = np.array([f(Xi, us[k], dt, m) for Xi in X.T]).T
        x_pred = X_pred @ Wm

        X_dev = (X_pred - x_pred[:, None]) * sqrtWc
        if np.any(np.isnan(X_dev)) or np.any(np.isinf(X_dev)):
            print(f"⚠️ NaN o inf detectado en X_dev en paso {k}")
            break

        try:
            U = np.hstack([X_dev, cholesky(Q)])
            _, S_pred = qr(U.T, mode='economic')
            S_pred = S_pred.T
        except np.linalg.LinAlgError:
            print(f"❌ QR failed at iteration {k}")
            break

        Z = np.array([h(Xi) for Xi in X_pred.T]).T
        z_pred = Z @ Wm
        Z_dev = (Z - z_pred[:, None]) * sqrtWc

        try:
            U = np.hstack([Z_dev, cholesky(R)])
            _, Sz = qr(U.T, mode='economic')
            Sz = Sz.T
        except np.linalg.LinAlgError:
            print(f"❌ QR (Z) failed at iteration {k}")
            break

        try:
            Z_centered = Z - z_pred[:, None]        # (1, 7)
            X_centered = X_pred - x_pred[:, None]   # (3, 7)
            Pxz = X_centered @ ((Z_centered * Wc[None, :]).T)  # (3, 1)
            K = Pxz @ np.linalg.inv(Sz @ Sz.T)  # (3, 1)
        except Exception as e:
            print(f"❌ Error en cálculo de K en paso {k}: {e}")
            break

        # Actualización del estado
        x = x_pred + K @ (zs[k] - z_pred)

        # 🔒 Clamp del parámetro de fricción
        x[0] = np.clip(x[0], -10.0, 10.0)
        x[1] = np.clip(x[1], -5.0, 5.0)
        x[2] = np.clip(x[2], 0.0, 1.0)

        # Clipping general por seguridad
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print(f"⚠️ Estado inválido en paso {k}, reiniciando a predicción.")
            x = x_pred.copy()

        U = K @ Sz
        for i in range(U.shape[1]):
            S_pred = cholesky_update(S_pred, U[:, i], -1)
        S = S_pred
        xs[k] = x
    return xs

# --- Simulación y ejecución ---
np.random.seed(0)
dt = 0.01
m = 1.0
true_b = 0.5
x_true = np.array([0.5, 1.0, true_b])
u = 0.0

# 🔧 Aumento de ruido para el parámetro de fricción
Q = np.diag([1e-4, 1e-4, 1e-3])
R = np.diag([1e-2])
S0 = cholesky(np.diag([1e-2, 1e-2, 0.01]))
x0 = np.array([0.0, 0.5, 0.0])
T = 5000
us = np.full(T, u)

xs_true = []
zs = []

for _ in range(T):
    x_true = simulate_dynamics(x_true, u, dt, m)
    z = measurement_function(x_true) + np.random.multivariate_normal([0], R)
    zs.append(z)
    xs_true.append(x_true)

zs = np.array(zs)
xs_true = np.array(xs_true)

xs_est = sr_ukf(simulate_dynamics, measurement_function, x0, S0, Q, R, us, zs, dt, m)

# --- Ploteo ---
plt.figure(figsize=(10, 6))
labels = ['Position', 'Velocity', 'Friction']
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(xs_true[:, i], label='True')
    plt.plot(xs_est[:, i], label='Estimated')
    plt.ylabel(labels[i])
    plt.legend()
plt.xlabel('Time Step')
plt.tight_layout()
plt.show()
