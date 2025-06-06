import numpy as np

def ca_generative_model(Ts=1e-2, L=1000, Q_scale=1e-6, R_scale=1e-3, seed=0):
    np.random.seed(seed)
    Ad = np.array([[1, Ts, 0.5 * Ts**2],
                   [0,  1,       Ts],
                   [0,  0,        1]])
    C = np.array([[0, 0, 1]])
    t = np.arange(0, L + Ts, Ts)
    x = np.zeros((3, len(t)))
    y = np.zeros(len(t))
    x[:, 0] = [0, -50, 10]
    Q = Q_scale * np.eye(3)
    R = R_scale

    for k in range(1, len(t)):
        x[:, k] = Ad @ x[:, k - 1] + np.sqrt(Q) @ np.random.randn(3)
        y[k] = C @ x[:, k] + np.sqrt(R) * np.random.randn()

    return Ad, C, t, x, y
