import numpy as np

def nonlinear_msd_model(Ts=1e-2, L=50, seed=0):
    np.random.seed(seed)

    t = np.arange(0, L + Ts, Ts)
    x = np.zeros((2, len(t)))
    y = np.zeros(len(t))
    
    u = 0
    x[:, 0] = [1.0, 0.5]
    R = 1e-5
    w = 3.0
    m = 1.0
    gamma = 0.8
    xi_1 = 1e-6
    xi_2 = 1e-6

    for k in range(1, len(t)):
        x[0, k] = x[0, k-1] + Ts * (x[1, k-1] + np.sqrt(xi_1) * np.random.randn())
        x[1, k] = x[1, k-1] + (Ts / m) * ( -w**2 * x[0, k-1]**3 - gamma * x[1, k-1] + u + np.sqrt(xi_2) * np.random.randn())
        y[k] = x[1, k] + np.sqrt(R) * np.random.randn()

    return t, x, y, w, m, gamma, u
