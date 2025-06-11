import numpy as np
import matplotlib.pyplot as plt

# Circuit parameters
R = 1.0  # Resistance in ohms
L = 1.0  # Inductance in henrys
C = 1.0  # Capacitance in farads
T = 0.01  # Time step in seconds

# Discretize using Euler method
Ad = np.array([[1 - T / (R * C), T / C],
               [-T / L, 1]])
Bd = np.array([[0],
               [T / L]])
C_mat = np.array([1, 0])  # Output is capacitor voltage

# Noise covariances
Qv = 0.001 * np.eye(2)  # process noise
Qxi = 0.01              # measurement noise variance

# Initial estimates
x_hat = np.zeros(2)
P = np.eye(2)

# Input signal
N = 1000
u = np.ones(N)  # Step input
x_true = np.zeros((2, N))  # [v_C; i_L]
y_meas = np.zeros(N)
x_est = np.zeros(N)

# Simulate system with noise
for k in range(1, N):
    process_noise = np.random.multivariate_normal(mean=[0, 0], cov=Qv)
    x_true[:, k] = Ad @ x_true[:, k-1] + Bd.flatten() * u[k-1] + process_noise
    y_meas[k] = C_mat @ x_true[:, k] + np.random.normal(0, np.sqrt(Qxi))

# Kalman filter loop
for k in range(1, N):
    # --- Predict ---
    x_pred = Ad @ x_hat + Bd.flatten() * u[k]
    P_pred = Ad @ P @ Ad.T + Qv
    Py = C_mat @ P_pred @ C_mat.T + Qxi
    Pxy = P_pred @ C_mat.T

    # --- Update ---
    L_gain = Pxy / Py
    x_hat = x_pred + L_gain * (y_meas[k] - C_mat @ x_pred)
    P = P_pred - np.outer(L_gain, C_mat) @ P_pred

    x_est[k] = x_hat[0]  # Estimated capacitor voltage

# Plot
plt.plot(x_true[0], label="True Capacitor Voltage")
plt.plot(y_meas, label="Measured Voltage", linestyle='dotted')
plt.plot(x_est, label="Estimated Voltage", linestyle='--')
plt.xlabel("Time step")
plt.ylabel("Voltage (V)")
plt.title("Series-Parallel RLC Circuit Kalman Filter Estimation")
plt.legend()
plt.grid()
plt.show()
