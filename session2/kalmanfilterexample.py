import numpy as np
import matplotlib.pyplot as plt

# RC parameters
R = 1.0
C = 1.0
T = 0.1
A = 1 - T / (R * C)
B = T / (R * C)
C_mat = 1.0

# Noise covariances
Qv = 0.01   # process noise
Qxi = 0.1   # measurement noise

# Initial estimates
x_hat = 0.0
P = 1.0

# Input signal
N = 100
u = np.ones(N)  # Step input
x_true = np.zeros(N)
y_meas = np.zeros(N)
x_est = np.zeros(N)

# Simulate system with noise
for k in range(1, N):
    x_true[k] = A * x_true[k-1] + B * u[k-1] + np.random.normal(0, np.sqrt(Qv))
    y_meas[k] = x_true[k] + np.random.normal(0, np.sqrt(Qxi))

# Kalman filter loop
for k in range(1, N):
    # --- Predict ---
    x_pred = A * x_hat + B * u[k]
    P_pred = A * P * A + Qv
    Py = C_mat * P_pred * C_mat + Qxi
    Pxy = P_pred * C_mat

    # --- Update ---
    L = Pxy / Py
    x_hat = x_pred + L * (y_meas[k] - C_mat * x_pred)
    P = P_pred - L * Py * L

    x_est[k] = x_hat

# Plot
plt.plot(x_true, label="True Voltage")
plt.plot(y_meas, label="Measured Voltage", linestyle='dotted')
plt.plot(x_est, label="Estimated Voltage", linestyle='--')
plt.xlabel("Time step")
plt.ylabel("Voltage")
plt.title("RC Circuit Kalman Filter Estimation")
plt.legend()
plt.grid()
plt.show()
