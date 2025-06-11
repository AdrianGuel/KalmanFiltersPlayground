import numpy as np
from scipy.linalg import solve_discrete_are
import plotly.graph_objects as go

# Simulation parameters
dt = 0.01
T = 10.0
steps = int(T / dt)
t = np.linspace(0, T, steps)

# True system parameters
m = 1.0
b_true = 2.0

# EKF parameters
Q = np.diag([1e-5, 1e-5, 1e-6])  # Process noise covariance
R = np.diag([1e-3, 1e-3])        # Measurement noise covariance
H = np.array([[1, 0, 0],
              [0, 1, 0]])        # Measurement matrix

# Initial state estimates
x_hat = np.array([1.0, 0.0, 1.0])  # [x, x_dot, b]
P = np.eye(3) * 0.1

# State storage
x_true = np.zeros((3, steps))
x_true[0, 0] = 1.0  # Initial position
x_hat_hist = np.zeros((3, steps))
x_hat_hist[:, 0] = x_hat
u_hist = np.zeros(steps)
K_hist = np.zeros((2, steps))
desired_position = 10.0
for k in range(steps - 1):
    # Recompute LQR gain using current estimate of b
    A_est = np.array([[1, dt],
                      [0, 1 - dt * x_hat[2] / m]])
    B = np.array([[0],
                  [dt / m]])
    Q_lqr = np.diag([10.0, 1.0])
    R_lqr = np.array([[0.1]])
    P_lqr = solve_discrete_are(A_est, B, Q_lqr, R_lqr)
    K = np.linalg.inv(R_lqr + B.T @ P_lqr @ B) @ (B.T @ P_lqr @ A_est)
    K = K.flatten()
    K_hist[:, k] = K

    # Control input
    x_ref = np.array([desired_position, 0.0])  # e.g., [2.0, 0.0]
    u = -K @ (x_hat[:2] - x_ref)
    u_hist[k] = u

    # Simulate true dynamics
    x_true[0, k + 1] = x_true[0, k] + dt * x_true[1, k]
    x_true[1, k + 1] = x_true[1, k] + dt * (u - b_true * x_true[1, k]) / m
    x_true[2, k + 1] = b_true

    # EKF prediction
    x1h, x2h, bh = x_hat
    x1h_pred = x1h + dt * x2h
    x2h_pred = x2h + dt * (u - bh * x2h) / m
    bh_pred = bh
    x_hat_pred = np.array([x1h_pred, x2h_pred, bh_pred])

    A = np.array([
        [1, dt, 0],
        [0, 1 - dt * bh / m, -dt * x2h / m],
        [0, 0, 1]
    ])
    P_pred = A @ P @ A.T + Q

    # EKF update
    z = x_true[:2, k + 1] + np.random.multivariate_normal(np.zeros(2), R)
    y = z - H @ x_hat_pred
    S = H @ P_pred @ H.T + R
    K_ekf = P_pred @ H.T @ np.linalg.inv(S)
    x_hat = x_hat_pred + K_ekf @ y
    P = (np.eye(3) - K_ekf @ H) @ P_pred
    x_hat_hist[:, k + 1] = x_hat

# Plot results
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=x_true[0], mode='lines', name='True Position'))
fig.add_trace(go.Scatter(x=t, y=x_hat_hist[0], mode='lines', name='Estimated Position'))
fig.add_trace(go.Scatter(x=t, y=x_true[1], mode='lines', name='True Velocity'))
fig.add_trace(go.Scatter(x=t, y=x_hat_hist[1], mode='lines', name='Estimated Velocity'))
fig.add_trace(go.Scatter(x=t, y=x_hat_hist[2], mode='lines', name='Estimated Friction b'))
fig.update_layout(title='Adaptive EKF-Based Full-State Feedback Control',
                  xaxis_title='Time [s]',
                  yaxis_title='States / Parameter')
fig.show()
