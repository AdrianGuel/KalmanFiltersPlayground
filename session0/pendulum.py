import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81    # gravity
l = 1.0     # pendulum length
dt = 0.01   # time step
T = 10.0    # total simulation time
steps = int(T / dt)

# True damping
b_true = 0.4

# Initial state: theta, dtheta, b
x_true = np.array([0.5, 0.0, b_true])  # true state for simulation
x_est = np.array([0.4, 0.0, 0.1])      # initial EKF guess

# Initial covariance
P = np.eye(3) * 0.1

# Process and measurement noise
Q = np.diag([1e-5, 1e-5, 1e-6])  # small process noise
R = np.array([[0.01]])           # measurement noise

# Observation matrix (linear)
H = np.array([[1.0, 0.0, 0.0]])  # Only theta is measured

# Storage
estimates = []
truth = []
measurements = []

# EKF functions
def f(x):
    theta, dtheta, b = x
    ddtheta = -g/l * theta - b * dtheta
    theta_new = theta + dtheta * dt
    dtheta_new = dtheta + ddtheta * dt
    return np.array([theta_new, dtheta_new, b])

def jacobian_F(x):
    theta, dtheta, b = x
    F = np.eye(3)
    F[0, 1] = dt
    F[1, 0] = - (g / l) * dt
    F[1, 1] = 1 - b * dt
    F[1, 2] = -dtheta * dt
    return F

def h(x):
    return np.array([x[0]])

# Main loop
for _ in range(steps):
    # --- Simulate true system ---
    theta, dtheta, b = x_true
    ddtheta = -g/l * theta - b * dtheta
    x_true[0] += dtheta * dt
    x_true[1] += ddtheta * dt
    # (b stays constant)

    # Simulated noisy measurement
    z = x_true[0] + np.random.normal(0, np.sqrt(R[0, 0]))

    # --- EKF Prediction ---
    F = jacobian_F(x_est)
    x_pred = f(x_est)
    P = F @ P @ F.T + Q

    # --- EKF Update ---
    y = z - h(x_pred)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(3) - K @ H) @ P

    # Store results
    estimates.append(x_est.copy())
    truth.append(x_true.copy())
    measurements.append(z)

# Convert to arrays
estimates = np.array(estimates)
truth = np.array(truth)
measurements = np.array(measurements)

# --- Plotting ---
time = np.linspace(0, T, steps)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, truth[:, 0], label='True θ')
plt.plot(time, estimates[:, 0], '--', label='Estimated θ')
plt.plot(time, measurements, ':', label='Measured θ')
plt.ylabel("Angle (rad)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, truth[:, 1], label='True θ̇')
plt.plot(time, estimates[:, 1], '--', label='Estimated θ̇')
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, [b_true] * steps, label='True b')
plt.plot(time, estimates[:, 2], '--', label='Estimated b')
plt.ylabel("Damping b")
plt.xlabel("Time (s)")
plt.legend()

plt.tight_layout()
plt.show()
