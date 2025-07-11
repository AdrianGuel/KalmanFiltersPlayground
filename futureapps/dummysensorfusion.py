import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Time step and duration
dt = 0.1
N = 200
time = np.arange(N) * dt

# System matrix (constant acceleration model)
F = np.array([
    [1, dt, 0.5*dt**2],
    [0, 1, dt],
    [0, 0, 1]
])

# Process noise
Q = np.diag([1e-4, 1e-3, 1e-2])

# Initial state
x_true = np.array([0.0, 1.0, 0.2])
x_est = np.zeros((3, N))
P = np.eye(3)

# Measurements
gps_R = np.array([[2.0]])      # Position noise variance
accel_R = np.array([[0.1]])    # Accel noise variance

H_gps = np.array([[1, 0, 0]])
H_accel = np.array([[0, 0, 1]])

# Storage
true_states = []
gps_meas = []
accel_meas = []

x = np.zeros(3)

for k in range(N):
    # --- Simulate true state ---
    x_true = F @ x_true + np.random.multivariate_normal(np.zeros(3), Q)
    true_states.append(x_true.copy())

    # Simulate sensor readings
    z_gps = H_gps @ x_true + np.random.normal(0, np.sqrt(gps_R))
    z_accel = H_accel @ x_true + np.random.normal(0, np.sqrt(accel_R))

    gps_meas.append(z_gps.item() if k % 10 == 0 else None)
    accel_meas.append(z_accel.item())

    # --- Kalman Filter Predict ---
    x = F @ x
    P = F @ P @ F.T + Q

    # --- Accelerometer update ---
    y = z_accel.item() - (H_accel @ x).item()
    S = H_accel @ P @ H_accel.T + accel_R
    K = P @ H_accel.T @ np.linalg.inv(S)
    x = x + (K @ np.array([[y]])).reshape(-1)
    P = (np.eye(3) - K @ H_accel) @ P

    # --- GPS update ---
    if k % 10 == 0:
        y = z_gps.item() - (H_gps @ x).item()
        S = H_gps @ P @ H_gps.T + gps_R
        K = P @ H_gps.T @ np.linalg.inv(S)
        x = x + (K @ np.array([[y]])).reshape(-1)
        P = (np.eye(3) - K @ H_gps) @ P

    x_est[:, k] = x

# Convert to arrays
true_states = np.array(true_states)        # (N, 3)
gps_meas = np.array([g if g is not None else np.nan for g in gps_meas])
accel_meas = np.array(accel_meas)

# --- Plotly subplots ---
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=("Position", "Velocity", "Acceleration")
)

# Position
fig.add_trace(go.Scatter(x=time, y=true_states[:, 0], name="True Position", line=dict(color="black")), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=x_est[0], name="Estimated Position", line=dict(dash="dot", color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=gps_meas, name="GPS Measurement", mode="markers", marker=dict(color="red", size=6)), row=1, col=1)

# Velocity
fig.add_trace(go.Scatter(x=time, y=true_states[:, 1], name="True Velocity", line=dict(color="black")), row=2, col=1)
fig.add_trace(go.Scatter(x=time, y=x_est[1], name="Estimated Velocity", line=dict(dash="dot", color="blue")), row=2, col=1)

# Acceleration
fig.add_trace(go.Scatter(x=time, y=true_states[:, 2], name="True Acceleration", line=dict(color="black")), row=3, col=1)
fig.add_trace(go.Scatter(x=time, y=x_est[2], name="Estimated Acceleration", line=dict(dash="dot", color="blue")), row=3, col=1)
fig.add_trace(go.Scatter(x=time, y=accel_meas, name="Accelerometer Measurement", mode="markers", marker=dict(color="orange", size=6)), row=3, col=1)

# Layout
fig.update_layout(
    height=900,
    width=1000,
    title_text="Kalman Filter Sensor Fusion: GPS + Accelerometer",
    legend=dict(x=0.01, y=0.99),
    margin=dict(l=50, r=30, t=50, b=50)
)
fig.update_xaxes(title_text="Time [s]", row=3, col=1)
fig.update_yaxes(title_text="Position", row=1, col=1)
fig.update_yaxes(title_text="Velocity", row=2, col=1)
fig.update_yaxes(title_text="Acceleration", row=3, col=1)

fig.show()
