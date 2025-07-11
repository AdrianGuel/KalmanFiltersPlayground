import numpy as np
import plotly.graph_objects as go

# Environment landmarks (known positions)
landmarks = np.array([[5, 10], [10, 5], [15, 15]])

# Simulation parameters
dt = 0.1
N = 100

# Robot state [x, y, theta]
x_true = np.array([2.0, 2.0, 0.0])
x_est = np.array([2.0, 2.0, 0.0])
P = np.eye(3) * 0.1  # Initial covariance

# Noise models
Q = np.diag([0.05, 0.05, 0.01])  # Process noise
R = np.diag([0.1, np.deg2rad(10)])  # Range and bearing noise

# Control input: constant velocity
v = 1.0
omega = np.deg2rad(15)

true_trajectory = []
estimated_trajectory = []
observations = []

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def motion_model(x, u, dt):
    theta = x[2]
    dx = u[0] * np.cos(theta) * dt
    dy = u[0] * np.sin(theta) * dt
    dtheta = u[1] * dt
    return np.array([x[0] + dx, x[1] + dy, wrap_angle(x[2] + dtheta)])

for k in range(N):
    # True motion
    u = [v, omega]
    x_true = motion_model(x_true, u, dt)
    true_trajectory.append(x_true.copy())

    # Simulated observations: range and bearing to landmarks
    z = []
    for lx, ly in landmarks:
        dx, dy = lx - x_true[0], ly - x_true[1]
        rng = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(R[0, 0]))
        brg = wrap_angle(np.arctan2(dy, dx) - x_true[2] + np.random.normal(0, np.sqrt(R[1, 1])))
        z.append([rng, brg])
    observations.append(z)

    # --- EKF Prediction ---
    theta = x_est[2]
    #Fx = np.eye(3)
    u = [v, omega]
    x_pred = motion_model(x_est, u, dt)

    # Jacobian of motion model
    jF = np.array([
        [1, 0, -v * np.sin(theta) * dt],
        [0, 1,  v * np.cos(theta) * dt],
        [0, 0, 1]
    ])

    P = jF @ P @ jF.T + Q
    x_est = x_pred

    # --- EKF Update for each landmark ---
    for i, (lx, ly) in enumerate(landmarks):
        dx = lx - x_est[0]
        dy = ly - x_est[1]
        q = dx**2 + dy**2
        rng = np.sqrt(q)
        brg = wrap_angle(np.arctan2(dy, dx) - x_est[2])

        # Measurement prediction
        z_hat = np.array([rng, brg])
        z_meas = np.array(observations[k][i])

        # Measurement Jacobian
        H = np.array([
            [-dx/rng, -dy/rng, 0],
            [dy/q, -dx/q, -1]
        ])

        # Kalman gain
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # Innovation
        y = z_meas - z_hat
        y[1] = wrap_angle(y[1])

        # State update
        x_est = x_est + K @ y
        x_est[2] = wrap_angle(x_est[2])
        P = (np.eye(3) - K @ H) @ P

    estimated_trajectory.append(x_est.copy())

# Convert trajectories to arrays
true_trajectory = np.array(true_trajectory)
estimated_trajectory = np.array(estimated_trajectory)

# Plotting using Plotly
fig = go.Figure()

# True trajectory
fig.add_trace(go.Scatter(
    x=true_trajectory[:, 0],
    y=true_trajectory[:, 1],
    mode='lines',
    name='True Path',
    line=dict(width=2)
))

# Estimated trajectory
fig.add_trace(go.Scatter(
    x=estimated_trajectory[:, 0],
    y=estimated_trajectory[:, 1],
    mode='lines',
    name='EKF Estimate',
    line=dict(dash='dash')
))

# Landmarks
fig.add_trace(go.Scatter(
    x=landmarks[:, 0],
    y=landmarks[:, 1],
    mode='markers',
    name='Landmarks',
    marker=dict(color='red', size=10, symbol='x')
))

fig.update_layout(
    title='2D EKF-SLAM: Sensor Fusion with Range-Bearing Observations',
    xaxis_title='X position',
    yaxis_title='Y position',
    legend=dict(x=0.01, y=0.99),
    width=800,
    height=600,
    template='simple_white',
    yaxis=dict(scaleanchor="x", scaleratio=1),  # equal aspect ratio
)

fig.show()
