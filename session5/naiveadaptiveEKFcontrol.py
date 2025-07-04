import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# System Parameters
np.random.seed(0)
b = 3
m = 1
Ts = 1e-2
L = 10
A = np.array([[0, 1], [0, -b/m]])
B = np.array([[0], [1/m]])
C = np.array([1, 0])
Ad = np.eye(2) + Ts * A
Bd = Ts * B

# Time vector and noise
t = np.arange(0, L + Ts, Ts)
w = 1e-5
v = 1e-3
Q = 1e-5 * np.eye(3)
Q[2, 2] = 1e-3
R = v**2
P = 1e-6 * np.eye(3)
P[2, 2] = 3e-2
alpha = 0.03

# Initialization
x_pred = np.zeros((3, len(t)))
x_est = np.zeros((3, len(t)))
y_pred = np.zeros(len(t))
Ce = np.array([1, 0, 0])
x_pred[:, 0] = np.array([2, 0.2, 1.1])
u_c = np.zeros(len(t))
k_c = np.array([6, 5])
x_deseada = np.array([50, 0])
x_c = np.zeros((2, len(t)))
x_c[:, 0] = np.array([1, 2])
y_c = np.zeros(len(t))

# Helper Functions
def fk(x, u, m, T):
    return np.array([
        x[0] + T * x[1],
        x[1] - T * x[2] * x[1] / m + T * u / m,
        x[2]
    ])

def Ak(x, m, T):
    return np.array([
        [1, T, 0],
        [0, 1 - T * x[2] / m, -T * x[1] / m],
        [0, 0, 1]
    ])

# Main Loop
p1, p2 = 8, 0
for k in range(1, len(t)):
    # Adaptive control
    k_c[0] = (p1 + p2) * m - x_est[2, k-1]
    k_c[1] = p1 * p2 * m
    u_c[k-1] = k_c[0] * (x_deseada[0] - x_est[0, k-1]) + k_c[1] * (x_deseada[1] - x_est[1, k-1])

    # True system
    x_c[:, k] = Ad @ x_c[:, k-1] + Bd.flatten() * u_c[k-1] + np.sqrt(w) * np.random.randn(2)
    y_c[k] = C @ x_c[:, k] + np.sqrt(v) * np.random.randn()

    # EKF Prediction
    x_pred[:, k] = fk(x_pred[:, k-1], u_c[k-1], m, Ts)
    y_pred[k] = Ce @ x_pred[:, k]
    A_k = Ak(x_pred[:, k], m, Ts)

    P_pred = A_k @ P @ A_k.T + Q
    P_y = Ce @ P_pred @ Ce.T + R
    P_xy = P_pred @ Ce.T
    Lk = P_xy / P_y

    # EKF Correction
    x_est[:, k] = x_pred[:, k] + Lk * (y_c[k] - y_pred[k])
    P = P_pred - np.outer(Lk, Lk) * P_y

    # Q-Learning Update
    Q = (1 - alpha) * Q + alpha * np.outer(Lk, Lk) * (y_c[k] - y_pred[k])**2

# Plotting with Plotly
fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("Output vs Estimated Position", "Velocity vs Estimated",
                                    "Estimated Friction b", "Control Input"))

# Subplot 1: Position
fig.add_trace(go.Scatter(x=t, y=y_c, name='y (output)', line=dict(color='black')), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=x_est[0, :], name='x₁ estimated', mode='lines', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=np.abs(y_c - x_est[0, :]), name='|y - x₁ est.|', line=dict(color='red')), row=1, col=1)

# Subplot 2: Velocity
fig.add_trace(go.Scatter(x=t, y=x_c[1, :], name='x₂ true', line=dict(color='black')), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=x_est[1, :], name='x₂ estimated', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=np.abs(x_c[1, :] - x_est[1, :]), name='|x₂ - x₂ est.|', line=dict(color='red')), row=2, col=1)

# Subplot 3: Friction
fig.add_trace(go.Scatter(x=t, y=x_est[2, :], name='b estimated', line=dict(color='blue', dash='dash')), row=3, col=1)
fig.add_trace(go.Scatter(x=t, y=[b]*len(t), name='b true', line=dict(color='green', dash='dot')), row=3, col=1)
fig.add_trace(go.Scatter(x=t, y=np.abs(b - x_est[2, :]), name='|b - b est.|', line=dict(color='red')), row=3, col=1)

# Subplot 4: Control
fig.add_trace(go.Scatter(x=t, y=u_c, name='u control', line=dict(color='black')), row=4, col=1)

fig.update_layout(height=950, width=950, title="Adaptive EKF for 1D Cart System",
                  legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))

# Adjust legends: only show one per label
# (We already passed name=..., now manage showlegend selectively)
for i, trace in enumerate(fig.data):
    if trace.name in ['error', '|y - x₁ est.|', '|x₂ - x₂ est.|', '|b - b est.|']:
        trace.showlegend = (i == 2 or i == 5 or i == 8)  # only first 'error' per subplot
    if trace.name in ['x₂ estimated']:
        trace.showlegend = (i == 4)
    if trace.name in ['b estimated']:
        trace.showlegend = (i == 7)
    if trace.name in ['b true']:
        trace.showlegend = (i == 8)

# Axis labels
fig.update_yaxes(title_text="x (position)", row=1, col=1)
fig.update_yaxes(title_text="ẋ (velocity)", row=2, col=1)
fig.update_yaxes(title_text="b (friction)", row=3, col=1)
fig.update_yaxes(title_text="u (control)", row=4, col=1)
fig.update_xaxes(title_text="Time [s]", row=4, col=1)

# Move and format legend
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(255,255,255,0.8)"
    )
)

fig.show()
