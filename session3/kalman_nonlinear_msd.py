import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nonlinear_msd_model import nonlinear_msd_model

def run_lkf_nonlinear_msd():
    Ts = 1e-2
    L = 50
    t, x, y, w, m, gamma, u = nonlinear_msd_model(Ts, L)

    P = 1e5 * np.eye(2)
    Q = 1e-6 * np.eye(2)
    R = 1e-5
    alpha = 0.1

    x_pred = np.zeros_like(x)
    x_est = np.zeros_like(x)
    y_pred = np.zeros(len(t))
    C = np.array([[0, 1]])

    x_pred[:, 0] = [0.1, 0.2]

    for k in range(1, len(t)):
        # Nonlinear state prediction
        x_pred[0, k] = x_pred[0, k-1] + Ts * x_pred[1, k-1]
        x_pred[1, k] = x_pred[1, k-1] + (Ts / m) * (-w**2 * x_pred[0, k-1]**3 - gamma * x_pred[1, k-1] + u)

        y_pred[k] = x_pred[1, k]

        # Linearized model Jacobian
        Ak = np.array([[1, Ts],
                       [-3 * Ts * w**2 * x_pred[0, k]**2 / m, 1 - Ts * gamma / m]])

        # Kalman Filter
        P_pred = Ak @ P @ Ak.T + Q
        P_y = C @ P_pred @ C.T + R
        P_xy = P_pred @ C.T
        L_gain = P_xy / P_y

        residual = y[k] - y_pred[k]
        x_est[:, k] = x_pred[:, k] + (L_gain.flatten() * residual)
        P = P_pred - L_gain @ C @ P_pred

        # Q-learning update
        Q = (1 - alpha) * Q + alpha * (L_gain @ L_gain.T) * residual**2

    return t, x, x_est, y

def plot_msd_results(t, x, x_est, y):
    labels = ["position", "velocity"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[f"{label.capitalize()}" for label in labels])

    # Position
    fig.add_trace(go.Scatter(x=t, y=x[0], mode='lines', name='Position Truth', line=dict(color='blue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_est[0], mode='lines', name='Position Estimate', line=dict(color='black', dash='dash')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.abs(x[0] - x_est[0]), mode='lines', name='Position Error', line=dict(color='red', dash='dot')),
                  row=1, col=1)

    # Velocity
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Velocity Measurement', line=dict(color='blue')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=x_est[1], mode='lines', name='Velocity Estimate', line=dict(color='black', dash='dash')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.abs(y - x_est[1]), mode='lines', name='Velocity Error', line=dict(color='red', dash='dot')),
                  row=2, col=1)

    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Velocity", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=700, width=900, title_text="Kalman Filter with Q-Learning: Nonlinear Mass-Spring-Damper",
                      legend=dict(x=0.01, y=0.99), showlegend=True)
    fig.show()

# Example usage
if __name__ == "__main__":
    t, x, x_est, y = run_lkf_nonlinear_msd()
    plot_msd_results(t, x, x_est, y)
