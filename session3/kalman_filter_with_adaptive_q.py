import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ca_generative_model import ca_generative_model

def run_lkf(method="bar_shalom"):
    Ts = 1e-2
    L = 1000
    Ad, C, t, x, y = ca_generative_model(Ts, L)
    R = 1e-2
    alpha = 0.001
    std_scale = 3
    std_dev = np.std(y)

    P = 1e4 * np.eye(3)
    Q = 1e-3 * np.eye(3)
    Q_base = Q.copy()

    x_pred = np.zeros_like(x)
    x_est = np.zeros_like(x)
    y_pred = np.zeros(len(t))
    v_sum = np.zeros(15)
    eps_max = 0.1
    Q_scale_factor = 10
    count = 0
    phi = 1e-4 * np.eye(3)  # For "Z" method

    x_pred[:, 0] = [3, -1, 10]

    for k in range(1, len(t)):
        # Prediction
        x_pred[:, k] = Ad @ x_pred[:, k - 1]
        y_pred[k] = C @ x_pred[:, k]

        # Covariance prediction
        P_pred = Ad @ P @ Ad.T + Q
        P_y = C @ P_pred @ C.T + R
        P_xy = P_pred @ C.T
        L_gain = P_xy / P_y

        # Update
        residual = y[k] - y_pred[k]
        x_est[:, k] = x_pred[:, k] + (L_gain.flatten() * residual)
        P = P_pred - L_gain @ C @ P_pred

        # Store residual
        v_sum[-1] = residual
        v_sum = np.roll(v_sum, -1)

        # === Q update method ===
        if method == "innovation":
            vk = np.mean(v_sum)
            Q = vk * (L_gain @ L_gain.T)

        elif method == "q_learning":
            Q = (1 - alpha) * Q + alpha * (L_gain @ L_gain.T) * residual**2

        elif method == "scaling_factor":
            alpha_k = (np.mean(v_sum) - R) / np.trace(C @ P_pred @ C.T)
            Q = alpha_k * Q_base

        elif method == "bar_shalom":
            eps = residual / P_y
            if eps > eps_max:
                Q *= Q_scale_factor
                count += 1
            elif count > 0:
                Q /= Q_scale_factor
                count -= 1

        elif method == "z_method":
            if np.abs(y[k]) > std_scale * std_dev:
                Q += phi
                count += 1
            elif count > 0:
                Q -= phi
                count -= 1

    return t, x, x_est, y

def plot_results(t, x, x_est, y):
    labels = ["position", "velocity", "acceleration"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[f"{label.capitalize()}" for label in labels])

    for i in range(3):
        fig.add_trace(go.Scatter(x=t, y=x[i, :], mode='lines', name=f'{labels[i]} truth',
                                 line=dict(color='blue')),
                      row=i+1, col=1)

        fig.add_trace(go.Scatter(x=t, y=x_est[i, :], mode='lines', name=f'{labels[i]} estimate',
                                 line=dict(color='black', dash='dash')),
                      row=i+1, col=1)

        fig.add_trace(go.Scatter(x=t, y=np.abs(x[i, :] - x_est[i, :]), mode='lines', name=f'{labels[i]} error',
                                 line=dict(color='red', dash='dot')),
                      row=i+1, col=1)

        fig.update_yaxes(title_text=labels[i].capitalize(), row=i+1, col=1)

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_layout(height=800, width=900, title_text="Kalman Filter State Estimation",
                      legend=dict(x=0.01, y=0.99), showlegend=True)
    fig.show()

# Example usage:
if __name__ == "__main__":
    t, x, x_est, y = run_lkf(method="bar_shalom")  # Change method here
    plot_results(t, x, x_est, y)
