# Import required libraries
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# --- SYSTEM PARAMETERS ---
l1, l2 = 1.0, 1.0        # Length of link 1 and link 2
m1, m2 = 1.0, 1.0        # Mass of link 1 and link 2
g = 9.81                 # Gravitational acceleration (m/s^2)
k_penalty = 100.0        # Penalty stiffness for joint angle limits

# --- PENALTY TORQUE FUNCTION ---
def penalty_torque(theta, theta_min, theta_max, k=100.0):
    """
    Creates a soft restoring torque if theta exceeds its limits.
    Returns zero if theta is within bounds.
    """
    return -k * max(0, theta - theta_max) + k * max(0, theta_min - theta)

# --- SYSTEM DYNAMICS ---
def dynamics(t, y):
    """
    Returns the derivatives of the state [theta1, theta2, dtheta1, dtheta2].
    Includes gravitational effects and soft joint limits.
    """
    theta1, theta2, dtheta1, dtheta2 = y

    # Apply penalty torques if angles exceed limits
    tau1 = penalty_torque(theta1, -np.pi/2, np.pi/2, k_penalty)
    tau2 = penalty_torque(theta2, -np.pi/2, np.pi/2, k_penalty)

    # Mass/inertia matrix terms
    M11 = m1 * (l1**2) / 3 + m2 * (l1**2 + l2**2 / 3 + l1 * l2 * np.cos(theta2))
    M12 = m2 * (l2**2 / 3 + l1 * l2 / 2 * np.cos(theta2))
    M21 = M12
    M22 = m2 * l2**2 / 3
    M = np.array([[M11, M12], [M21, M22]])

    # Coriolis and gravity terms
    C1 = -m2 * l1 * l2 / 2 * np.sin(theta2) * dtheta2**2 - (m1 * l1 / 2 + m2 * l1) * g * np.sin(theta1)
    C2 = m2 * l1 * l2 / 2 * np.sin(theta2) * dtheta1**2 - m2 * l2 / 2 * g * np.sin(theta1 + theta2)
    C = np.array([C1, C2])

    # Net torques: control input - nonlinear forces
    tau = np.array([tau1, tau2])
    ddtheta = np.linalg.solve(M, tau - C)

    return [dtheta1, dtheta2, ddtheta[0], ddtheta[1]]

# --- SIMULATION ---
y0 = [np.pi / 4, np.pi / 4, 0.0, 0.0]         # Initial state
t_span = (0, 10)                              # Simulation time span (seconds)
t_eval = np.linspace(*t_span, 100)            # Evaluation time points
sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval)

# --- FORWARD KINEMATICS ---
theta1, theta2 = sol.y[0], sol.y[1]
x1 = l1 * np.cos(theta1)
y1 = l1 * np.sin(theta1)
x2 = x1 + l2 * np.cos(theta1 + theta2)
y2 = y1 + l2 * np.sin(theta1 + theta2)

# --- ANIMATION FRAMES ---
frames = []
for i in range(len(t_eval)):
    frames.append(go.Frame(data=[
        go.Scatter(x=[0, x1[i], x2[i]],
                   y=[0, y1[i], y2[i]],
                   mode='lines+markers',
                   line=dict(color='blue', width=4),
                   marker=dict(size=8, color='red'))
    ]))

# --- INITIAL PLOT ---
fig = go.Figure(
    data=[go.Scatter(x=[0, x1[0], x2[0]],
                     y=[0, y1[0], y2[0]],
                     mode='lines+markers',
                     line=dict(color='blue', width=4),
                     marker=dict(size=8, color='red'))],
    layout=go.Layout(
        title="🔧 2-DOF Robotic Arm with Soft Joint Limits",
        xaxis=dict(range=[-2, 2], title='X', scaleanchor="y"),
        yaxis=dict(range=[-2, 2], title='Y'),
        updatemenus=[dict(type='buttons',
                          buttons=[dict(label='▶ Play',
                                        method='animate',
                                        args=[None])])]),
    frames=frames
)

fig.show()
