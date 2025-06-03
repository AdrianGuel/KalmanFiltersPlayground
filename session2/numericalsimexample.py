# A simple 2DOF robotic arm example for numerical methods in python:
# Adrian Guel 2025

## info about the used numerical methods:
# RK45 — Dormand-Prince Method
#     Order: 5th-order accuracy with 4th-order error estimation.
#     Name: Often referred to as "DOPRI5" (Dormand–Prince 1980).
#     Purpose: Very popular in many scientific applications due to its balance of speed and accuracy.
#     Mechanism: Uses 6 function evaluations per step to compute both a 5th-order estimate and a 4th-order estimate of the solution, which allows for adaptive step size control.

#  RK23 — Bogacki–Shampine Method
#     Order: 3rd-order accuracy with 2nd-order error estimation.
#     Name: Sometimes called BS23 (Bogacki–Shampine 1989).
#     Purpose: Less accurate but faster for problems that don’t require high precision.
#     Mechanism: Computes a 3rd-order estimate and compares it to a 2nd-order estimate for adaptive step sizing.

import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# --- SYSTEM PARAMETERS ---
l1, l2 = 1.0, 1.0
m1, m2 = 1.0, 1.0
g = 9.81
k_penalty = 100.0

# --- PENALTY TORQUE FUNCTION ---
def penalty_torque(theta, theta_min, theta_max, k=100.0):
    return -k * max(0, theta - theta_max) + k * max(0, theta_min - theta)

# --- SYSTEM DYNAMICS ---
def dynamics(t, y):
    theta1, theta2, dtheta1, dtheta2 = y
    tau1 = penalty_torque(theta1, -np.pi/2, np.pi/2, k_penalty)
    tau2 = penalty_torque(theta2, -np.pi/2, np.pi/2, k_penalty)

    M11 = m1 * l1**2 / 3 + m2 * (l1**2 + l2**2 / 3 + l1 * l2 * np.cos(theta2))
    M12 = m2 * (l2**2 / 3 + l1 * l2 / 2 * np.cos(theta2))
    M21 = M12
    M22 = m2 * l2**2 / 3
    M = np.array([[M11, M12], [M21, M22]])

    C1 = -m2 * l1 * l2 / 2 * np.sin(theta2) * dtheta2**2 - (m1 * l1 / 2 + m2 * l1) * g * np.sin(theta1)
    C2 = m2 * l1 * l2 / 2 * np.sin(theta2) * dtheta1**2 - m2 * l2 / 2 * g * np.sin(theta1 + theta2)
    C = np.array([C1, C2])

    tau = np.array([tau1, tau2])
    ddtheta = np.linalg.solve(M, tau - C)
    return [dtheta1, dtheta2, ddtheta[0], ddtheta[1]]

# --- SIMULATION SETUP ---
y0 = [np.pi / 4, np.pi / 4, 0.0, 0.0]
t_span = (0, 10)
t_eval = np.linspace(*t_span, 100)

# Solve using RK45 and RK23
sol_rk45 = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, method='RK45')
sol_rk23 = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, method='RK23')

theta1_rk45, theta2_rk45 = sol_rk45.y[0], sol_rk45.y[1]
theta1_rk23, theta2_rk23 = sol_rk23.y[0], sol_rk23.y[1]

# --- PLOTLY ANGLE COMPARISON ---
angle_plot = go.Figure()
angle_plot.add_trace(go.Scatter(x=t_eval, y=theta1_rk45, mode='lines', name='θ1 RK45', line=dict(color='green')))
angle_plot.add_trace(go.Scatter(x=t_eval, y=theta2_rk45, mode='lines', name='θ2 RK45', line=dict(color='orange')))
angle_plot.add_trace(go.Scatter(x=t_eval, y=theta1_rk23, mode='lines', name='θ1 RK23', line=dict(color='green', dash='dash')))
angle_plot.add_trace(go.Scatter(x=t_eval, y=theta2_rk23, mode='lines', name='θ2 RK23', line=dict(color='orange', dash='dash')))

angle_plot.update_layout(
    title='Comparison of Joint Angles: RK45 vs RK23',
    xaxis_title='Time (s)',
    yaxis_title='Angle (rad)',
    height=500
)

# --- ANIMATION USING RK45 ---
x1 = l1 * np.cos(theta1_rk45)
y1 = l1 * np.sin(theta1_rk45)
x2 = x1 + l2 * np.cos(theta1_rk45 + theta2_rk45)
y2 = y1 + l2 * np.sin(theta1_rk45 + theta2_rk45)

frames = []
for i in range(len(t_eval)):
    frames.append(go.Frame(data=[
        go.Scatter(x=[0, x1[i], x2[i]],
                   y=[0, y1[i], y2[i]],
                   mode='lines+markers',
                   line=dict(color='blue', width=4),
                   marker=dict(size=8, color='red'))
    ]))

arm_plot = go.Figure(
    data=[go.Scatter(x=[0, x1[0], x2[0]],
                     y=[0, y1[0], y2[0]],
                     mode='lines+markers',
                     line=dict(color='blue', width=4),
                     marker=dict(size=8, color='red'))],
    layout=go.Layout(
        title="2-DOF Robotic Arm Animation (RK45)",
        xaxis=dict(range=[-2, 2], title='X', scaleanchor="y"),
        yaxis=dict(range=[-2, 2], title='Y'),
        updatemenus=[dict(type='buttons',
                          buttons=[dict(label='▶ Play',
                                        method='animate',
                                        args=[None])])]),
    frames=frames
)

# --- DISPLAY ---
angle_plot.show()
arm_plot.show()


