import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Parameters
Nx = 1       # state dimension
Nxa = 3      # augmented state (state + process + sensor noise)
Nz = 1       # measurement dimension
h = np.sqrt(3)

# SPKF Weights
Wmx = np.zeros(2 * Nxa + 1)
Wcx = np.zeros_like(Wmx)
Wmx[0] = (h**2 - Nxa) / h**2
Wmx[1:] = 1 / (2 * h**2)
Wcx[:] = Wmx
Wmxz = Wmx.reshape(-1, 1)

# Noise covariances
SigmaW = 1.0
SigmaV = 2.0
maxIter = 40

# Initial states
xtrue = 2 + np.random.randn()
xhat = 2.0
SigmaX = 1.0

# Data storage
xstore = [xtrue]
xhatstore = []
SigmaXstore = []

# SPKF loop
for k in range(maxIter):
    # Step 1a: Augmented state and Cholesky
    xhata = np.array([xhat, 0.0, 0.0])
    Pxa = np.diag([SigmaX, SigmaW, SigmaV])
    sPxa = np.linalg.cholesky(Pxa)

    # Sigma points
    X = xhata[:, np.newaxis] + h * np.hstack([np.zeros((Nxa, 1)), sPxa, -sPxa])

    # Step 1a-iv: Predict state
    Xx = np.sqrt(5 + X[0, :]) + X[1, :]
    xhat = np.dot(Wmx, Xx)

    # Step 1b: Predict covariance
    Xs = (Xx[1:] - xhat) * np.sqrt(Wcx[1])
    Xs1 = Xx[0] - xhat
    SigmaX = Xs @ Xs.T + Wcx[0] * Xs1**2

    # Simulate system
    w = np.random.randn() * np.sqrt(SigmaW)
    v = np.random.randn() * np.sqrt(SigmaV)
    ztrue = xtrue**3 + v
    xtrue = np.sqrt(5 + xtrue) + w

    # Step 1c: Predict measurement
    Z = Xx**3 + X[2, :]
    zhat = np.dot(Wmx, Z)

    # Step 2a: Gain matrix
    Zs = (Z[1:] - zhat) * np.sqrt(Wcx[1])
    Zs1 = Z[0] - zhat
    SigmaXZ = Xs @ Zs.T + Wcx[0] * Xs1 * Zs1
    SigmaZ = Zs @ Zs.T + Wcx[0] * Zs1**2
    Lx = SigmaXZ / SigmaZ

    # Step 2b: Update state
    xhat = xhat + Lx * (ztrue - zhat)

    # Step 2c: Update covariance
    SigmaX = SigmaX - Lx * SigmaZ * Lx

    # Store results
    xstore.append(xtrue)
    xhatstore.append(xhat)
    SigmaXstore.append(SigmaX)

# Convert results to arrays
xstore = np.array(xstore)
xhatstore = np.array(xhatstore)
SigmaXstore = np.array(SigmaXstore)
iterations = np.arange(maxIter)

# Plot
fig = make_subplots(rows=2, cols=1, subplot_titles=("Sigma-Point Kalman Filter in Action", "Estimation Error with Bounds"))

# Top plot: true vs estimate with bounds
fig.add_trace(go.Scatter(x=iterations, y=xstore[:-1], mode='lines', name='True State'), row=1, col=1)
fig.add_trace(go.Scatter(x=iterations, y=xhatstore, mode='lines', name='Estimate'), row=1, col=1)
fig.add_trace(go.Scatter(x=iterations, y=xhatstore + 3*np.sqrt(SigmaXstore), mode='lines', name='+3σ', line=dict(dash='dot')), row=1, col=1)
fig.add_trace(go.Scatter(x=iterations, y=xhatstore - 3*np.sqrt(SigmaXstore), mode='lines', name='-3σ', line=dict(dash='dot')), row=1, col=1)

# Bottom plot: estimation error
fig.add_trace(go.Scatter(x=iterations, y=xstore[:-1] - xhatstore, mode='lines', name='Error'), row=2, col=1)
fig.add_trace(go.Scatter(x=iterations, y=3*np.sqrt(SigmaXstore), mode='lines', name='±3σ Bound', line=dict(dash='dot')), row=2, col=1)
fig.add_trace(go.Scatter(x=iterations, y=-3*np.sqrt(SigmaXstore), mode='lines', showlegend=False, line=dict(dash='dot')), row=2, col=1)

fig.update_layout(height=700, title_text="SPKF Example: Nonlinear System State Estimation", showlegend=True)
fig.update_xaxes(title_text="Iteration", row=1, col=1)
fig.update_yaxes(title_text="State", row=1, col=1)
fig.update_xaxes(title_text="Iteration", row=2, col=1)
fig.update_yaxes(title_text="Estimation Error", row=2, col=1)
fig.show()
