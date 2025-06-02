import numpy as np
import plotly.graph_objects as go

# === Define parameters ===
ybar = np.array([[1], [2]])  # Mean vector (2x1)
covar = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix (2x2)
n_samples = 5000

# === Cholesky Decomposition ===
A_chol = np.linalg.cholesky(covar)
x_chol = np.random.randn(2, n_samples)
y_chol = ybar + A_chol @ x_chol

# === LDL Decomposition ===
# LDLT via scipy's linalg.ldl (or manual LDL decomposition for symmetric matrices)
def ldl_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    D = np.zeros_like(A)
    for j in range(n):
        D[j, j] = A[j, j] - L[j, :j] @ D[:j, :j] @ L[j, :j].T
        for i in range(j+1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ D[:j, :j] @ L[j, :j].T) / D[j, j]
    return L, D

L_ldl, D_ldl = ldl_decomposition(covar)
x_ldl = np.random.randn(2, n_samples)
y_ldl = ybar + (L_ldl @ np.sqrt(D_ldl)) @ x_ldl


# === Fit bivariate Gaussian ===
fitted_mean = np.mean(y_chol, axis=1)
fitted_cov = np.cov(y_chol)

# === Confidence Ellipse ===
def get_cov_ellipse(cov, center, n_std=1.0, num_points=100):
    """
    Returns points of an ellipse representing the n_std confidence interval.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])  # shape: (2, N)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes_lengths = n_std * np.sqrt(eigvals)
    transform = eigvecs @ np.diag(axes_lengths)
    ellipse = transform @ unit_circle + center[:, None]
    return ellipse

ellipse = get_cov_ellipse(fitted_cov, fitted_mean, n_std=1.0)

# === Plot using Plotly ===
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_chol[0, :], y=y_chol[1, :],
    mode='markers',
    marker=dict(size=2, color='blue'),
    name='Cholesky'
))

fig.add_trace(go.Scatter(
    x=y_ldl[0, :], y=y_ldl[1, :],
    mode='markers',
    marker=dict(size=2, color='red'),
    name='LDL'
))

# Confidence ellipse (1σ)
fig.add_trace(go.Scatter(
    x=ellipse[0], y=ellipse[1],
    mode='lines',
    line=dict(color='black', width=2),
    name='1σ Ellipse (Gaussian Fit)'
))

fig.update_layout(
    title="Correlated Random Vectors via Cholesky and LDL",
    xaxis_title="Variable 1",
    yaxis_title="Variable 2",
    legend=dict(x=0.8, y=0.95)
)

fig.show()
