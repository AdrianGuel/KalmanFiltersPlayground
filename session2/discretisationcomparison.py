import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, inv

# RLC parameters
R = 1.0
L = 1.0
C = 1.0
T = 0.5  # Sampling time
N = 100  # Time steps

# Continuous-time system
A = np.array([[-1 / (R * C), 1 / C],
              [-1 / L,       0]])
B = np.array([[0],
              [1 / L]])
C_mat = np.array([1, 0])  # output is capacitor voltage

# Euler discretization
Ad_euler = np.eye(2) + T * A
Bd_euler = T * B

# Exact discretization
Ad_exact = expm(A * T)
try:
    Bd_exact = inv(A) @ (Ad_exact - np.eye(2)) @ B
except np.linalg.LinAlgError:
    Bd_exact = np.zeros_like(B)
    print("Matrix A is singular, cannot compute Bd_exact analytically.")

# Input
u = np.ones(N)

# Simulation storage
x_euler = np.zeros((2, N))
x_exact = np.zeros((2, N))

# Simulate both systems
for k in range(1, N):
    x_euler[:, k] = Ad_euler @ x_euler[:, k-1] + Bd_euler.flatten() * u[k-1]
    x_exact[:, k] = Ad_exact @ x_exact[:, k-1] + Bd_exact.flatten() * u[k-1]

# Plot comparison
plt.plot(x_euler[0], label="Euler Discretization", linestyle="--")
plt.plot(x_exact[0], label="Exact Discretization", linestyle="-")
plt.xlabel("Time step")
plt.ylabel("Capacitor Voltage (V)")
plt.title("Discretization Comparison: Series-Inductor + Parallel RC Circuit")
plt.legend()
plt.grid()
plt.show()
