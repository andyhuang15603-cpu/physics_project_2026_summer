'''
The Schrodinger equation under the representation of density matrix is given by:

\frac{\partial \rho}{\partial t} = -\frac{i}{\hbar} [H, \rho].

The initial state is given by:
\rho(0) = \frac{1}{2} \begin{bmatrix}
1 + \cos \Theta & \sin \Theta \\
\sin \Theta & 1 - \cos \Theta
\end{bmatrix}.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

Omega = 2 * np.pi * 0.5
Theta = np.deg2rad(30)

sigma_x = np.array([
    [0, 1],
    [1, 0]
], dtype = complex)
sigma_y = np.array([
    [0, -1j],
    [1j, 0]
], dtype = complex)
sigma_z = np.array([
    [1, 0],
    [0, -1]
], dtype = complex)

H = 0.5 * Omega * sigma_z
# Moving \hbar^{-1} into the definition of the Hanmiltonian for simplicity.

def Equation(t, rho_flat):
    rho = rho_flat.reshape(2, 2)
    drho_dt = -1j * (np.dot(H, rho) - np.dot(rho, H))
    return drho_dt.flatten()

psi_0 = np.array([np.cos(Theta / 2), np.sin(Theta / 2)], dtype = complex)
rho_0 = np.outer(psi_0, psi_0.conj())

t_max = 12.0
t_eval = np.linspace(0, t_max, 300)

solution = solve_ivp(Equation, [0, t_max], rho_0.flatten(), t_eval = t_eval)

sx_expect = []
sy_expect = []
sz_expect = []

for idx in range(len(t_eval)):
    rho_t = solution.y[:, idx].reshape(2, 2)
    sx_expect.append(np.trace(np.dot(rho_t, sigma_x)).real)
    sy_expect.append(np.trace(np.dot(rho_t, sigma_y)).real)
    sz_expect.append(np.trace(np.dot(rho_t, sigma_z)).real)

fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(t_eval, sx_expect, label = '$S_x(t)$', color = 'green')
ax.plot(t_eval, sy_expect, label = '$S_y(t)$', color = 'blue')
ax.plot(t_eval, sz_expect, label = '$S_z(t)$', color = 'orange')
ax.set_xlabel('Time ($t$)')
ax.set_ylabel('Expectation Value')
ax.set_title('Expectation Values of Spin Components')
ax.legend()
ax.grid(True)
plt.show()