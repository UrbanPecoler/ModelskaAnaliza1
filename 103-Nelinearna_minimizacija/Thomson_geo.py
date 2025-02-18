import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# SAVE FIGURE
save = False
# =========================================================================

# KROG
# TODO 
def potential_energy(thetas):
    N = len(thetas)
    energy = 0
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi = np.cos(thetas[i]), np.sin(thetas[i])
            xj, yj = np.cos(thetas[j]), np.sin(thetas[j])

            rij = np.sqrt((xi - xj)**2 + (yi - yj)**2)

            energy += 1 / rij
    return energy

# Število nabojev
# N = 3  
Ns = np.array([2, 3, 5, 7, 9, 12])

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
ax = ax.flatten()

for idx, N in enumerate(Ns):
    initial_thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)

    result = minimize(potential_energy, initial_thetas, method='BFGS')

    optimal_thetas = result.x

    x_coords = np.cos(optimal_thetas)
    y_coords = np.sin(optimal_thetas)

    circle = plt.Circle((0, 0), 1, color='lightgreen', alpha=0.5)
    ax[idx].add_artist(circle)

    ax[idx].plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)), alpha=0.6)  # Unit circle
    ax[idx].plot(x_coords, y_coords, 'o', color="blue", markersize=10, label='Charges')
    ax[idx].set_aspect('equal')
    ax[idx].set_xlim(-1.5, 1.5)
    ax[idx].set_ylim(-1.5, 1.5)
    ax[idx].set_title(f'N = {N}')
    ax[idx].legend()

fig.tight_layout()
if save:
    plt.savefig("./Images/2dkrog")
plt.show()


# ELEKTRIČNO POLJE
def electric_field(x, y, charges, positions):
    Ex, Ey = 0, 0
    for q, (xi, yi) in zip(charges, positions):
        dx, dy = x - xi, y - yi
        r = np.sqrt(dx**2 + dy**2)
        if r > 1e-9:
            Ex += q * dx / r**3
            Ey += q * dy / r**3
    return Ex, Ey


def electric_potential(x, y, charges, positions):
    """ Zračuna električni potencial v točki (x, y) """
    V = 0
    for q, (xi, yi) in zip(charges, positions):
        dx, dy = x - xi, y - yi
        r = np.sqrt(dx**2 + dy**2)
        if r > 1e-9:
            V += q / r
    return V

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

for idx, N in enumerate(Ns):
    charges = np.ones(N)  # All charges are equal
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    positions = np.array([(np.cos(theta), np.sin(theta)) for theta in thetas])

    x_vals = np.linspace(-3, 3, 200)
    y_vals = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x_vals, y_vals)

    V = np.zeros_like(X)
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            V[i, j] = electric_potential(X[i, j], Y[i, j], charges, positions)

    contour_fill = ax[idx // 3, idx % 3].contourf(X, Y, V, levels=100, cmap='viridis', alpha=0.8)  # Colormap for potential

    ax[idx // 3, idx % 3].plot(np.cos(np.linspace(0, 2 * np.pi, 100)), 
                               np.sin(np.linspace(0, 2 * np.pi, 100)), 
                               color="black", alpha=0.6)  # Unit circle
    ax[idx // 3, idx % 3].scatter(np.cos(thetas), 
                                   np.sin(thetas), 
                                   color='red', 
                                   s=150, zorder=5, 
                                   edgecolor='black', 
                                   linewidth=1.5, 
                                   label='Charges')

    ax[idx // 3, idx % 3].set_xlim(-3, 3)
    ax[idx // 3, idx % 3].set_ylim(-3, 3)
    ax[idx // 3, idx % 3].set_aspect('equal')
    ax[idx // 3, idx % 3].set_title(f'Električni potencial za {N} nabojev', fontsize=10)
    ax[idx // 3, idx % 3].axis('off')  

plt.tight_layout()

if save:
    plt.savefig("./Images/epolje_krog")
plt.show()
