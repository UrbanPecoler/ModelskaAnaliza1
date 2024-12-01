import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Naloga1 import is_inside, point_gen

num_points = 1000000  

alpha_surface = 0.3

cube_points = np.random.uniform(-1, 1, (num_points, 3))

spherical_points = np.random.normal(size=(num_points, 3))
spherical_points /= np.linalg.norm(spherical_points, axis=1)[:, np.newaxis]
spherical_points *= np.random.uniform(0, 1, (num_points, 1))
def plot_body(ax, points, is_cube=True):
    # Mask points inside the body
    inside_mask = np.array([is_inside(x, y, z) for x, y, z in points])
    inside_points = points[inside_mask]

    # Create a 3D scatter plot
    ax.scatter(
        inside_points[:, 0], inside_points[:, 1], inside_points[:, 2],
        c='green', marker='o', alpha=0.6, label='Inside Points'
    )

    # Plot the semi-transparent bounding surface
    if is_cube:
        # Cube surface with 2D arrays for corners
        r = [-1, 1]
        X, Y = np.meshgrid(r, r)
        for s in [-1, 1]:
            Z = np.full_like(X, s)  # Create a 2D array with the same shape as X
            ax.plot_surface(X, Y, Z, color='gray', alpha=alpha_surface)  # Top and bottom
            ax.plot_surface(X, Z, Y, color='gray', alpha=alpha_surface)  # Front and back
            ax.plot_surface(Z, X, Y, color='gray', alpha=alpha_surface)  # Left and right
    else:
        # Sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=alpha_surface)


    # Plot settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

# Set up the figure
# fig = plt.figure(figsize=(12, 6))

# Plot for cube bounding box
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d') 
ax.set_title('Kocka - "Bounding Box"')
plot_body(ax, cube_points, is_cube=True)
# plt.savefig("./Images/Kocka_BB")
# plt.show()
plt.close()

# Plot for spherical bounding box
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d') 
ax.set_title('Krogla "Bounding Box"')
plot_body(ax, spherical_points, is_cube=False)
# plt.savefig("./Images/Krogla_BB")
# plt.show()
plt.close()



# ==========================================================================================
# PLOT ZA RAZLIČNE GOSTOTE
def plot_density(n):
    p_values = [1, 3, 6]  
    cmap = "viridis"


    points = np.random.uniform(-1, 1, (n, 3))
    radii = np.linalg.norm(points, axis=1)  

    fig = plt.figure(figsize=(18, 6))

    for idx, p in enumerate(p_values):
        densities = radii**p

        inside_mask = np.array([is_inside(x, y, z) for x, y, z in points])
        inside_points = points[inside_mask]
        inside_densities = densities[inside_mask]

        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        scatter = ax.scatter(
            inside_points[:, 0], inside_points[:, 1], inside_points[:, 2],
            c=inside_densities, cmap=cmap, marker='o', alpha=0.7
        )

        ax.set_title(f'p = {p}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(scatter, ax=ax, shrink=0.6, label='Gostota')

    plt.tight_layout()
    plt.savefig("./Images/Density_object")
    plt.show()

plot_density(num_points)
