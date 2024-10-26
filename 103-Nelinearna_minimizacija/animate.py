import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def distance_on_sphere(p1, p2):
    return np.linalg.norm(p1 - p2)

def potential_energy(params, N):
    energy = 0
    points = []

    for i in range(N):
        theta = params[2*i]     # Latitude (0 <= theta <= pi)
        phi = params[2*i + 1]   # Longitude (0 <= phi <= 2*pi)
        points.append(spherical_to_cartesian(theta, phi))
    
    for i in range(N):
        for j in range(i+1, N):
            energy += 1.0 / distance_on_sphere(points[i], points[j])
    
    return energy

N = 5

initial_guess = np.random.rand(2 * N)
initial_guess[::2] *= np.pi      # theta in [0, pi]
initial_guess[1::2] *= 2 * np.pi # phi in [0, 2*pi]

result = minimize(potential_energy, initial_guess, args=(N,), method='BFGS')
optimal_angles = result.x

initial_points = [spherical_to_cartesian(initial_guess[2*i], initial_guess[2*i+1]) for i in range(N)]
final_points = [spherical_to_cartesian(optimal_angles[2*i], optimal_angles[2*i+1]) for i in range(N)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 15) 
v = np.linspace(0, np.pi, 8)       
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_wireframe(x, y, z, color='b', linewidth=0.5, alpha=0.6)  

scatter = ax.scatter([], [], [], color='r', s=100)
lines = []

def interpolate_points(t):
    points = []
    for i in range(N):
        x = (1 - t) * initial_points[i][0] + t * final_points[i][0]
        y = (1 - t) * initial_points[i][1] + t * final_points[i][1]
        z = (1 - t) * initial_points[i][2] + t * final_points[i][2]
        points.append([x, y, z])
    return points

def update(frame):
    global lines
    t = frame / 100  

    interpolated_points = interpolate_points(t)
    scatter._offsets3d = ([p[0] for p in interpolated_points],
                          [p[1] for p in interpolated_points],
                          [p[2] for p in interpolated_points])

    # Remove old lines
    for line in lines:
        line.remove()
    lines = []

    for i in range(N):
        for j in range(i + 1, N):
            x_line = [interpolated_points[i][0], interpolated_points[j][0]]
            y_line = [interpolated_points[i][1], interpolated_points[j][1]]
            z_line = [interpolated_points[i][2], interpolated_points[j][2]]
            line = ax.plot(x_line, y_line, z_line, color='k', linewidth=0.8)  
            lines.append(line[0])

    return (scatter,) + tuple(lines)

ax.set_box_aspect([1, 1, 1]) 

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=False)

plt.show()
