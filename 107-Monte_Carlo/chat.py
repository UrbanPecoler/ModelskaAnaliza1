import numpy as np
import matplotlib.pyplot as plt

def simulate_neutron_escape_with_angles(num_samples, lam):
    pobegli = 0
    n_sipanj = np.zeros(num_samples)
    angles = []  # Shranjevanje kotov ob pobegu

    for _ in range(num_samples):
        # Začetna pozicija znotraj plošče
        x = np.random.uniform(0, 1)
        y = 0
        z = 0

        while 0 < x < 1:
            # Generacija naključnih kotov
            xi2 = np.random.rand()
            xi3 = np.random.rand()
            cos_theta = 2 * xi2 - 1
            theta = np.arccos(cos_theta)
            phi = 2 * np.pi * xi3
            
            # Generacija naključnega koraka in posodobitev položaja
            step = np.random.exponential(lam)
            dx = step * np.sin(theta) * np.cos(phi)
            dy = step * np.sin(theta) * np.sin(phi)
            dz = step * cos_theta
            
            x += dx
            # print(x)
            y += dy
            z += dz


        # Pretvorba kota v ustrezno območje
        r = np.sqrt(y**2 + z**2)
        if x>=1:
            angle = np.arctan(z/x)
        if x<=0:
            angle = np.arctan(-z/x)
        angles.append(angle)
        #print("DONE")

    return angles

# Parametri simulacije
num_samples = 50000
lam = 0.5

# Simulacija
angles = simulate_neutron_escape_with_angles(num_samples, lam)
# print(angles)
# Histogram normaliziran na gostoto
bin_edges = np.linspace(-np.pi, np.pi, 100)
hist, edges = np.histogram(angles, bins=bin_edges)

# Risanje grafa
plt.figure(figsize=(10, 6))
plt.bar(
    (edges[:-1] + edges[1:]) / 2, hist,
    width=np.diff(edges), align="center",
    color="blue", alpha=0.7, edgecolor="black"
)
plt.title(r"Porazdelitev kotov pobega nevtronov ($\mu=0.5$)", fontsize=16)
plt.xlabel(r"$\theta$", fontsize=14)
plt.ylabel(r"$P / \Delta \theta$", fontsize=14)
plt.grid(True)
plt.show()
