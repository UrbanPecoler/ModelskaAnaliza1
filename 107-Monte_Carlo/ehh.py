import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Izotropni
def run_single():
    r = np.array([0, 0, 0], np.float64)
    dir = np.array([0, 0, 1], np.float64)
    costheta = 1
    phi = 0

    i = 0
    while True:
        s = -lp * np.log(np.random.random())
        r += s * dir

        if r[2] < 0:  # Sipanje nazaj
            return False, [r, phi, np.arccos(costheta)], i
        if r[2] > d:  # Prepuščanje
            return True, [r, phi, np.arccos(costheta)], i

        i += 1

        phi = 2 * np.pi * np.random.random()
        costheta = 2 * np.random.random() - 1
        sintheta = np.sqrt(1 - costheta**2)

        dir = np.array([sintheta * np.cos(phi), sintheta * np.sin(phi), costheta])

def run_N(N):
    data = []
    endstates = []
    bounces = []
    for _ in tqdm(range(N), leave=False):
        T, pos, n = run_single()
        data.append(int(T))
        endstates.append(pos)
        bounces.append(n)
    
    return data, endstates, bounces

# Parametri simulacije
alpha = 1 / 2
beta = 1 / 2
d = 1
lp = 0.2
N = 10**4

# Simulacija
data, endstates, bounces = run_N(N)

# Theta vrednosti
thetas = np.array([endstate[-1] for endstate in endstates])  # Izračunaj theta iz končnih stanj

# Prepuščeni (T = True) in sipani (T = False)
passed_thetas = thetas[np.array(data) == 1]  # Prepuščeni nevtroni
scattered_thetas = thetas[np.array(data) == 0]  # Sipani nazaj

# Histogram za prepuščene nevtrone (zelena)
bin_edges = np.linspace(0, np.pi, 40)
passed_hist, _ = np.histogram(passed_thetas, bins=bin_edges)
scattered_hist, _ = np.histogram(scattered_thetas, bins=bin_edges)

# Normalizacija
total = len(thetas)
passed_hist = passed_hist / total
scattered_hist = scattered_hist / total

# Sredine binov za izris
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Izris histogramov
plt.bar(bin_centers, passed_hist, width=bin_edges[1] - bin_edges[0], color="green", edgecolor="black", alpha=0.7, label="Prepuščeni")
plt.bar(bin_centers, scattered_hist, width=bin_edges[1] - bin_edges[0], color="red", edgecolor="black", alpha=0.7, label="Sipani nazaj")

# Oznake
plt.xlabel(r"$\theta$")
plt.ylabel("Delež nevtronov")
plt.title(f"$\lambda= {lp}$")
plt.legend()
plt.savefig("./Images/hist_0_2")
plt.show()
