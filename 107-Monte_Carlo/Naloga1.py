import math
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time

from tqdm import tqdm

def point_gen(method="kocka"):
    if method == "kocka":
        return np.random.uniform(-1, 1, 3)
    
    elif method == "krogla":
        while True:
            radius = 1
            x, y, z = np.random.uniform(-1, 1, 3)
            if x**2 + y**2 + z**2 <= 1:
                return x * radius, y * radius, z * radius

def is_inside(*points):
    x = y = z = 0
    if len(points) == 3:
        x, y, z = points
    elif len(points) == 1 and isinstance(points[0], tuple):
        x, y, z = points[0]
    return math.sqrt(abs(x)) + math.sqrt(abs(y)) + math.sqrt(abs(z)) <= 1

def plot_MC(n):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    stevec  = 0
    for i in tqdm(range(1,n)):
        x, y, z = np.random.uniform(-1, 1, 3)
        if is_inside(x, y, z):
            stevec += 1
            ax.scatter(x,y,z,c="green", s=4)

        # else:
        #     ax.scatter(x,y,z, c="gray", s=4, alpha=0.2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")        
    plt.show()
    

# plot_MC(100000)
# ==========================================
# KOCKA
# TODO
# Lepši grafi, osi, naslovi, legende, hlines analitičnih vrednosti
# Težišče za y in z os
# Za kroglo, daj vse grafe na en plot, napako pa nariši skupaj z napako kocke

# method = "krogla"

# nmin = 3
# nmax = 7
# Ns = np.logspace(nmin, nmax, num=100)

# # Izračun mase
# mase = []
# st_devs = []
# prava_masa = 0.08888889

# # Izračun težišča
# x_tezisce = []
# y_tezisce = []
# z_tezisce = []

# # Izračun vztrajnostnega momenta
# pravi_moment = 0.004232
# vztraj_mom_z = []
# vztraj_mom_x = []
# vztraj_mom_y = []


# for n in tqdm(Ns):
#     stevec  = 0
#     inertia_z = 0
#     inertia_x = 0
#     inertia_y = 0
#     x_vals = []
#     y_vals = []
#     z_vals = []
#     for i in range(1, int(n)):
#         points = point_gen(method=method)
#         if is_inside(*points):
#             x, y, z = points
#             stevec += 1
#             x_vals.append(x)
#             y_vals.append(y)
#             z_vals.append(z)
#             inertia_z += x**2 + y**2
#             inertia_x += z**2 + y**2
#             inertia_y += x**2 + z**2

#     razmerje = stevec/n
    
#     if method == "kocka":
#         masa = 8 * razmerje

#     elif method == "krogla":
#         masa = 4/3 * np.pi * razmerje

#     st_dev = np.sqrt((razmerje*(1-razmerje))/n)
#     mase.append(masa)
#     st_devs.append(st_dev)

#     x_tezisce.append(np.mean(x_vals))
#     y_tezisce.append(np.mean(y_vals))
#     z_tezisce.append(np.mean(z_vals))

#     inertia_z /= n
#     vztraj_mom_z.append((inertia_z * 10) - 0.006)

#     inertia_x /= n
#     vztraj_mom_x.append((inertia_x * 10) - 0.006)

#     inertia_y /= n
#     vztraj_mom_y.append((inertia_y * 10) - 0.006)

# cmap = plt.get_cmap("viridis")
# color_x = cmap(0)
# color_y = cmap(1/3)
# color_z = cmap(2/3)

# plt.errorbar(Ns, mase, yerr=st_devs, fmt="o", markersize=2)
# plt.hlines(prava_masa, xmin=10**nmin, xmax=10**nmax, color="red", linestyles="dashed")
# plt.xscale("log")
# plt.title("Izračun mase")
# plt.xlabel("N")
# plt.ylabel("Masa")
# if method == "kocka":
#     plt.savefig("./Images/Masa_kocka")
# elif method == "krogla":
#     plt.savefig("./Images/Masa_krogla")
# plt.show()

# fig, ax1 = plt.subplots(figsize=(8, 6))

# line1, = ax1.plot(Ns, np.abs(np.array(mase) - prava_masa), label="Absolutna napaka", color="blue")
# ax1.set_xscale("log")
# ax1.set_xlabel("N")
# ax1.set_ylabel("Abs. error", color="tab:blue")
# ax1.tick_params(axis='y', labelcolor='tab:blue')


# ax2 = ax1.twinx()
# line2, = ax2.plot(Ns, st_devs, label=r"$\sigma$", color="orange")
# ax2.set_ylabel(r"$\sigma$", color='tab:orange')
# ax2.tick_params(axis='y', labelcolor='tab:orange')


# plt.title("Napaka mase")
# fig.tight_layout()
# ax1.legend(handles=[line1], loc="upper left")
# ax2.legend(handles=[line2], loc="upper right")
# if method == "kocka":
#     plt.savefig("./Images/Napaka-kocka")
# elif method == "krogla":
#     plt.savefig("./Images/Napaka-krogla")
# plt.show()

# plt.plot(Ns, x_tezisce, "o-", label="x-koordinata", color=color_x)
# plt.plot(Ns, y_tezisce, "o-", label="y-koordinata", color=color_y)
# plt.plot(Ns, z_tezisce, "o-", label="z-koordinata", color=color_z)
# plt.hlines(0, xmin=10**nmin, xmax=10**nmax, color="red", linestyles="dashed")
# plt.xscale("log")
# plt.title("Težišče")
# plt.xlabel("N")
# plt.ylabel("Vrednost koordinate")
# plt.legend()
# if method == "kocka":
#     plt.savefig("./Images/tezisce-kocka")
# elif method == "krogla":
#     plt.savefig("./Images/tezisce-krogla")
# plt.show()

# plt.plot(Ns, vztraj_mom_z,  "o-", label=r"$I_{zz}$", color=color_z)
# plt.plot(Ns, vztraj_mom_x, "o-", label=r"$I_{xx}$", color=color_x)
# plt.plot(Ns, vztraj_mom_y, "o-", label=r"$I_{yy}$", color=color_y)
# plt.hlines(pravi_moment, xmin=10**nmin, xmax=10**nmax, color="red", linestyles="dashed", label="Analitična vrednost")
# plt.xscale("log")
# plt.title("Vztrajnostni moment")
# plt.xlabel("N")
# plt.ylabel("I")
# plt.legend()
# if method == "kocka":
#     plt.savefig("./Images/Inertia-kocka")
# elif method == "krogla":
#     plt.savefig("./Images/Inertia-krogla")
# plt.show()


# # ČASOVNA ZAHTEVNOST

# time_kocka = []
# time_krogla = []
# Ns = np.logspace(nmin, nmax, num=30)

# for n in tqdm(Ns):
#     stevec  = 0

#     time_start = time.time()
#     for i in range(1, int(n)):
#         points = point_gen(method="kocka")
#         if is_inside(*points):
#             stevec += 1
#     time_kocka.append(time.time() - time_start)
    
#     time_start = time.time()
#     for i in range(1, int(n)):
#         points = point_gen(method="krogla")
#         if is_inside(*points):
#             stevec += 1
#     time_krogla.append(time.time() - time_start)

# plt.plot(Ns, time_kocka, label="Kocka")
# plt.plot(Ns, time_krogla, label="krogla")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("N")
# plt.ylabel("time")
# plt.legend()
# plt.savefig("./Images/time_complexity")
# plt.show()


# ================================================================
# ODVISNOST OD GOSTOTE
# TODO
# Uredi grafe
# Nariši krivulje za N = 10^3, 10^4, 10^5, 10^6

num_points = 100000  # Number of random points to generate for Monte Carlo
p_values = np.linspace(0, 6, 50)  # Different values of p
method = 'kocka' 

# Results storage
masses = []
st_devs = []

x_centroids = []

moment_inertias = []

for p in tqdm(p_values):
    counter = 0
    inertia_z = 0
    weighted_x_sum = 0
    total_mass = 0
    mass = 0

    for _ in range(num_points):
        points = point_gen(method=method)
        if is_inside(*points):
            x, y, z = points
            r = np.sqrt(x**2 + y**2 + z**2)
            density = r**p 

            counter += 1
            mass += density
            weighted_x_sum += x * mass
            inertia_z += (x**2 + y**2) * mass
            total_mass += mass

    avg_mass_ratio = counter / num_points
    estimated_mass = 8 * mass / num_points 
    std_dev = np.sqrt((avg_mass_ratio * (1 - avg_mass_ratio)) / num_points)

    centroid_x = weighted_x_sum / total_mass if total_mass != 0 else 0
    moment_of_inertia = inertia_z * (estimated_mass / total_mass) if total_mass != 0 else 0

    # Store results
    masses.append(estimated_mass)
    st_devs.append(std_dev)
    x_centroids.append(centroid_x)
    moment_inertias.append(moment_of_inertia)

plt.plot(p_values, masses, "o", markersize=2)
# plt.yscale("log")
plt.xlabel("p")
plt.ylabel("masa")
plt.title("Masa v odvisnosti od p")
plt.savefig("./Images/p_masa")
plt.show()

plt.plot(p_values, moment_inertias, "o", markersize=2)
# plt.yscale("log")
plt.title("Vztrajnostni moment v odvisnosti od p")
plt.xlabel("p")
plt.ylabel("Vztrajnostni moment")
plt.savefig("./Images/p_inertia")
plt.show()