from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# plt.rcParams.update({'font.size': 16})
save = False

# PARAMETRI
N=25
v_k = 1.6
v0s = [0.2, 0.4, 0.7, 1, 1.2, 1.5, 1.8, 2.0]
x = [i/N for i in range(N)]
colors = colormaps['Set2'](np.linspace(0, 1, len(v0s)))

# FUNKCIJE
def F(sez):
    """Primarna funkcija, ki bo minimizirana"""
    N = sez.shape[0]
    return np.sum((np.diff(sez))**2)/N

def min_fun(v0, pogoj="vz=vk", v_k=v_k):
    """Funkcija sprejme argument pogoj.
    vz=vk --> Minimizira funkcijo kjer sta končna in začetna vrednost hitrosti enaki
    vk --> Minimizira funkcijo, ki ima določeno končno hitrost
    Prost --> Minimizira funkcijo s prostim robnim pogojem"""

    x0 = np.random.uniform(size=N)
    
    cons = [{"type": "eq", "fun": lambda sez: 1 - np.sum(sez) / sez.shape[0]},
            {"type": "eq", "fun": lambda sez: sez[0] - v0}]

    if pogoj == "vz=vk":
        cons.append({"type": "eq", "fun": lambda sez: sez[0] - sez[-1]})
    elif pogoj == "vk":
        cons.append({"type": "eq", "fun": lambda sez: sez[-1] - v_k})
    
    res = minimize(F, x0=x0, method="SLSQP", constraints=cons)
    return res.x

# PLOTTING
# V_0 = V_K
plt.figure(figsize=(10, 5))

for i, v0 in enumerate(v0s):
    t = np.arange(0, 1, step=1/min_fun(v0, pogoj="vz=vk").shape[0])
    plt.plot(t, min_fun(v0, pogoj="vz=vk"), "o-", label=r"$v_0$ = {}".format(v0), color=colors[i])

plt.ylabel(r"$ \nu $", fontsize=16)
plt.xlabel(r"$ \tau $", fontsize=16)
plt.legend(ncol=4, frameon=True, edgecolor="k")
if save:
    plt.savefig("./Images/Semafor_v0vk")
plt.show()

# FIKSNA KONČNA HITROST
plt.figure(figsize=(10, 5))

for i, v0 in enumerate(v0s):
    t = np.arange(0, 1, step=1/min_fun(v0, pogoj="vk").shape[0])
    plt.plot(t, min_fun(v0, pogoj="vk"), "o-", label=r"$v_0$ = {}".format(v0), color=colors[i])

plt.ylabel(r"$ \nu $", fontsize=16)
plt.xlabel(r"$ \tau $", fontsize=16)
plt.legend(ncol=4, frameon=True, edgecolor="k")
if save:
    plt.savefig("./Images/Semafor_fix-vk")
plt.show()

# PROST ROBNI POGOJ
plt.figure(figsize=(10, 5))

for i, v0 in enumerate(v0s):
    t = np.arange(0, 1, step=1/min_fun(v0, pogoj="prost").shape[0])
    plt.plot(t, min_fun(v0, pogoj="prost"),"o-", label=r"$v_0$ = {}".format(v0), color=colors[i])

plt.ylabel(r"$ \nu $", fontsize=16)
plt.xlabel(r"$ \tau $", fontsize=16)
plt.legend(ncol=4, frameon=True, edgecolor="k")
if save:
    plt.savefig("./Images/Semafor_prost")
plt.show()

# =========================================================================================================
# ZVEZNA SLIKA
t = np.linspace(0, 1, 1000, endpoint=True)

def hitrost1(t, v0):
    return v0 + 3/2*(1-v0)*(2*t - t**2)

def hitrost2(t, v0, v_max):
    return 3*(2-v0-v_max)*(t - t**2) + (v_max - v0)*t + v0

def plot_zvezno(model="osnovni", save=save):
    v0s = np.array([0.2, 0.4, 0.7, 1, 1.2, 1.5, 1.8, 2.0])
    
    norm = mcolors.Normalize(vmin=v0s.min(), vmax=v0s.max())
    cmap = cm.viridis

    for v0 in v0s:
        if model.lower() == "osnovni":
            plt.plot(t, hitrost1(t, v0), color=cmap(norm(v0)))
            
        elif model.lower() == "radar zacetna":
            v_max = 1.5
            plt.plot(t, hitrost2(t, v0, v_max), color=cmap(norm(v0)))
            
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    plt.ylabel(r"$ \nu $")
    plt.xlabel(r" $ \tau $")
    plt.title(r"$ \nu (\tau) $")

    if save:
        if model.lower() == "osnovni":
            plt.savefig("./Images/osnovni_model")

        elif model.lower() == "radar zacetna":
            plt.savefig("./Images/radar_model")

    plt.show()

plot_zvezno()
plot_zvezno(model="radar zacetna")