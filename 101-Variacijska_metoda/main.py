import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from tqdm import tqdm

# SAVEFIG
save_fig = False
# ============================================================================================================

# PRVI DEL NALOGE
t = np.linspace(0, 1, 1000, endpoint=True)

def hitrost1(t, v0):
    return v0 + 3/2*(1-v0)*(2*t - t**2)

def hitrost2(t, v0, v_max):
    return 3*(2-v0-v_max)*(t - t**2) + (v_max - v0)*t + v0  # Naj bi blo tk -- usaj grafi so pravilni

def pot1(t, v0):
    return v0*t + 1/2*(1-v0)*(3*t**2 - t**3)

def pot2(t, v0, v_max):
    return (2 - v0 - v_max)*(1.5 * t**2 - t**3) + 1/2*(v_max-v0)*t**2 + v0*t  # Naj bi blo tk -- usaj grafi so pravilni

def pospesek1(t, v0):
    return 3*(1-v0)*(1 - t)

def pospesek2(t, v0, v_max):
    return 3*(2-v0-v_max)*(1 - 2*t) + v_max - v0

def plot_firstPart(model="osnovni", save=save_fig):
    """KATERI MODEL GLEDAMO
    # OSNOVNI - odvod pri 1 == 0
    # RADAR ZACETNA - pri t = 1 imam fiksno hitrost, spreminjam zacetno hitrost"""
    v0s = np.arange(0, 3.1, 0.1)

    norm = mcolors.Normalize(vmin=v0s.min(), vmax=v0s.max())
    cmap = cm.viridis

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))
    ax1, ax2, ax3, ax4 = ax.flatten()
    for v0 in v0s:
        if model.lower() == "osnovni":
            ax1.plot(t, hitrost1(t, v0), color=cmap(norm(v0)))
            ax2.plot(t, pot1(t, v0), color=cmap(norm(v0)))
            ax3.plot(t, pospesek1(t, v0), color=cmap(norm(v0)))
            # najdem v in x pri določenih časih
            x = []
            v = []
            for t_ in t:
                v.append(hitrost1(t_, v0))
                x.append(pot1(t_, v0))
            ax4.plot(x, v, color=cmap(norm(v0)))

        elif model.lower() == "radar zacetna":
            v_max = 1.5
            ax1.plot(t, hitrost2(t, v0, v_max), color=cmap(norm(v0)))
            ax2.plot(t, pot2(t, v0, v_max), color=cmap(norm(v0)))
            ax3.plot(t, pospesek2(t, v0, v_max), color=cmap(norm(v0)))
            # najdem v in x pri določenih časih
            x = []
            v = []
            for t_ in t:
                v.append(hitrost2(t_, v0, v_max))
                x.append(pot2(t_, v0, v_max))
            ax4.plot(x, v, color=cmap(norm(v0)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ax1.set_ylabel(r"$ \nu $")
    ax1.set_xlabel(r" $ \tau $")
    ax1.set_title(r"$ \nu (\tau) $")

    ax2.set_ylabel(r"$ \chi $")
    ax2.set_xlabel(r" $ \tau $")
    ax2.set_title(r"$ \chi (\tau) $")

    ax3.set_ylabel(r"$ \alpha $")
    ax3.set_xlabel(r" $ \tau $")
    ax3.set_title(r"$ \alpha (\tau) $")

    ax4.set_ylabel(r"$ \nu $")
    ax4.set_xlabel(r" $ \chi $")
    ax4.set_title(r"$ \nu (\chi) $")

    fig.tight_layout()

    cbar = fig.colorbar(sm, ax=ax.ravel().tolist())
    cbar.set_label(r"$v_0 $", fontsize=20)
    if save:
        if model.lower() == "osnovni":
            plt.savefig("./Images/osnovni_model")

        elif model.lower() == "radar zacetna":
            plt.savefig("./Images/radar_model")

    plt.show()

# plot_firstPart()
# plot_firstPart(model="radar zacetna")


# NEFIZIKALNI DEL MODELA
def plot_nefizi(save=save_fig):
    v0s = np.linspace(1.5, 3.12, 20)
    err_v0s = np.linspace(3.12, 7, 30)

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    for v0 in v0s:
        ax1.plot(t, hitrost1(t, v0), color="green")
        ax2.plot(t, pot1(t, v0), color="green")

    for v0 in err_v0s:
        ax1.plot(t, hitrost1(t, v0), color="red")
        ax2.plot(t, pot1(t, v0), color="red")

    ax1.plot(t, hitrost1(t, 2.98), color="blue", label="Mejna vrednost, $v_0 = 2.98 $")
    ax2.plot(t, pot1(t, 2.98), color="blue", label="Mejna vrednost, $v_0 = 2.98 $")

    ax1.hlines(0, 0, 1, color="black")
    ax2.hlines(1, 0, 1, color="black")

    ax1.set_ylabel(r"$ \nu $")
    ax1.set_xlabel(r" $ \tau $")
    ax1.set_title(r"$ \nu (\tau) $")
    ax1.legend()

    ax2.set_ylabel(r"$ \chi $")
    ax2.set_xlabel(r" $ \tau $")
    ax2.set_title(r"$ \chi (\tau) $")
    ax2.legend()

    fig.tight_layout()
    if save:
        plt.savefig("./Images/odpoved_modela")
    plt.show()

# plot_nefizi()

# DRUGI DEL -- SODE POTENCE
# TODO

def hitrost_sode(t, v0, p):
    return v0 + (4*p - 1) / (2*p) * (1 - v0) * (1 - (1 - t)**((2*p) / (2*p - 1)))

def plot_sode(save=save_fig):
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))
    ax1, ax2, ax3, ax4 = ax.flatten()

    # v_0 = 0.5, 1, 1.5, 2
    v_01 = 0.5
    v_02 = 1
    v_03 = 1.5
    v_04 = 2
    ps = np.arange(1, 6, 1)

    norm = mcolors.Normalize(vmin=ps.min(), vmax=ps.max())
    cmap = cm.viridis

    for p in ps:
        ax1.plot(t, hitrost_sode(t, v_01, p), color=cmap(norm(p)))
        ax2.plot(t, hitrost_sode(t, v_02, p), color=cmap(norm(p)))
        ax3.plot(t, hitrost_sode(t, v_03, p), color=cmap(norm(p)))
        ax4.plot(t, hitrost_sode(t, v_04, p), color=cmap(norm(p)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ax1.set_ylabel(r"$ \nu $")
    ax1.set_xlabel(r" $ \tau $")
    ax1.set_title(r"$v_0 = 0.5 $")

    ax2.set_ylabel(r"$ \nu $")
    ax2.set_xlabel(r" $ \tau $")
    ax2.set_title(r"$v_0 = 1.0 $")

    ax3.set_ylabel(r"$ \nu $")
    ax3.set_xlabel(r" $ \tau $")
    ax3.set_title(r"$v_0 = 1.5 $")

    ax4.set_ylabel(r"$ \nu $")
    ax4.set_xlabel(r" $ \tau $")
    ax4.set_title(r"$v_0 = 2.0 $")


    cbar = fig.colorbar(sm, ax=ax.ravel().tolist())
    cbar.set_label(r"p", fontsize=20)

    if save:
        plt.savefig("./Images/sode_potence")

    plt.show()

# plot_sode()

def plot_lim(save=save_fig):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    ax1, ax2 = ax
    ps = np.arange(0.5001, 0.6, 0.01)

    norm = mcolors.Normalize(vmin=ps.min(), vmax=ps.max())
    cmap = cm.viridis
    for p in ps:
        print(p)
        ax1.plot(t, hitrost_sode(t, 0.5, p), color=cmap(norm(p)))
        ax2.plot(t, hitrost_sode(t, 1.5, p), color=cmap(norm(p)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax.ravel().tolist())
    cbar.set_label(r"p", fontsize=20)

    ax1.set_ylabel(r"$ \nu $")
    ax1.set_xlabel(r" $ \tau $")
    ax1.set_title(r"$v_0 = 0.5 $")

    ax2.set_ylabel(r"$ \nu $")
    ax2.set_xlabel(r" $ \tau $")
    ax2.set_title(r"$v_0 = 1.5 $")

    if save:
        plt.savefig("./Images/lim_p")

    plt.show()

# plot_lim()

# TRETJI DEL -- KVADRAT HITROSTI
# TODO 

def hitrost_kvadrat(t, v0, tc):
    cosh = np.cosh(tc*(t-1))/np.cosh(tc)
    faktor = (1 - v0/tc * np.tanh(tc)) / (1 - 1/tc * np.tanh(tc))
    return faktor * (1 - cosh) + v0 * cosh

def plot_C_difV(save=save_fig):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    ax1, ax2 = ax
    v0s = np.arange(0, 3.1, 0.1)

    # Create a colormap
    norm = mcolors.Normalize(vmin=v0s.min(), vmax=v0s.max())
    cmap = cm.viridis

    tc1 = 2
    tc2 = 10
    for v0 in v0s:
        ax1.plot(t, hitrost_kvadrat(t, v0, tc1), color=cmap(norm(v0)))
        ax2.plot(t, hitrost_kvadrat(t, v0, tc2), color=cmap(norm(v0)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])


    ax1.set_ylabel(r"$ \nu $")
    ax1.set_xlabel(r" $ \tau $")
    ax1.set_title(r"$\tau_C = 2 $")

    ax2.set_ylabel(r"$ \nu $")
    ax2.set_xlabel(r" $ \tau $")
    ax2.set_title(r"$\tau_C = 10 $")

    fig.tight_layout()

    cbar = fig.colorbar(sm, ax=ax.ravel().tolist())
    cbar.set_label(r"$v_0 $", fontsize=20)
    if save:
        plt.savefig("./Images/C_razlicne_v")
    plt.show()

def plot_v_difC(save=save_fig):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    ax1, ax2 = ax
    # v0s = np.arange(0, 3.1, 0.1)
    tcs = np.arange(1, 20, 0.25)

    norm = mcolors.Normalize(vmin=tcs.min(), vmax=tcs.max())
    cmap = cm.viridis

    v01 = 0.5
    v02 = 1.5
    for tc in tcs:
        ax1.plot(t, hitrost_kvadrat(t, v01, tc), color=cmap(norm(tc)))
        ax2.plot(t, hitrost_kvadrat(t, v02, tc), color=cmap(norm(tc)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    ax1.set_ylabel(r"$ \nu $")
    ax1.set_xlabel(r" $ \tau $")
    ax1.set_title(r"$v_0 = 0.5 $")

    ax2.set_ylabel(r"$ \nu $")
    ax2.set_xlabel(r" $ \tau $")
    ax2.set_title(r"$v_0 = 1.5 $")

    fig.tight_layout()

    cbar = fig.colorbar(sm, ax=ax.ravel().tolist())
    cbar.set_label(r"C", fontsize=20)
    if save:
        plt.savefig("./Images/v_razlicni_C")
    plt.show()

# plot_v_difC()
# plot_C_difV()


# ČETRTI DEL -- ZAPOREDNI SEMAFORJI
t1 = np.linspace(0, 1, 1000, endpoint=True)
t2 = np.linspace(1, 2, 1000, endpoint=True)
t3 = np.linspace(2, 3, 1000, endpoint=True)
t4 = np.linspace(3, 4, 1000, endpoint=True)
t5 = np.linspace(4, 5, 1000, endpoint=True)
t6 = np.linspace(5, 6, 1000, endpoint=True)
t7 = np.linspace(6, 7, 1000, endpoint=True)
t8 = np.linspace(7, 8, 1000, endpoint=True)

v0s = np.arange(0, 3, 0.05)
colors = cm.viridis(np.linspace(0, 1, len(v0s))) 

fig, ax = plt.subplots(figsize=(10, 6))

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=v0s.min(), vmax=v0s.max()))
sm.set_array([])  # Only needed for older versions of Matplotlib
cbar = plt.colorbar(sm, ax=ax, label='v0') 


for i, v0 in enumerate(v0s):
    v0_1 = hitrost1(t1, v0)[-1]
    v0_2 = hitrost1(t1, v0_1)[-1]
    v0_3 = hitrost1(t1, v0_2)[-1]
    v0_4 = hitrost1(t1, v0_3)[-1]
    v0_5 = hitrost1(t1, v0_4)[-1]
    v0_6 = hitrost1(t1, v0_5)[-1]
    v0_7 = hitrost1(t1, v0_6)[-1]

    ax.plot(t1, hitrost1(t1, v0), color=colors[i])
    ax.plot(t2, hitrost1(t1, v0_1), color=colors[i])
    ax.plot(t3, hitrost1(t1, v0_2), color=colors[i])
    ax.plot(t4, hitrost1(t1, v0_3), color=colors[i])
    ax.plot(t5, hitrost1(t1, v0_4), color=colors[i])
    ax.plot(t6, hitrost1(t1, v0_5), color=colors[i])
    ax.plot(t7, hitrost1(t1, v0_6), color=colors[i])
    ax.plot(t8, hitrost1(t1, v0_7), color=colors[i])

ax.set_xlabel(r'$\nu $')
ax.set_ylabel(r'$\tau $')

for x in range(1, 9): 
    ax.axvline(x=x, color='black', linestyle=':', linewidth=0.8)

if save_fig:
    plt.savefig("./Images/zaporedni_semaforji")
plt.show()