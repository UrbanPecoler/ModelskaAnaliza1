import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import pandas as pd

from tqdm import tqdm

def diff_eq(zac_p, t, alpha, beta):
    Y = np.zeros(3)
    V = zac_p
    Y[0] = - alpha * V[0]*V[1]
    Y[1] = alpha * V[0]*V[1] - beta * V[1]
    Y[2] = beta * V[1]
    return Y 

def plot_epidemic(zac_p, alpha, beta, draw="line", Is=None, Ds=None):
    t = np.arange(0, 100, 0.5)
    if draw == "line":
        RES = odeint(diff_eq, zac_p, t, args=(alpha, beta))
        plt.plot(t, RES[:,0], label="Dovzetni")
        plt.plot(t, RES[:,1], label="Bolni")
        plt.plot(t, RES[:,2], label="Imuni")

        plt.legend()
        print(max(RES[:, 1]))
        plt.show()

    elif draw == "heatmap":
        # Odvisnost od alpha in beta
        B_max = np.zeros((len(alpha), len(beta)))
        B_time = np.zeros((len(alpha), len(beta)))
        for i, alpha in tqdm(enumerate(alphas), total=len(alphas)):
            for j, beta in enumerate(betas):
                RES = odeint(diff_eq, zac_p, t, args=(alpha, beta))
                B = RES[:, 1]
                B_max[i, j] = max(B)
                B_time[i, j] = t[B.argmax()]

        # Odvisnost od I in D
        if Is is None or Ds is None:
            raise ValueError("Is and Ds must be provided when draw is 'heatmap'.")

        alpha = 1.5
        beta = 0.5
        
        B_max_i = np.zeros((len(Is), len(Ds)))
        B_time_i = np.zeros((len(Is), len(Ds)))      
        for i, I in tqdm(enumerate(Is), total=len(Is)):
            for j, D in enumerate(Ds):
                B0 = 1 - I - D
                if B0 >= 0:
                    zac_p = np.array([D, B0, I])

                    RES = odeint(diff_eq, zac_p, t, args=(alpha, beta))
                    B = RES[:, 1]
                    B_max_i[j, i] = max(B)
                    B_time_i[j, i] = t[B.argmax()]
                
        # Custom color setup za primer, ko bolni sploh niso naraščali
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))
        green_max = int(0.1 * 255)  # Scale 0.1 to colormap index range
        green_time = 0
        dark_green = [0.0, 0.7, 0.0, 1]  # RGBA colors
  
        viridis_colors_max, viridis_colors_time = [viridis_colors.copy() for _ in range(2)]
        viridis_colors_max[green_max] = dark_green
        viridis_colors_time[green_time] = dark_green

        cmap_max = ListedColormap(viridis_colors_max)
        cmap_time = ListedColormap(viridis_colors_time)
        norm = BoundaryNorm(np.linspace(0, 1, 256), ncolors=256)

        # PLOT AMPLITUDE 
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
        im1 = ax1.imshow(B_max, extent=[0, 5, 0, 5], origin='lower', aspect='auto', cmap=cmap_max, norm=norm)
        # ax1.colorbar(label="Maximum of B")
        ax1.set_xlabel("Beta")
        ax1.set_ylabel("Alpha")

        im2 = ax2.imshow(B_max_i, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap="viridis")
        # ax2.colorbar(label="Maximum of B")
        ax2.set_xlabel("Imuni na začetku")
        ax2.set_ylabel("Dovzetni na začetku")

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        fig.tight_layout()
        plt.show()
        
        # PLOT TIME
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
        im1 = ax1.imshow(B_time, extent=[0, 5, 0, 5], origin='lower', aspect='auto', cmap=cmap_time, vmin=0, vmax=5)
        # plt.colorbar(label="Time of B")
        ax1.set_xlabel("Beta")
        ax1.set_ylabel("Alpha")
        ax1.set_title("Heatmap of Maximum B for varying Alpha and Beta")

        im2 = ax2.imshow(B_time_i, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap="viridis")
        # plt.colorbar(label="Time of B")
        ax2.set_xlabel("Imuni na začetku")
        ax2.set_ylabel("Dovzetni na začetku")
        # ax2.set_title("Heatmap of Maximum B for varying Alpha and Beta")

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        fig.tight_layout()
        plt.show()

    elif draw == "timing":
        # Čas trajanja epidemije, za pogoj ko je B > 0.05
        pass
        

zac_p = np.array([0.9, 0.1, 0.0])

# alpha_ = 0.1
# beta_ = 0.3
# plot_epidemic(zac_p, alpha_, beta_)

alphas = np.arange(0, 5, 0.1)  
betas = np.arange(0, 5, 0.1)
Is = np.arange(0, 1, 0.05)
Ds = np.arange(0, 1, 0.05)
plot_epidemic(zac_p, alphas, betas, draw="heatmap", Is=Is, Ds=Ds)

# ===================================================================================
# MAIN PART ANALIZE
# TODO
# X -- B_max v odvisnosti od alpha in beta (heatmap) 
# X -- B_max v odvisnosti od I in D (na isti fig k zgornji)
# X -- čas maksimuma v odvisnosti od alpha in beta (heatmap)
# trajanje epidemije (kjer je B > 0.05)

# I0 v odvisnosti od vrha, da vn najdem kritično precepljenost
# Epidemija z več stadiji (B1 oboleli, pa ne vejo in B2 oboleli ki vedo + M smrtnost)


# PARAMETRI

# RES = odeint(diff_eq, zac_p, t)
# D = RES[:, 0]
# B = RES[:, 1]
# I = RES[:, 2]

# print(max(B))
# print(B.argmax()+1)


alpha = 0.1
beta = 5

zac_p = np.array([0.9, 0.1, 0.0])

t = np.arange(0, 100, 0.5)

RES = odeint(diff_eq, zac_p, t, args=(alpha, beta))
B = RES[:, 1]
print(max(RES[:, 1]))

