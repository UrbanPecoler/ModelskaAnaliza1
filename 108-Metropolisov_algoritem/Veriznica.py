import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

from tqdm import tqdm


def molekularna_veriznica(steps, alpha, T, hlim=-18, bounds: tuple = (0,0), num_el=17):
    
    kB = 1 # Boltzmanovo konstanto postavil na 1

    configurations = np.zeros((steps, num_el))
    configurations[0] = np.append(np.append(bounds[0], np.random.randint(-10, 0., num_el-2, dtype=int)), bounds[1])

    Es = np.zeros(steps)
    Es[0] = np.sum(alpha*configurations[0]) + 1/2 * np.sum(((configurations[0]-np.roll(configurations[0], -1, axis=0))[:-1])**2)
    num_accepted = 0.

    for i in range(1, steps):
        E_old = Es[i-1]
        old_configuration = configurations[i-1]

        k = np.random.randint(1, num_el-1) # k-tega spremenimo
        possible_new_configuration = old_configuration.copy()
        
        possible_new_configuration[k] += np.random.choice([-1, 1]) # k-tega spremenimo za + ali - 1
        possible_new_configuration[k] = max(possible_new_configuration[k], hlim) # Nesme biti manj kot hlim
        possible_E_new = np.sum(alpha*possible_new_configuration) + 1/2 * np.sum(((possible_new_configuration-np.roll(possible_new_configuration, -1, axis=0))[:-1])**2)

        if possible_E_new < E_old:
            configurations[i] = possible_new_configuration
            Es[i] = possible_E_new
            num_accepted += 1
        
        else:
            ksi = np.random.rand()
            if ksi < np.exp(-(possible_E_new-E_old)/kB*T):
                configurations[i] = possible_new_configuration
                Es[i] = possible_E_new
                num_accepted += 1
            else:
                configurations[i] = old_configuration
                Es[i] = E_old
    
    return configurations, Es, num_accepted/steps

num_el = 17
steps = 5000
alphas = [0.1, 1, 10]
Ts = [0.1, 1, 10]
steps_ = np.arange(0, steps)
elements = np.arange(0, num_el)

# NARIŠI VERIŽNICO
fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
n = 5

for i, alpha in enumerate(alphas):
    for j, T in enumerate(Ts):
        # Simulacija
        configurations, _, _ = molekularna_veriznica(steps, alpha, T)
        
        # Graf
        ax = axes[i, j]
        for k in range(1, n):
            step = int((steps / n) * k)
            ax.plot(elements, configurations[step], marker='.', markersize=8, linestyle='--', label=f'{step} korak')
        
        # Oznake osi samo na robovih
        if j == 0:  # Najbolj levi grafi
            ax.set_ylabel('Konfiguracija')

        if i == 2:  # Spodnji grafi
            ax.set_xlabel('Elementi verižnice')

        ax.set_title(f'α = {alpha}, T = {T}')
        ax.set_xlabel('Elementi verižnice')
        ax.set_ylabel('Konfiguracija')
        ax.legend()

# Prilagoditev prostora med grafi
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("./Images/Veriznica_plot")
plt.show()



# # ENERGIJA SISTEMA V ODVISNOSTI OD KORAKA
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6), sharex=True)

energija = False
for i, alpha in enumerate(alphas):
    ax = axes[i]
    for j, T in enumerate(Ts):
        configuration, Es, _ = molekularna_veriznica(steps, alpha, T)

        if energija:   
            # Graf
            ax.plot(steps_, Es, label=r'$k_BT =$' + f'{T}')
            ax.set_ylabel("E")
            ax.set_title(r'$\alpha =$' + f'{alpha}')
            ax.legend()
        else:
            avg_height = np.average(configuration, axis=1)
            ax.plot(steps_, avg_height, label=r'$k_BT =$' + f'{T}')
            ax.set_ylabel("Povprečna višina")
            ax.set_title(r'$\alpha =$' + f'{alpha}')
            ax.legend()

axes[2].set_xlabel("Število korakov")
fig.tight_layout()
if energija:
    plt.savefig("./Images/avg_E")
else:
    plt.savefig("./Images/avg_h")
plt.show()


# # POVPREČNA ENERGIJA SISTEMA
def plot_energy(num_iter, Ts, alphas):
    num_iter = 3000
    alphas = np.append(0.1, np.arange(1, 5, 1))
    Ts = np.linspace(1, 10, 20)

    energies = np.zeros((len(alphas), len(Ts)))
    sigma_energies = np.zeros((len(alphas), len(Ts)))

    accepted_steps = np.zeros((len(alphas), len(Ts)))

    for i, alpha in enumerate(alphas):
        print(f"Obdelujem {i+1}/{len(alphas)} za alpha = {alpha}")
        
        for j, T in tqdm(enumerate(Ts)):
            configurations, Es, accepted_ratio = molekularna_veriznica(num_iter, alpha, T)
            
            configurations_eq = configurations[1000:]
            Es_eq = Es[1000:]

            energies[i][j] = np.average(Es_eq)
            sigma_energies[i][j] = np.std(Es_eq)

            accepted_steps[i][j] = accepted_ratio
    plt.figure(figsize=(12, 10))
    plt.title(r'Povprečna energija v odvisnosti od $k_BT$ pri različnih vrednostih $\alpha$.')
    
    for i, alpha in enumerate(alphas):
        plt.errorbar(Ts, energies[i], yerr=sigma_energies[i], fmt='.', linestyle="dashed", markersize=8, capsize=4, 
                     barsabove=False, ecolor='black', label=f'$alpha = {alpha}$')

    plt.xlabel(r'$k_BT$')
    plt.ylabel('Povprečna energija')
    plt.legend()
    # plt.savefig("./Images/avg_E_sist")
    plt.show()

num_iter = 3000
alphas = np.append(0.1, np.arange(1, 5, 1))
Ts = np.linspace(1, 10, 20)

# plot_energy(num_iter, Ts, alphas)



# ===========================================================================================================
# # NEOMEJENO NIVOJEV
num_el = 17
num_iter = 50000
alphas = [0.1, 1.0, 10.0]
kBTs = [0.1, 1.0, 10.0]
elements = np.arange(num_el)
n = 10  # Število korakov za prikaz

results = {}
for i, alpha in enumerate(alphas):
    for j, T in enumerate(kBTs):
        results[(i, j)] = molekularna_veriznica(num_iter, alpha, T, hlim=-1000000)

# Vizualizacija
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, sharey=True)
cmap = get_cmap("viridis")  
norm = Normalize(vmin=0, vmax=n-1)  

for j, T in enumerate(kBTs):
    ax = axes[j]
    ax.set_title(rf"$k_BT = {T}, \alpha = {alphas[2]}$")
    for i in range(n):
        step = int((num_iter / n) * i)
        color = cmap(norm(i)) 
        ax.plot(elements, results[(2, j)][0][step], marker='.', markersize=8, linestyle='--', color=color)
    ax.set_xticks(np.arange(num_el))
    
    if j == 0:  
        ax.set_ylabel(r"$h$")

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  
cbar = fig.colorbar(sm, ax=axes[-1], orientation="vertical", shrink=0.8)
cbar.set_label("Koraki simulacije", rotation=270, labelpad=15)
plt.savefig("./Images/neomejeno_veriznica")
plt.show()



# PRIMERJAVA Z VERIŽNICO
def veriznica(x, a, b, c, d):
    # return (a/2)*(np.exp((x+c)/a) + np.exp(-(x+c)/a)) - b
    return d*np.cosh((x+c)/a)+b

alpha = 5
kBT = 1
config, _, _ = molekularna_veriznica(num_iter, alpha, kBT, hlim=-1000000)
print(np.shape(config))
config_last = config[-1][:]

elements_ = np.arange(len(config_last))
popt, pcov = curve_fit(veriznica, elements_, config_last, maxfev=10000)

a_fit, b_fit, c_fit, d_fit = popt
# print(f"Prilagojeni parameter a: {a_fit}")

plt.plot(elements, config_last, 'o-', label='Podatki')
plt.plot(elements, veriznica(elements, *popt), '--', label=f'Fit: a={a_fit:.2f}, b={b_fit:.2f}, c={c_fit:.2f}')
plt.xlabel('Elementi')
plt.ylabel('Konfiguracija')
plt.legend()
plt.savefig("./Images/Veriznica_fit")
plt.show()



# PRIMERJAVA E ZA ODPRTE IN ZAPRTE NIVOJE
num_el = 17
num_iter = 10000
alphas = [0.1, 1.0, 10.0]
kBTs = [0.1, 1.0, 10.0]
# alphas = np.append(0.1, np.arange(1, 5, 1))
# Ts = np.linspace(1, 10, 20)
elements = np.arange(num_el)

energies = np.zeros((len(alphas), len(Ts)))
sigma_energies = np.zeros((len(alphas), len(Ts)))

energies_c = np.zeros((len(alphas), len(Ts)))
sigma_energies_c = np.zeros((len(alphas), len(Ts)))


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6), sharex=True)
for i, alpha in enumerate(alphas):
    ax = axes[i]
    for j, T in enumerate(kBTs):
        _, Es, _ = molekularna_veriznica(steps, alpha, T)
        _, Es_c, _ = molekularna_veriznica(steps, alpha, T, hlim=-100000)
 
        # Graf
        ax.plot(steps_, Es, label=r'$k_BT =$' + f'{T}')
        ax.plot(steps_, Es_c, "--", label=r'$k_BT =$' + f'{T} - Odprt')
        ax.set_ylabel("E")
        ax.set_title(r'$\alpha =$' + f'{alpha}')
        ax.legend()

        # # configurations_eq = configurations[1000:]
        # Es_eq = Es[1000:]
        # Es_eq_c = Es_c[1000:]

        # energies[i][j] = np.average(Es_eq)
        # sigma_energies[i][j] = np.std(Es_eq)
        # energies_c[i][j] = np.average(Es_eq_c)
        # sigma_energies_c[i][j] = np.std(Es_eq_c)

axes[2].set_xlabel("Število korakov")
fig.tight_layout()
plt.savefig("./Images/primerjava_E")
plt.show()

# plot_energy(Ts, energies_c, sigma_energies_c, alphas)
