import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


num_sim = 10000
dt = 0.5

def model_izumrtja(beta, N_0, dt=dt, num_sim=num_sim):
    izumrtje_t = []
    populations = []

    for _ in range(num_sim):
        N = N_0
        t = 0
        times = [0]
        population = [N_0]

        while N > 0:
            deaths = np.random.poisson(beta * N * dt)
            N -= deaths
            t += dt
            times.append(t)
            if N < 0:
                population.append(0)
            else:
                population.append(N)
            
        izumrtje_t.append(t)
        populations.append(population)

    return np.array(izumrtje_t), populations




# ====================================================================
# PLOTTING

def plot_statistika1(Ns, betas, dt=dt, num_sim=num_sim):
    colors = ['blue', 'green', 'red']  # Barve za različne beta vrednosti
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(betas) - 1)) for i in range(len(betas))]

    num_plots = len(Ns)
    fig, axs = plt.subplots(1, num_plots, figsize=(12, 6), sharey=True)

    if num_plots == 1:
        axs = [axs]

    for ax, N_initial in zip(axs, Ns):
        for beta, color in zip(betas, colors):
            izumrtje_t, _ = model_izumrtja(beta, N_initial, num_sim=num_sim)
            num_bins = int(max(izumrtje_t)) + 1
            ax.hist(izumrtje_t, bins=num_bins, histtype='step', density=True, linewidth=1.5, label=f'β = {beta}', color=color)
        
        ax.set_title(f'N = {N_initial}')
        ax.set_xlabel('Čas izumrtja')
        ax.legend()

    axs[0].set_ylabel("Gostota verjetnosti")    
    plt.tight_layout()
    plt.show()

# plot_statistika1([25, 250], [0.1, 0.5, 1])

def avg_pop(pop):
    max_dolzina = max(map(len, pop))
    avg_populacija = np.zeros((len(pop), max_dolzina))
    for i, pop_ in enumerate(pop):
        avg_populacija[i,:len(pop_)] = np.array(pop_)

    avg = np.mean(avg_populacija, axis=0)
    return avg

def analiticna_res(N, beta, t):
    return N * np.exp(-beta * t)

def plot_populacija(Ns, betas, dt=dt, num_sim=num_sim, plot_by="beta"):
    # DODAJ ŠE ZVEZNO VERZIJO (ANALITIČNO)
    cmap = plt.cm.viridis
    
    if plot_by == "beta":
        items, varying, fixed_label = betas, Ns, "N"
        colors = [cmap(i / (len(Ns) - 1)) for i in range(len(Ns))]
    elif plot_by == "N":
        items, varying, fixed_label = Ns, betas, r"$\beta$"
        colors = [cmap(i / (len(betas) - 1)) for i in range(len(betas))]
    else:
        raise ValueError("plot_by must be 'beta' or 'N'")
    
    fig, axs = plt.subplots(1, len(items), figsize=(12, 6), sharey=True)
    if len(items) == 1:
        axs = [axs]
    
    for ax, item in zip(axs, items):
        xlim = []
        for var, color in zip(varying, colors):
            if plot_by == "beta":
                beta, N_initial = item, var
            else:
                beta, N_initial = var, item
            
            _, populacija = model_izumrtja(beta, N_initial, dt=dt, num_sim=num_sim)
            pop = avg_pop(populacija) / N_initial
            t = np.arange(0, len(pop)) * dt
            
            # NI PRAV!!! 
            # t_ = np.linspace(0, len(pop), 10000)
            # if plot_by == "beta":
            #     ax.plot(analiticna_res(1, var, t), label=rf"zvezna $N =$ {var}")
            # else:
            #     ax.plot(analiticna_res(1, item, t), label=rf"zvezna $\beta =$ {var}")

            xlim.append(t[np.argmax(pop < 0.005)])
            ax.step(t, pop, label=f'{fixed_label} = {var}', color=color)
        
        ax.set_title(f'{plot_by.capitalize()} = {item}')
        ax.set_xlabel('Čas')
        ax.set_xlim(-0.1, max(xlim))
        ax.legend()
    
    axs[0].set_ylabel("Delež populacije")
    plt.tight_layout()
    plt.show()

plot_populacija([25, 250, 2500], [0.5, 1, 5], plot_by="N")


# NAPAKA KRIVULJE OD ANALITIČNE VREDNOSTI


# ODVISNOST NAPAKE ANALITIČNE FUNKCIJE OD ČASA SAMPLANJA dt


# POPVPREČEN ČAS IN STD DEVIACIJA ČASA UMIRANJA V ODVISNOSTI OD BETA




# ROJSTVA IN SMRTI