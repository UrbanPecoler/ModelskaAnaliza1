import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import psutil

from tqdm import tqdm
import time

# TODO

def random_guess(N):
    initial_guess = np.random.rand(2 * N)
    initial_guess[::2] *= np.pi      # theta in [0, pi]
    initial_guess[1::2] *= 2 * np.pi # phi in [0, 2*pi]
    return initial_guess

# PRVI DEL -- THOMSONOV PROBLEM
def sphere2kart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def razdalja_sfera(p1, p2):
    return np.linalg.norm(p1 - p2)

neg = False # True samo za primer, ko imam + in - naboje na krogli

def E_potencial(params, N, neg=neg):
    energy = 0
    points = []
    points_neg = []

    for i in range(N):
        theta = params[2*i]     # Latitude (0 <= theta <= pi)
        phi = params[2*i + 1]   # Longitude (0 <= phi <= 2*pi)
        points.append(sphere2kart(theta, phi))

        if neg:
            theta_neg = params[2*(i + N)]     # Latitude for negative charges
            phi_neg = params[2*(i + N) + 1]   # Longitude for negative charges
            points_neg.append(sphere2kart(theta_neg, phi_neg))

    for i in range(N):
        for j in range(i+1, N):
            energy += 1.0 / razdalja_sfera(points[i], points[j])
    
    return energy


def thomson_plot(N_list, porocilo=True):

    if porocilo:
        fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(12, 12))
    else:
        fig, axs = plt.subplots(nrows=4, ncols=3, subplot_kw={'projection': '3d'}, figsize=(18, 16))

    for idx, N in enumerate(N_list):
        initial_guess = random_guess(N)

        result = minimize(E_potencial, initial_guess, args=(N,), method='BFGS')
        optimal_angles = result.x

        optimal_points = [sphere2kart(optimal_angles[2*i], optimal_angles[2*i+1]) for i in range(N)]
        if porocilo:
            ax = axs[idx // 2, idx % 2]

        else:
            ax = axs[idx // 3, idx % 3]

        # Sfera
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_wireframe(x, y, z, color='black', linewidth=0.5, alpha=0.15)  
        # ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.6, rstride=1, cstride=1, antialiased=True)
        initial_points = [sphere2kart(initial_guess[2*i], initial_guess[2*i+1]) for i in range(N)]

        for point in optimal_points:
            ax.scatter(point[0], point[1], point[2], color='g', s=80)

        # for point in initial_points:
        #     ax.scatter(point[0], point[1], point[2], color='r', s=20)

        for i in range(N):
            for j in range(i + 1, N):
                x_line = [optimal_points[i][0], optimal_points[j][0]]
                y_line = [optimal_points[i][1], optimal_points[j][1]]
                z_line = [optimal_points[i][2], optimal_points[j][2]]
                ax.plot(x_line, y_line, z_line, color='g', linewidth=0.6, alpha=0.6)  


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Razporeditev {N} nabojev")
    plt.tight_layout()
    if porocilo:
        plt.savefig("./Images/sfera_4")
    else:
        plt.savefig("./Images/sfera_12")
    plt.show()

# N_p = [3, 4, 6, 10]
# N_a = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20]
# thomson_plot(N_p, porocilo=True)
# thomson_plot(N_a, porocilo=False)

# ============================================================================================
e_wiki = np.loadtxt("Energija_wiki.txt")

# List of methods to iterate over
methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'SLSQP']

colors = plt.get_cmap('Set2').colors 

color_map = {
    'Nelder-Mead': colors[0],
    'Powell': colors[1],
    'CG': colors[2],
    'BFGS': colors[3],
    'SLSQP': colors[4]
}

# MAIN PARAMETRA
N_max = 70
step = 3

# ============================================================================================
# RAZDALJA MED TOČKAMI V ODVISNOSTI OD N IN VOLUMEN POLIEDRA za AMEBO 
# TODO

def dist_vol(N, method='Nelder-Mead'):
    """Compute the average pairwise distance between points for a given N."""
    initial_guess = random_guess(N)
    result = minimize(E_potencial, initial_guess, args=(N,), method=method)

    optimal_angles = result.x
    optimal_points = [sphere2kart(optimal_angles[2 * i], optimal_angles[2 * i + 1]) for i in range(N)]

    hull = ConvexHull(optimal_points)

    total_distance = []
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                distance = razdalja_sfera(optimal_points[simplex[i]], optimal_points[simplex[j]])
                total_distance.append(distance)

    volume = hull.volume

    return np.average(np.array(total_distance)), volume


def plot_razdalja():
    Ns = np.arange(4, N_max, step)
    # Ns = [4, 20, 50, 100]

    distances = []
    volumes = []
    for N in tqdm(Ns):
        avg_distance, vol = dist_vol(N)
        distances.append(avg_distance)
        volumes.append(vol)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax1.plot(Ns, distances, "o-", color=color_map['Nelder-Mead'], label="povprečna razdalja")
    ax2.plot(Ns, volumes, "o-", color=color_map['Nelder-Mead'], label="Volumen poliedra")
    ax2.axhline((4/3) * np.pi, linestyle='--', color='black', label='Volumen krogle')
    ax1.set_xlabel("N")
    ax1.set_ylabel("Povprečna razdalja med točkami")
    ax2.set_xlabel("N")
    ax2.set_ylabel("Volumen poliedra")
    ax1.legend()
    ax2.legend()
    plt.savefig("./Images/Razdalja")
    plt.close()

# plot_razdalja()

# ENERGIJA V ODVISNOSTI OD N za AMEBO
def plot_energy():
    Ns = np.arange(2, N_max, step, dtype=int)
    e_ = []

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    for N in tqdm(Ns):
        initial_guess = random_guess(N)
        
        result = minimize(E_potencial, initial_guess, args=(N,), method="Nelder-Mead")
        e_.append(result.fun)

    ax1.scatter(Ns, e_, color=colors[0], label="Nelder-Mead")
    ax2.scatter(Ns, e_/N, color=colors[0], label="Nelder-Mead")

    # Linearni fit za N > 10
    filtered_Ns = Ns[10:]
    filtered_e_N = e_[10:] / Ns[10:]
    slope, intercept = np.polyfit(filtered_Ns, filtered_e_N, 1)
    fit_line = slope * np.array(filtered_Ns) + intercept
    ax2.plot(filtered_Ns, fit_line, color=colors[0], linestyle='--', label=f'"Nelder-Mead" fit')

    ax1.set_xlabel("N")
    ax1.set_ylabel("Energija")
    ax1.set_title("Energija v odvisnosti od števila nabojev")
    ax2.set_xlabel("N")
    ax2.set_ylabel("Energija/N")
    ax2.set_title("Linearna interpolacija energije normirane na število nabojev")
    ax2.legend()
    fig.tight_layout()
    plt.savefig("./Images/Energija_N")
    plt.close()

# plot_energy()

# LIMITA ENERGIJE
def energy_limit():
    N_max = 100
    step = 5
    Ns = np.arange(2, N_max, step, dtype=int)
    e_ = []

    for N in tqdm(Ns):
        initial_guess = random_guess(N)
        
        result = minimize(E_potencial, initial_guess, args=(N,), method="Nelder-Mead")
        e_.append(result.fun/N**2)

    plt.plot(Ns, e_, color=colors[0], label="Nelder-Mead")
    plt.axhline(x=0.5, linestyle='--', color='black', label='x = 0.5')
    plt.xlabel("E/N^2")
    plt.ylabel("N")
    plt.savefig("./Images/Energija_konv")
    plt.close()

# energy_limit()

# NAPAKA ENERGIJE, ČASOVNA ZAHTEVNOST, POMNILNIK, ŠTEVILO ITERACIJ
def plot_analysis():
    Ns = np.arange(2, N_max, step, dtype=int)

    results = {
        'Nelder-Mead': {'e': [], 't': [], 'mem': [], 'it': []},
        'Powell': {'e': [], 't': [], 'mem': [], 'it': []},
        'CG': {'e': [], 't': [], 'mem': [], 'it': []},
        'BFGS': {'e': [], 't': [], 'mem': [], 'it': []},
        'SLSQP': {'e': [], 't': [], 'mem': [], 'it': []},
    }

    def get_memory_usage():
        """ Function to track memory in MB """
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 2)

    def rel_err(method, Ns=Ns, e_wiki=e_wiki):
        """ Relativna napaka energije """
        e_wiki = [e_wiki[N_ - 2] for N_ in Ns]
        return np.abs(np.array(e_wiki)-np.array(results[method]["e"]))/np.array(e_wiki)

    fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axs

    for N in tqdm(Ns):
        initial_guess = random_guess(N)

        for method in methods:
            mem_start = get_memory_usage()
            start = time.time()

            result = minimize(E_potencial, initial_guess, args=(N,), method=method)

            results[method]['e'].append(result.fun)
            results[method]['t'].append(time.time() - start)
            results[method]['mem'].append(np.abs(mem_start - get_memory_usage()))
            results[method]['it'].append(result.nit)

    for method in methods:
        # print(results[method]["mem"])
        ax1.plot(Ns, results[method]['t'], "o-", color=color_map[method], label=method)
        ax2.plot(Ns, results[method]['it'], "o-", color=color_map[method], label=method)
        ax3.plot(Ns, results[method]['mem'], "o-", color=color_map[method], label=method)
        ax4.plot(Ns, rel_err(method), "o-", color=color_map[method], label=method)


    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Methods')
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Methods')
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Methods')
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Methods')

    ax1.set_title("Časovna zahtevnost")
    ax2.set_title("Število iteracij")
    ax3.set_title("Poraba pomnilnika")
    ax4.set_title("Relativna napaka energije")

    ax1.set_ylabel("Časovna zahtevnost")
    ax2.set_ylabel("Število iteracij")
    ax3.set_ylabel("Memory usage")
    ax4.set_ylabel("Relativna napaka energije")

    ax4.set_xlabel("N")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    ax4.set_yscale("log")

    # ax4.set_ylim(1e-15, 10)
    plt.subplots_adjust(hspace=0.4, right=0.75)
    fig.tight_layout()
    plt.savefig("./Images/Analiza_metod")
    plt.close()

# plot_analysis()

# PRAVILNOST MINIMIZACIJE
def plot_min_hist():
    N = 5
    e_val = e_wiki[3]
    iter = 500

    method_stats = {method: {"counter": 0, "difference": 0} for method in methods}

    for i in tqdm(range(iter)):
        initial_guess = random_guess(N)

        for method in methods:
            result = minimize(E_potencial, initial_guess, args=(N,), method=method)
            if round(result.fun, 3) == round(e_val, 3):
                method_stats[method]["counter"] += 1
            
            elif np.abs(e_val - result.fun) > method_stats[method]["difference"]:
                method_stats[method]["difference"] = np.abs(e_val - result.fun)

    method_names = list(method_stats.keys())
    counters = [(method_stats[method]['counter']/iter) * 100 for method in method_names]

    plt.figure(figsize=(10, 6)) 
    bars = plt.bar(method_names, counters, color=[color_map[method] for method in method_names])  
    plt.axhline(iter, color='black', linestyle='--')

    plt.xlabel('Methods')
    plt.xticks(rotation=45)

    plt.ylabel('Counter')
    plt.title('Method Counter Histogram')

    plt.ylim(0, 105)
    for bar, counter in zip(bars, counters):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{counter:.1f}%', ha='center', va='bottom')

    plt.savefig("./Images/Minimizacija_pravilnost")
    plt.close()

# plot_min_hist()

# DIPOLNI MOMENT
# TODO 
def dipole_moment(params, N):
    dipole = np.zeros(3) 
    for i in range(N):
        theta = params[2*i]    
        phi = params[2*i + 1]  
        r_i = sphere2kart(theta, phi)
        dipole += r_i  
    return np.linalg.norm(dipole)  # Return magnitude of dipole moment

dipol_wiki = np.loadtxt("Dipol_wiki.txt")

N_max = 50
step = 1
Ns = np.arange(2, N_max, step, dtype=int)
dipol_wiki = [dipol_wiki[N_ - 2] for N_ in Ns]

def plot_dipole():
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))  
    axs = axs.flatten()  

    for i, method in enumerate(methods):
        dipole_moments = []
        for N in tqdm(Ns):
            initial_guess = random_guess(N)
            result = minimize(E_potencial, initial_guess, args=(N,), method=method)
            dipole = dipole_moment(result.x, N) 
            dipole_moments.append(dipole)

        axs[i].scatter(Ns, dipol_wiki, color="gray", label="Dipol Wiki", s=150)
        axs[i].scatter(Ns, dipole_moments, color=color_map[method], label=method, s=70)
        axs[i].set_title(f"Dipole Moment using {method}")
        axs[i].set_xlabel('N')
        axs[i].set_ylabel('Dipole Moment')
        axs[i].legend()
        axs[i].grid()

    plt.tight_layout()  
    plt.savefig("./Images/Dipole")
    plt.close()

# plot_dipole()


""" Ideje za katere mi je zmanjkalo časa, da bi jih naredil pravilno in uporabil v poročilu"""
# ==============================================================================================
# KVADRUPOLNI MOMENT
# TODO Ni še čisto prav
def quadrupole_tensor(points):
    """Calculate the quadrupole tensor Q_ij for a set of points on a sphere."""
    Q = np.zeros((3, 3)) 
    for p in points:
        r2 = np.dot(p, p)  # r_k^2
        for i in range(3):
            for j in range(3):
                Q[i, j] += 3 * p[i] * p[j] - (r2 if i == j else 0)
    return Q

def plot_quadrupole_3d(N, points, quadrupole_tensor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot charges on the sphere
    ax.scatter(*np.array(points).T, color='blue', s=50, label='Charges')

    # Plot the principal axes of the quadrupole moment
    eigvals, eigvecs = np.linalg.eig(quadrupole_tensor)
    for val, vec in zip(eigvals, eigvecs.T):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], length=np.abs(val)*2, color='red', linewidth=2, label='Principal Axis')

    # Plot a sphere for reference
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Quadrupole Moment Principal Axes for N = {N}')
    plt.close()

# N = 7  
# initial_guess = random_guess(N)

# result = minimize(E_potencial, initial_guess, args=(N,), method='BFGS')
# optimal_angles = result.x
# optimal_points = [sphere2kart(optimal_angles[2*i], optimal_angles[2*i+1]) for i in range(N)]

# Q = quadrupole_tensor(optimal_points)
# plot_quadrupole_3d(N, optimal_points, Q)

# SPECIFIČNI ZAČETNI POGOJI
# TODO - Pogruntaj kaj naredit s tem, oz če uporabiš to v poročilu?


# RAZLIČNO PREDZNAČENI NABOJI NA KROGLI
# NOTE Ni pravilno...
def random_guess_neg(N):
    return np.random.rand(2 * N * 2) * np.array([np.pi, 2 * np.pi] * (N + N))  

neg = True
def thomson_plot(N):
    initial_guess = random_guess_neg(N)
    result = minimize(E_potencial, initial_guess, args=(N,), method='BFGS')
    print(result)
    optimal_angles = result.x

    optimal_points_pos = [sphere2kart(optimal_angles[2*i], optimal_angles[2*i+1]) for i in range(N)]
    optimal_points_neg = [sphere2kart(optimal_angles[2*(i + N)], optimal_angles[2*(i + N) + 1]) for i in range(N)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Sfera z sivim gradientom
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.6, rstride=1, cstride=1, antialiased=True)

    for point in optimal_points_pos:
        ax.scatter(point[0], point[1], point[2], color='g', s=80)  # Pozitivni naboji

    for point in optimal_points_neg:
        ax.scatter(point[0], point[1], point[2], color='b', s=50)  # Negativni naboji

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# thomson_plot(8)




