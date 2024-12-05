import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

from tqdm import tqdm

class IsingModel:
    def __init__(self, N, kBT, J=1, H=0):
        self.N = N  # Size of the grid (N x N)
        self.kBT = kBT  # Temperature
        self.J = J  # Coupling constant
        self.H = H  # External magnetic field
        self.configuration = np.random.choice([-1, 1], size=(N, N))  # Initial configuration (random spin matrix)
        self.Es = np.zeros(0)  # Array to store energy over time
        self.num_domains = np.zeros(0)  # Array to store number of domains over time
        self.mag = np.zeros(0)  # Array to store magnetization over time
        self.num_accepted = 0  # Number of accepted configurations
    
    def calculate_energy(self):
        energy = 0
        for i in range(self.N):
            for j in range(self.N):
                neighbor_sum = (
                    self.configuration[i, (j + 1) % self.N] +  # right neighbor
                    self.configuration[i, (j - 1) % self.N] +  # left neighbor
                    self.configuration[(i + 1) % self.N, j] +  # bottom neighbor
                    self.configuration[(i - 1) % self.N, j]    # top neighbor
                )
                energy += -self.J * self.configuration[i, j] * neighbor_sum / 2. - self.H * self.configuration[i, j]
        return energy
    
    def count_domains(self):
        _, up_domain = label(np.where(self.configuration == -1, 0, 1))
        _, down_domain = label(np.where(self.configuration == 1, 0, 1))
        return up_domain + down_domain
    
    def calculate_susceptibility(self):
        magnetization = np.sum(self.configuration)
        magnetization_squared = np.sum(self.configuration ** 2)

        average_magnetization = np.abs(magnetization) / self.configuration.size
        average_magnetization_squared = magnetization_squared / self.configuration.size

        susceptibility = (average_magnetization_squared - average_magnetization ** 2) / self.kBT
        return susceptibility
    
    def run_simulation(self, num_iter):
        self.Es = np.zeros(num_iter)
        self.num_domains = np.zeros(num_iter)
        self.mag = np.zeros(num_iter)
        self.num_accepted = 0.0

        self.Es[0] = self.calculate_energy()
        self.num_domains[0] = self.count_domains()
        self.mag[0] = np.sum(self.configuration)

        for i in range(1, num_iter):
            E_old = self.Es[i-1]
            old_configuration = self.configuration.copy()

            j, k = np.random.randint(0, self.N, 2)
            possible_new_configuration = old_configuration.copy()
            possible_new_configuration[j, k] = -possible_new_configuration[j, k]

            del_E = -2 * self.J * possible_new_configuration[j, k] * (possible_new_configuration[(j - 1) % self.N, k] + 
                                                                    possible_new_configuration[(j + 1) % self.N, k] + 
                                                                    possible_new_configuration[j, (k - 1) % self.N] + 
                                                                    possible_new_configuration[j, (k + 1) % self.N]) - 2 * self.H * possible_new_configuration[j, k]

            if del_E < 0:
                self.configuration = possible_new_configuration
                self.Es[i] = E_old + del_E
                self.num_domains[i] = self.count_domains()
                self.mag[i] = np.sum(self.configuration)
                self.num_accepted += 1

            else:
                ksi = np.random.rand()
                if ksi < np.exp(-del_E / self.kBT):
                    self.configuration = possible_new_configuration
                    self.Es[i] = E_old + del_E
                    self.num_domains[i] = self.count_domains()
                    self.mag[i] = np.sum(self.configuration)
                    self.num_accepted += 1
                else:
                    self.configuration = old_configuration
                    self.Es[i] = E_old
                    self.num_domains[i] = self.count_domains()
                    self.mag[i] = np.sum(self.configuration)

        acceptance_rate = float(self.num_accepted) / float(num_iter)
        return self.configuration, self.Es, self.num_domains, self.mag, acceptance_rate

# # Parameters
def visualise_ising(kBT):
    N = 50  # Grid size
    num_iter = 200000  # Number of iterations

    model = IsingModel(N, kBT)

    initial_configuration = model.configuration.copy()  # Save initial configuration
    half_configuration, half_Es, half_num_domains, half_mag, half_acceptance_rate = model.run_simulation(num_iter//2)
    configuration, Es, num_domains, mag, acceptance_rate = model.run_simulation(num_iter)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(initial_configuration, cmap='coolwarm', interpolation='nearest')
    axes[0].set_title('Začetna konfiguracija')

    # axes[1].imshow(half_configuration, cmap='coolwarm', interpolation='nearest')
    # axes[1].set_title('Konfiguracija')

    axes[1].imshow(configuration, cmap='coolwarm', interpolation='nearest')
    axes[1].set_title('Končna konfiguracija')

    fig.tight_layout()
    plt.savefig(f"./Images/Ising_config_T{kBT}")
    plt.show()

kBT = 1
# visualise_ising(kBT)


# ENERGIJA V ODVISNOSTI OD KORAKA
# NOTE NI PRAV
# # Parameters
def analysis_plot(num_iter, N=20, parameter="energy"):
    
    plt.figure(figsize=(10, 6))

    if parameter == "energy":
        mode = 1
        plt.ylabel('Energy')
        plt.title('Energy vs. Number of Steps for different k_B T')
    if parameter == "domain":
        mode = 2
        plt.ylabel('Število domen')
        plt.title('Domains vs. Number of Steps for different k_B T')
    
    N = 20  # Size of the grid
    kBT_values = [1, 2.269, 3]

    for kBT in kBT_values:
        model = IsingModel(N, kBT)
        alaysis_param = model.run_simulation(num_iter)[mode]
        plt.plot(alaysis_param, label=f'k_B T = {kBT}')

    plt.xlabel('Number of Steps')
    plt.legend()
    if parameter == "energy":
        plt.savefig("./Images/Ising_energy-step")
    elif parameter == "domain":
        plt.savefig("./Images/Ising_domain-step")
    plt.close()

num_iter = 100000
# analysis_plot(num_iter, N=50, parameter="energy")
# analysis_plot(num_iter, N=50, parameter="domain")

# ANALIZA OSTALIH PARAMETROV
# Parameters
def analysis_params(num_iter, Hs, kBTs, run=True, N=20):
    # Preveri ali so podatki že shranjeni
    if run:
        # Initialize arrays for averages and standard deviations
        avg_E = np.zeros((len(Hs), len(kBTs)))
        std_E = np.zeros_like(avg_E)
        avg_mag = np.zeros_like(avg_E)
        std_mag = np.zeros_like(avg_E)
        avg_num_domains = np.zeros_like(avg_E)
        std_num_domains = np.zeros_like(avg_E)
        ratios = np.zeros_like(avg_E)
        chi = np.zeros_like(avg_E)
        c = np.zeros_like(avg_E)

        # Loop through different H and kBT values
        for i, H in enumerate(Hs):
            print(f"Progress: {i + 1}/{len(Hs)}")

            for j, kBT in tqdm(enumerate(kBTs), total=len(kBTs)):
                model = IsingModel(N=N, kBT=kBT, H=H)
                
                result = model.run_simulation(num_iter)
                
                config, Es, num_domains, mag, ratio = result 
                
                # Calculate averages and standard deviations from the simulation results
                avg_E[i][j] = np.average(Es[50000:])
                std_E[i][j] = np.std(Es[50000:])
                avg_mag[i][j] = np.average(mag[50000:])
                std_mag[i][j] = np.std(mag[50000:])
                avg_num_domains[i][j] = np.average(num_domains[50000:])
                std_num_domains[i][j] = np.std(num_domains[50000:])
                
                # Calculate susceptibility and specific heat
                chi[i][j] = np.var(mag[50000:]) / (N * kBT)
                c[i][j] = np.var(Es[50000:]) / (N * kBT**2)
                
                ratios[i][j] = ratio

        # Saving the results in a .npz file
        np.savez('ising_results_long_small.npz', avg_E=avg_E, std_E=std_E, avg_mag=avg_mag,
                 std_mag=std_mag, avg_num_domains=avg_num_domains, 
                 std_num_domains=std_num_domains, chi=chi, c=c, ratios=ratios)

    else:
        # Load the data from the file
        data = np.load('ising_results_long_small.npz')
        avg_E = data['avg_E']
        std_E = data['std_E']
        avg_mag = data['avg_mag']
        std_mag = data['std_mag']
        avg_num_domains = data['avg_num_domains']
        std_num_domains = data['std_num_domains']
        chi = data['chi']
        c = data['c']
        ratios = data['ratios']

    return avg_E, std_E, avg_mag, std_mag, avg_num_domains, std_num_domains, chi, c, ratios


run = True  
num_iter = 100000
# kBTs = np.linspace(0.1, 5, 6)
kBTs = [1, 2, 3, 4, 5, 6, 7, 8]
Hs = np.linspace(0, 1, 5)
avg_E, std_E, avg_mag, std_mag, avg_num_domains, std_num_domains, chi, c, ratios = analysis_params(num_iter, Hs, kBTs, run=run)

# Vizualizacija
plt.figure(figsize=(10, 6))
for i, H in enumerate(Hs):
    plt.errorbar(kBTs, avg_E[i], yerr=std_E[i], marker="o", linestyle="dotted", markersize=8, label=f'H = {H}')
plt.xlabel('k_B T')
plt.ylabel('E')
plt.title('Povprečna energija vs. k_B T za različne H')
plt.legend()
plt.savefig("./Images/Ising_avgE")
plt.show()

plt.figure(figsize=(10, 6))
for i, H in enumerate(Hs):
    plt.errorbar(kBTs, avg_mag[i], yerr=std_E[i], marker="o", linestyle="dotted", markersize=8, label=f'H = {H}')
plt.xlabel('k_B T')
plt.ylabel('Povprečna magnetizacija')
plt.title('Povprečna lastna magnetizacija vs. k_B T za različne H')
plt.legend()
plt.savefig("./Images/Ising_avgMag")
plt.show()

plt.figure(figsize=(10, 6))
for i, H in enumerate(Hs):
    plt.errorbar(kBTs, avg_num_domains[i], yerr=std_num_domains[i], marker="o", linestyle="dotted", markersize=8, label=f'H = {H}')
plt.xlabel('k_B T')
plt.ylabel('Povprečno št. domen')
plt.title('Povprečno št. domen vs. k_B T za različne H')
plt.legend()
plt.savefig("./Images/Ising_avgDomain")
plt.show()

plt.figure(figsize=(10, 6))
for i, H in enumerate(Hs):
    plt.plot(kBTs, c[i], "o-", label=f'H = {H}')
plt.xlabel('k_B T')
plt.ylabel('Specifična toplota')
plt.title('Specifična toplota vs. k_B T za različne H')
plt.legend()
plt.savefig("./Images/Ising_avgSpecTopl")
plt.show()

plt.figure(figsize=(10, 6))
for i, H in enumerate(Hs):
    plt.plot(kBTs, chi[i], "o-", label=f'H = {H}')
plt.xlabel('k_B T')
plt.ylabel('Susceptibilnost')
plt.title('Susceptibilnost vs. k_B T za različne H')
plt.legend()
plt.savefig("./Images/Ising_avgSuscept")
plt.show()
