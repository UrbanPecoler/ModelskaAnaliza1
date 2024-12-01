import numpy as np
import time
import matplotlib.pyplot as plt

from tqdm import tqdm

def estimate_transmissivity_1D_model(num_samples, lam=0.5):

    pobegli = 0.
    n_sipanj = np.zeros(num_samples)
    pot = np.zeros(num_samples)
    
    for i in range(num_samples):

        x = np.random.exponential(lam)
        m = 0 # st sipanj

        while x < 1. and x > 0.:
            dx = np.random.choice([-1., 1.]) + np.random.exponential(lam)
            x += dx
            m += 1

        if x >= 1.:
            pobegli += 1

        n_sipanj[i] = m

    T = pobegli/num_samples
    T_err = 1. / float(num_samples)**(3./2.) * np.sqrt(float(num_samples)*float(pobegli) - float(pobegli)**2)

    return T, T_err, n_sipanj


def estimate_transmissivity_3D_model(num_samples, lam=0.5):

    pobegli = 0.
    n_sipanj = np.zeros(num_samples)
    
    for i in range(num_samples):

        x = np.random.exponential(lam)
        y = 0.
        z = 0.
        m = 0 # st sipanj

        while x < 1. and x > 0.:
            x += np.random.exponential(lam) * np.cos(2*np.pi*np.random.rand()) * np.sin(np.pi*np.random.rand())
            y += np.random.exponential(lam) * np.sin(2*np.pi*np.random.rand()) * np.sin(np.pi*np.random.rand())
            z += np.random.exponential(lam) * np.cos(np.pi*np.random.rand())
            m += 1

        if x >= 1.:
            pobegli += 1
        
        n_sipanj[i] = m

    T = pobegli/num_samples
    T_err = 1. / float(num_samples)**(3./2.) * np.sqrt(float(num_samples)*float(pobegli) - float(pobegli)**2)

    return T, T_err, n_sipanj


num = int(10000)
lmbd = 0.2
lmbd_S = "0_2"
Ts1D, Ts1D_err, n1D = estimate_transmissivity_1D_model(num, lam=lmbd)
Ts3D, Ts3D_err, n3D = estimate_transmissivity_3D_model(num, lam=lmbd)

# Create a histogram with bins for each natural number
bins3D = np.arange(-0.5, np.max(n3D), 1)  # Adjust the range based on the natural numbers in your array
bins1D = np.arange(-0.5, np.max(n1D), 1)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax1.set_yscale('log')
ax1.hist(n1D, bins=bins1D, color='red', edgecolor='black', alpha=1, label='1D model')

ax2.hist(n3D, bins=bins3D, color='blue', edgecolor='black', alpha=0.4, label='3D model')

# Add labels and title
ax1.legend()
ax1.set_xlabel('Število sipanj')
ax1.set_ylabel('Število nevtronov')
ax2.set_xlabel('Število sipanj')
ax2.set_ylabel('Število nevtronov')

ax1.set_title(r'$\lambda = {}$, 1-D'.format(lmbd))
ax2.set_title(r'$\lambda = {}$, 3-D'.format(lmbd))

plt.savefig(f"./Images/3_lmbd{lmbd_S}")
plt.show()


# # PREPUSTNOST V ODVISNOSTI OD N ZA \lambda = 1/2
# num_samples = np.logspace(3, 5, dtype=int)


# Ts1D = np.zeros_like(num_samples, dtype=float)
# Ts1D_err = np.zeros_like(num_samples, dtype=float)
# Ts3D = np.zeros_like(num_samples, dtype=float)
# Ts3D_err = np.zeros_like(num_samples, dtype=float)

# for i, num in tqdm(enumerate(num_samples)):
#     # print(i/len(num_samples))
#     Ts1D[i], Ts1D_err[i], n = estimate_transmissivity_1D_model(num)
#     Ts3D[i], Ts3D_err[i], n = estimate_transmissivity_3D_model(num)
# # Transmisivnost

# fig = plt.figure(figsize=(10, 5))

# ax = fig.add_subplot(1, 1, 1)
# ax.grid(alpha=0.8)
# ax.set_xscale('log')
# ax.set_xlabel(r'$T$')
# ax.errorbar(num_samples, Ts1D, yerr=Ts1D_err, capsize=3, marker='.', markersize=7, color='red', label=r'1D model')
# ax.errorbar(num_samples, Ts3D, yerr=Ts3D_err, capsize=3, marker='.', markersize=7, color='blue', label=r'3D model')
# ax.legend()
# plt.savefig("./Images/3_prepustnost_n")
# plt.show()

# # PREPUSTNOST V ODVISNOSTI OD PROSTE POTI
# lambdas = np.linspace(0, 2, 40)
# print(lambdas)
# num = 10**4

# Ts1D = np.zeros_like(lambdas, dtype=float)
# Ts3D = np.zeros_like(lambdas, dtype=float)

# for i, lmbd in tqdm(enumerate(lambdas)):
#     Ts1D[i], _, _ = estimate_transmissivity_1D_model(num, lam=lmbd)
#     Ts3D[i], _, _ = estimate_transmissivity_3D_model(num, lam=lmbd)

# plt.plot(lambdas, Ts1D, "o--", markersize=3, color="green", label="Naprej 1D")
# plt.plot(lambdas, 1-Ts1D, "o--",  markersize=3, color="green", label="Nazaj 1D")
# plt.plot(lambdas, Ts3D, "o-",  markersize=3, color="orange", label="Naprej 3D")
# plt.plot(lambdas, 1-Ts3D, "o-",  markersize=3, color="orange", label="Nazaj 3D")

# plt.legend()
# plt.xlabel(r"$\lambda$")
# plt.ylabel("T")
# plt.savefig("./Images/3_prepustnost_lmbd")
# plt.show()


# # KOTNA PORAZDELITEV IZOTROPNEGA SIPANJA

