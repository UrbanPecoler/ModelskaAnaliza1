import numpy as np
import time
import matplotlib.pyplot as plt

from tqdm import tqdm

# TODO 
# POPRAVI GRAFE
# 

def transmitivnost(num_samples, lmbd=1):
    escaped = 0

    for _ in range(num_samples):
        theta = np.arccos(2.*np.random.rand()-1.)
        r = (np.random.rand())**(1./3.)

        d = -r * np.cos(theta) + np.sqrt(1 - r**2 * (1 - np.cos(theta)**2))
        s = np.random.exponential(lmbd)

        if s > d:
            escaped += 1

    t = float(escaped) / float(num_samples)
    t_err = 1. / float(num_samples)**(3./2.) * np.sqrt(float(num_samples)*float(escaped) - float(escaped)**2)

    return t, t_err

# DODAJ ŠE 1-T
def plot_transmitivnost(lmbd):
    num_samples = np.logspace(3, 7, dtype=int, num=100)

    ts = np.zeros_like(num_samples, dtype=float)
    ts_err = np.zeros_like(num_samples, dtype=float)

    for i, num in tqdm(enumerate(num_samples)):
        ts[i], ts_err[i] = transmitivnost(num, lmbd=lmbd)


    plt.figure(figsize=(10, 5))

    plt.errorbar(num_samples, ts, yerr=ts_err, capsize=3, marker='.', markersize=7)
    plt.xscale('log')
    plt.xlabel(r'$N$')
    plt.ylabel(r'$T$')
    plt.savefig("./Images/t_n")
    plt.show()


plot_transmitivnost(1)

def lim_t(num):
    lambdas = np.logspace(-3, 3)
    ts= []
    for lmbd in tqdm(lambdas):
        # print(lmbd)
        ts.append(transmitivnost(num, lmbd=lmbd)[0])
    
    plt.plot(lambdas, ts, label="absorbcija")
    plt.plot(lambdas, 1-np.array(ts), label="pobeg")
    plt.xscale("log")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("T")
    plt.legend()
    plt.savefig("./Images/t_lambda")
    plt.show()

lim_t(100000)