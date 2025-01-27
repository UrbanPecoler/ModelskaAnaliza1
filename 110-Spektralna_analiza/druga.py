import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import wiener

naloga = "druga"

# IMPORT DATA
signal0 = np.loadtxt("./Data/signal0.dat") # Nezašumljen signal
signal1 = np.loadtxt("./Data/signal1.dat")
signal2 = np.loadtxt("./Data/signal2.dat")
signal3 = np.loadtxt("./Data/signal3.dat")
data = [signal0, signal1, signal2, signal3]

def plot_data():
    plt.plot(signal3, label="signal 3")
    plt.plot(signal2, label="signal 2")
    plt.plot(signal1, label="signal 1")
    plt.plot(signal0, label="signal 0")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Amplituda")
    plt.savefig("./Images/druga_data")
    plt.show()

plot_data()


# ==========================================================================================
def prenosna_funkcija(t, tau=16):
    r =  1. / (2.*tau) * np.exp(-np.abs(t)/(tau))
    for i in range(int(len(r)/2)):
        r[-i]=r[i]
    return r

def deconvolution(signal, r):
    signal_fft = np.fft.fft(signal)
    r_fft = np.fft.fft(r)
    
    epsilon = 1e-10  # Majhen člen za izogib deljenju z nič
    deconv_fft = signal_fft / (r_fft + epsilon)
    
    deconvolved = np.fft.ifft(deconv_fft)
    return deconvolved.real

def wiener_deconvolution(signal, r, noise_power=0.1):
    signal_fft = np.fft.fft(signal)
    r_fft = np.fft.fft(r)
    
    S = signal * r
    Sf = np.fft.fft(S)
    # Wienerjev filter
    wiener_filter = np.abs(r_fft)**2 / (np.abs(r_fft)**2 + noise_power)
    
    deconv_fft = (signal_fft / (r_fft + 1e-10)) * wiener_filter
    
    deconvolved = np.fft.ifft(deconv_fft)
    return deconvolved.real

def plot_deconvolution(signal):
    t = np.arange(len(signal))
    r = prenosna_funkcija(t)

    plt.plot(t, signal, label="c")
    plt.plot(t, deconvolution(signal, r), label="u z dekonvolucijo")
    plt.legend()
    plt.savefig("./Images/druga_deconvolution")
    plt.show()

plot_deconvolution(signal0)

def plot_Wiener(signal, N, j):
    t = np.arange(len(signal))
    r = prenosna_funkcija(t)

    deconvolved = wiener_deconvolution(signal, r)

    # Risanje grafov
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    ax1.plot(t, signal, label="c")
    ax1.plot(t, deconvolved, label="Wiener deconvolution")
    ax1.legend()


    # FFT za različne komponente
    C = np.abs(np.fft.fft(signal))
    S = np.abs(np.fft.fft(np.convolve(r, signal, mode='same')))
    fft_r = np.fft.fft(r)
    N = np.ones_like(C) * N  
    Phi = np.abs(S) ** 2 / (np.abs(C) ** 2 + 50**2)

    ax2.plot(t, C, label='$|C(f)|^2$', color='blue')
    ax2.plot(t, S, label='$|S(f)|^2$', color='green')
    ax2.plot(t, Phi, label='$\Phi(f)$', color='red')
    ax2.plot(t, N, label='$|N(f)|^2$', color='purple')

    ax2.set_xlabel('Frekvenca')
    ax2.set_ylabel('Amplituda')
    ax2.set_title('Spektri in Wienerjev filter')
    ax2.legend()
    ax2.set_ylim(bottom=10**(-3))
    # ax2.grid(True)
    ax2.set_yscale("log")
   
    fig.tight_layout()
    plt.savefig(f"./Images/druga_Wiener_signal{j}")
    plt.show()

plot_Wiener(signal0, 0.001, 0)
plot_Wiener(signal1, 0.1, 1)
plot_Wiener(signal2, 5, 2)
plot_Wiener(signal3, 10, 3)
