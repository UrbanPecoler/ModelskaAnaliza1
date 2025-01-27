import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy.signal import find_peaks, peak_widths, peak_prominences

# IMPORT DATA
data_val2 = np.loadtxt('./Data/val2.dat')
data_val3 = np.loadtxt("./Data/val3.dat")

def plot_data(data1, data2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    ax1.plot(data1)
    ax2.plot(data2)
    
    ax1.set_title("Časovna odvisnost signala val2.dat")
    ax2.set_title("Časovna odvisnost signala val3.dat")
    ax1.set_xlabel("t")
    ax2.set_xlabel("t")
    ax1.set_ylabel("Amplituda")
    fig.tight_layout()
    plt.savefig("./Images/prva_plot_signali")
    plt.show()

# plot_data(data_val2, data_val3)

# ======================================================================================
# OKENSKE FUNKCIJE
def hann_window(n):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))

def bartlett_window(n):
    return 1 - np.abs((np.arange(n) - (n - 1) / 2) / ((n - 1) / 2))

def gauss_window(n):
    sigma = 0.4 * (n - 1) / 2  # standardni odmik za Gaussovo okno
    return np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2) / sigma) ** 2)

def welch_window(n):
    return 1 - ((np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)) ** 2

def no_window(n):
    return np.ones(n)

def DFT(data, window_func, method="forward", naloga="prva"):
    n = len(data)  
    window = window_func(n)  
    windowed_data = data * window

    if method == "forward":
        # Izračun DFT
        fft_result = np.fft.fft(windowed_data)

        # Izračun PSD
        psd = np.abs(fft_result) ** 2 / n

        # Frekvence
        freqs = np.fft.fftfreq(n)
        return freqs[:len(psd)//2], psd[:len(psd)//2]
        
    elif method == "inverse":
        # Izračun IDFT
        idft_result = np.fft.ifft(data)
        return idft_result

def plot_okenske(n):
    t = np.linspace(0, n - 1, n)
    plt.figure(figsize=(10, 6))
    plt.plot(t, hann_window(n), label='Hann', linewidth=2)
    plt.plot(t, bartlett_window(n), label='Bartlett', linewidth=2)
    plt.plot(t, gauss_window(n), label='Gauss', linewidth=2)
    plt.plot(t, welch_window(n), label='Welch', linewidth=2)
    plt.plot(t, no_window(n), label="No window", linewidth=2)
    plt.title(f'Okenske funkcije')
    plt.xlabel('t')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid()
    plt.savefig("./Images/prva_okenske")
    plt.show()

n = 512
# plot_okenske(n)

def plot_DFT():
    datas = [data_val2, data_val3]
    window_functions = [hann_window, bartlett_window, gauss_window, welch_window, no_window]
    legend_names = ["Hann", "Bartlett", "Gauss", "Welch", "No window"]

    cmap = plt.cm.Set1
    color_idx = np.linspace(0, 0.5, len(window_functions))
    colors = [cmap(id) for id in color_idx]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    for i, data in enumerate(datas):
        for j, window_func in enumerate(window_functions):
            freqs, psd = DFT(data, window_func)
            axs[i].plot(freqs, psd, label=legend_names[j], color=colors[j])
            axs[i].set_yscale("log")
            axs[i].legend()
    fig.tight_layout()
    plt.savefig("./Images/prva_dft")
    plt.show()

# plot_DFT()

def plot_okolica_vrhov():
    data = data_val2
    window_functions = [hann_window, bartlett_window, gauss_window, welch_window, no_window]
    legend_names = ["Hann", "Bartlett", "Gauss", "Welch", "No window"]

    cmap = plt.cm.Set1
    color_idx = np.linspace(0, 0.5, len(window_functions))
    colors = [cmap(id) for id in color_idx]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    for j, window_func in enumerate(window_functions):
        freqs, psd = DFT(data, window_func)
        ax1.plot(freqs, psd, label=legend_names[j], color=colors[j])
        ax1.set_yscale("log")
        ax1.set_xlim(left=0.13, right=0.14)
        ax1.set_ylim(bottom=5, top=150)
        ax1.set_title("2. vrh")

        ax2.plot(freqs, psd, label=legend_names[j], color=colors[j])
        ax2.set_yscale("log")
        ax2.set_xlim(left=0.215, right=0.225)
        ax2.set_ylim(bottom=10, top=70)
        ax2.set_title("1. vrh")
    plt.legend()
    fig.tight_layout()
    plt.savefig("./Images/prva_okolica_vrhov")
    plt.show()

# plot_okolica_vrhov()

# ================================================================================================
# # ANALIZA VRHOV
windows = {
    "No Window": no_window,
    "Hann": hann_window,
    "Bartlett": bartlett_window,
    "Gauss": gauss_window,
    "Welch": welch_window
}

# Shranjevanje rezultatov za histogram
peak_data = {"height": [], "FWHM": [], "prominence": []}
window_names = []

# Analiza za vsako okensko funkcijo
for name, window_func in windows.items():
    freqs, psd = DFT(data_val3, window_func)
    
    # Najdi vrhove z izračunom prominenc
    peaks, properties = find_peaks(psd, height=0.1, prominence=0.05)
    peak_heights = properties["peak_heights"]
    prominences = properties["prominences"]

    # Uporabi le najvišja dva vrhova
    if len(peaks) >= 2:
        top_indices = np.argsort(peak_heights)[-4:]
        top_peaks = peaks[top_indices]
    else:
        continue

    # Izračun FWHM za vrhove
    widths = peak_widths(psd, top_peaks, rel_height=0.5)[0]

    # Dodaj podatke za histogram
    peak_data["height"].append(peak_heights[top_indices])
    peak_data["FWHM"].append(widths)
    peak_data["prominence"].append(prominences[top_indices])
    window_names.append(name)

# Risanje histogramov za višino vrha, FWHM in prominenco
def plot_histogram(data, title, ylabel):
    x_labels = ["1. vrh", "2. vrh", "3. vrh", "4. vrh"]
    width = 0.15
    x = np.arange(len(x_labels))
    
    plt.figure(figsize=(10, 6))
    for i, (name, values) in enumerate(zip(window_names, data)):
        plt.bar(x + i * width, values, width=width, label=name)
    
    plt.xticks(x + width * (len(window_names) - 1) / 2, x_labels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./Images/prva_histogram_{ylabel}-4")
    plt.show()

# Višina vrhov
peak_heights_data = np.array(peak_data["height"])
# plot_histogram(peak_heights_data, "Višina vrhov za različna okna", "Višina vrha")

# FWHM vrhov
fwhm_data = np.array(peak_data["FWHM"])
# plot_histogram(fwhm_data, "FWHM vrhov za različna okna", "FWHM")

# Prominenca vrhov
prominence_data = np.array(peak_data["prominence"])
# plot_histogram(prominence_data, "Prominenca vrhov za različna okna", "Prominenca")

# ================================================================================================
# PUŠČANJE
# leakage_data = []
# window_names = []

# for name, window_func in windows.items():
#     freqs, psd = DFT(data_val3, window_func)
    
#     peaks, properties = find_peaks(psd, height=0.1, prominence=0.05)
#     if len(peaks) == 0:
#         continue
    
#     main_peak = peaks[np.argmax(properties["peak_heights"])]

#     window_size = 5
#     main_region = slice(max(main_peak - window_size, 0), min(main_peak + window_size + 1, len(psd)))
#     E_main = np.sum(psd[main_region])

#     E_total = np.sum(psd)

#     leakage = (1 - (E_main / E_total)) - 0.3
#     leakage_data.append(leakage)
#     window_names.append(name)

# plt.figure(figsize=(10, 6))
# plt.bar(window_names, leakage_data, color='skyblue')
# plt.title("Puščanje za različna okna")
# plt.ylabel("Puščanje")
# plt.xlabel("Okenska funkcija")
# plt.grid(axis='y')
# plt.savefig("./Images/prva_puščanje-4")
# plt.show()


# DFT ZA KRAJŠE INTERVALE, MANJ TOČK!!!!!
