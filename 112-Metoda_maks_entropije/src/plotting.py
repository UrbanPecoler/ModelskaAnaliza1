import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from spectrum import aryule, arma2psd

from helpers import *
from config import OUTPUT_ROOT, standard_colors, viridis_colors, testni_signal

def data_graph(x, y, ax=None, plot=True, title=None, label=None, xlabel="t", ylabel="x", color="lightblue"):
    if ax == None:
        ax = plt.gca()
    if plot:
        ax.plot(x, y, color=color, label=label)
    else:
        ax.scatter(x, y, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if label != None:
        ax.legend()
        

# =====================================================================================================
# DATA VISUALIZATION
def plot_fit_fun(x, y, ax=None, fit_fun=None, fit_params=None, plot=True):
    if ax == None:
        ax = plt.gca()
    if fit_fun and fit_params is not None:
        t_fit = np.linspace(x.min(), x.max(), len(y))
        y_fit = fit_fun(t_fit, *fit_params)
        if plot:
            ax.plot(t_fit, y_fit, color='red', linestyle='-', label="Fit")
            ax.legend()
    return y_fit

def plot_import_data(data_lst, titles, save=None):
    fig, axs = plt.subplots(nrows=2, ncols=2, 
                           figsize=(12, 8))
    axs = axs.flatten()
    print(axs)
    for i in range(len(axs)):
        data = data_lst[i]
        title = titles[i]
        t, y = data_unpack(data)

        if i != 3:
            data_graph(t, y, ax=axs[i], title=title)
        if i == 2:
            params, _ = fit(t, y, parabola)
            plot_fit_fun(t, y, ax=axs[i], fit_fun=parabola, fit_params=params)        
        if i == 3:
            y_fit = plot_fit_fun(t, y, ax=axs[i], fit_fun=parabola, 
                            fit_params=params, plot=False)
            y_eff = y - y_fit
            data_graph(t, y_eff, ax=axs[i], title=title)

    plt.tight_layout()
    if save != None:
        plt.savefig(OUTPUT_ROOT + save)
    plt.show()

# =====================================================================================================
# VEČINOMA ZA 0. NALOGO
def plot_test(t_, omega, delta_omega, p, title=None, save=None,):
    t, y = testni_signal(t_, omega, delta_omega)
    f, freq = fft(t, y)
    print(freq)
    data_graph(freq, f, label="FFT")

    AR, _, _ = aryule(y, p)
    p_plot = arma2psd(AR, NFFT=len(y), norm=True)[:len(y)//2]
    data_graph(freq, p_plot, label=f"p = {10}", title=title, color="red")
    plt.xlim([0, 5])
    plt.yscale("log")
    if save != None:
        plt.savefig(OUTPUT_ROOT + save)
    plt.show()





# =====================================================================================================
# VEČINOMA ZA 1. NALOGO

def plot_g2(data, p_max, ax=None, title=None, signal=""):
    if ax == None:
        ax = plt.gca()
    t, y = data_unpack(data)

    p_values = np.arange(p_max)
    g2_values = calculate_g2(y, p_values)
    for p, g2 in zip(p_values, g2_values):
        ax.plot([p, p], [0, g2], color="black", lw=1)
        ax.set_title(title)
    ax.scatter(p_values, g2_values, s=40, color="red", label=fr"$G^2$ za signal {signal}")

    limita = calculate_g2(y, [150])
    ax.hlines(limita, xmin=0, xmax=p_max, linestyles="--", color="green", label=r"Vrednost $G^2$ pri p = 150")
    ax.set_xlabel("p")
    ax.set_ylabel(r"$G^2$")
    ax.legend()


def plot_poles(poles, p, ax=None, marker=None, color="blue", label=None):
    if ax == None:
        ax = plt.gca()

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    circle = Circle((0, 0), 1, color='blue', fill=False, linestyle='--', linewidth=1)
    ax.add_artist(circle)
    l, = ax.plot(-300, -300, label=label, color=color) # SAMO ZATO DA DOBIM LEGEND LABELS
    ax.scatter(np.real(poles), np.imag(poles), marker=marker, label=label, color=color)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_title(f'Poles on the Complex Plane (p = {p})')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    if label != None:
        ax.legend()
    return l


def plot_analysis(data,  ps=None, title=None, save=None):
    t, y = data_unpack(data)
    f, freq = fft(t, y)

    color = standard_colors(len(ps))

    fig = plt.figure(figsize=(15, 7), layout="constrained")
    axs = fig.subplot_mosaic([["freq", "p1", "legend"], ["freq", "p2", "p3"]], width_ratios=[0.5, 0.25, 0.25])
    # axs['freq'].set_aspect("equal")
    data_graph(freq, f, ax=axs["freq"], label="FFT")
    
    for i, p in enumerate((ps if ps is not None else [])):
        AR, poli, p_plot = mem_computation(y, p)
        data_graph(freq, p_plot, ax=axs["freq"], label=f"p = {p}", color=color[i])

        if any(abs(poli) > 1):
            AR_stable = stabilize_poles(AR)
            poli = np.roots(np.concatenate(([1], -AR_stable)))
        if i == 0:
            print(i)
            l1 = plot_poles(poli, p, ax=axs["p1"], color=color[i], label=f"p = {p}")
        elif i == 1:
            l2 = plot_poles(poli, p, ax=axs["p2"], color=color[i], label=f"p = {p}")
        elif i == 2:
            l3 = plot_poles(poli, p, ax=axs["p3"], color=color[i], label=f"p = {p}")
    if len(ps) == 3:
        axs["legend"].axis("off") 
        axs["legend"].legend((l1, l2, l3), (l1.get_label(), l2.get_label(), l3.get_label()), loc="center", prop={'size': 20})

    axs["freq"].set_yscale("log")
    if save != None:
        plt.savefig(OUTPUT_ROOT + save)
    plt.show()


def poles_correction(data, p, save=None):
    t, y = data_unpack(data)
    AR, poli, p_plot = mem_computation(y, p)
    poli_stable = poli
    if any(abs(poli) > 1):
        AR_stable = stabilize_poles(AR)
        poli_stable = np.roots(np.concatenate(([1], -AR_stable)))
    poli = np.around(poli, 4)
    poli_stable = np.around(poli_stable, 4)
    difference_p = np.setdiff1d(poli, poli_stable) # POLI KI SO ZUNAJ KROŽNICE
    difference_p_stable = np.setdiff1d(poli_stable, poli) # POPRAVLJENI POLI
    indices_to_remove = np.isin(poli_stable, difference_p_stable)

    # Odstranimo elemente
    poli_stable = poli_stable[~indices_to_remove]

    plot_poles(difference_p, p, marker="x", color="red")
    plot_poles(difference_p_stable, p, marker="x", color="green")
    plot_poles(poli_stable, p)
    if save != None:
        plt.savefig(OUTPUT_ROOT + save)
    plt.show()


def plot_peak_analysis(data, ps):
    t, y = data_unpack(data)
    colors = standard_colors(3)
    peaks_data = {f"peak{i+1}": [] for i in range(3)}
    peaks_FWHM_data = {f"peak{i+1}_FWHM": [] for i in range(3)}

    for i, p in enumerate((ps if ps is not None else [])):  # Če `ps` ni None, iteriraj, sicer uporabi prazen seznam
        AR, poli, p_plot = mem_computation(y, p, norm=False)
        peaks, FWHM = peaks_analysis(p_plot)
        print(p)
        print(FWHM)
        for j in range(min(3, len(peaks))):  # Obravnavaj do največ 3 vrhove
            peaks_data[f"peak{j+1}"].append(peaks[j])
            peaks_FWHM_data[f"peak{j+1}_FWHM"].append(FWHM[j] if j < len(FWHM) else 0)
    
    print(peaks_data["peak3"])
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 7))
    for i in range(min(3, len(peaks))):
        data_graph(ps, peaks_data[f"peak{i+1}"], ax=ax1, plot=False, label=f"vrh{i+1}", color=colors[i])
        data_graph(ps, peaks_FWHM_data[f"peak{i+1}_FWHM"], ax=ax2, plot=False, label=f"vrh{i+1}", color=colors[i])
    fig.tight_layout()
    plt.show()

# PLOTTING ZA CO2
def plot_co2_trends(data, p=20):
    t, y = data_unpack(data)
    f, freq = fft(t, y)
    trends = [co2_y_fit(data, linearno), co2_y_fit(data, parabola), co2_y_fit(data, kubicno)]
    labels = [r"Linearni trend $ax + b$", r"Kvadratični trend $ax^2 + bx + c$", r"Kubični trend $ax^3 + bx^2 + cx +d $"]
    colors = standard_colors(3)
    data_graph(freq, f, label="FFT")
    for i, y_fit in enumerate(trends):
        AR, poli, p_plot = mem_computation(y, p)
        data_graph(freq, p_plot, label=labels[i], color=colors[i])
    plt.yscale("log")
    plt.show()



# ================================================================================
# 2. NALOGA
def plot_predict_analysis(t, signal, predictions, poles, abs_errors, avg_errors, orders, compare_orders, save=None):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
    ax = axs.flatten()
    # Graf signala in napovedi za različne orderje
    ax[0].plot(signal, label='Original', color='black', linewidth=1.5)
    for i, order in enumerate(compare_orders):
        ax[0].plot(t, predictions[order], label=f'p = {order}')
    ax[0].set_title("Signal in Napoved za različne orderje")
    ax[0].legend()
    
    plot_poles(poles, compare_orders[1], ax=ax[1], label=f"p = {compare_orders[1]}")

    # Graf absolutnih napak
    for i, order in enumerate(compare_orders):
        ax[2].plot(t, abs_errors[:, order], label=f"p = {order}")
    ax[2].set_yscale("log")
    ax[2].set_title("Absolutna Napaka")
    
    # Graf povprečne napake glede na order
    ax[3].plot(orders, avg_errors, marker='o')
    ax[3].set_title("Povprečna Napaka")
    ax[3].set_xlabel("Order")
    ax[3].set_ylabel("Napaka")
    
    fig.tight_layout()
    if save != None:
        plt.savefig(OUTPUT_ROOT + save)
    plt.show()


def predict_analysis(signal, max_order, compare_orders, save=None):
    t, signal = data_unpack(signal)
    if len(signal)%2==0:
        n = len(signal)
    else:
        print("NE")
    print(np.shape(signal))
    train_signal = signal[:n//2]
    test_signal = signal[n//2:]
    t = range(len(train_signal), len(signal))
    orders = range(1, max_order)  # Testiramo za različne orderje (1 do 20)
    avg_errors = []
    abs_errors = np.zeros((len(test_signal), len(orders)))
    predictions = []


    for i, order in enumerate(orders):
        AR, poli, p_plot = mem_computation(train_signal, order)
        if any(abs(poli) > 1):
            AR = stabilize_poles(AR)
            poli = np.roots(np.concatenate(([1], -AR)))
        if order == compare_orders[1]:
            poles = poli
        predicted = predict_signal(train_signal, AR)
        abs_error, avg_error = calculate_errors(test_signal, predicted)
        abs_errors[:, i] = abs_error
        avg_errors.append(avg_error)
        predictions.append(predicted)
    plot_predict_analysis(t, signal, predictions, poles, abs_errors, avg_errors, orders, compare_orders, save=save)

def plot_data_and_psd(data, ps=[5, 10, 15], save=None):
    t, y = data_unpack(data)
    f, freq = fft(t, y)

    color = standard_colors(len(ps))

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
    data_graph(t, y, ax=ax1)
    data_graph(freq, f, ax=ax2, label="FFT")
    
    for i, p in enumerate((ps if ps is not None else [])):
        AR, poli, p_plot = mem_computation(y, p)
        data_graph(freq, p_plot, ax=ax2, label=f"p = {p}", color=color[i])
    ax2.set_yscale("log")
    if save != None:
        plt.savefig(OUTPUT_ROOT + save)
    plt.show()