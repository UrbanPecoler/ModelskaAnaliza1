import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
from spectrum import aryule, arma2psd
import matplotlib.pyplot as plt

# DATA HELPERS
def data_unpack(data):
    if isinstance(data[0], np.float64):
        t, y = np.arange(len(data)), data
    else:
        t, y = data[:, 0], data[:, 1]
    return t, y

def parabola(a, b, c, x):
    return a * x**2 + b*x + c

def linearno(a, b, x):
    return a * x + b

def kubicno(a, b, c, d, x):
    return a*x**3 + b*x**2 + c*x + d

def fit(x, y, fun):
    # začetna vrednost parametrov = 1
    p0 = np.ones(fun.__code__.co_argcount - 1)  # odštejem 1 zaradi x argumenta
    params, cov_matrix = curve_fit(fun, x, y, p0=p0)
    return params, cov_matrix

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

def co2_y_fit(data, fun):
    t, y = data_unpack(data)
    print(fun)
    params, _ = fit(t, y, fun)
    y_fit = plot_fit_fun(t, y, fit_fun=fun, 
                    fit_params=params, plot=False)
    return y - y_fit
    
# ================================================================================
# TESTNI SIGNAL


# ================================================================================
# ANALIZA MME HELPERS
def fft(x, y):
    f = abs(np.fft.fft(y))**2
    f = f/np.max(f)
    freqs = np.fft.fftfreq(len(y), d=x[1]-x[0])
    return f[:len(y)//2], freqs[:len(y)//2]

def mem_computation(y, p, norm=True):
    AR, _, _ = aryule(y, p)
    poli = np.roots(np.concatenate(([1], -AR)))
    p_plot = arma2psd(AR, NFFT=len(y), norm=norm)[:len(y)//2]
    return AR, poli, p_plot


def stabilize_poles(a):
    poles = np.roots(np.concatenate(([1], -a)))
    stabilized_poles = [z / abs(z) if abs(z) > 1 else z for z in poles]    
    stabilized_a = -np.poly(stabilized_poles)[1:]  
    return np.conjugate(stabilized_a)


def autocorrelation(y, max_lag):
    """Izračun avtokorelacijskih koeficientov R(k)."""
    N = len(y)
    R = []
    for k in range(max_lag + 1):
        R_k = np.sum(y[k:] * y[:N-k])
        R.append(R_k)
    return np.array(R)

def calculate_g2(y, p_values):
    """Izračun G^2 za različne rede AR modela."""
    g2_values = []

    for p in p_values:
        AR, _, _ = aryule(y, p)  
        R = autocorrelation(y, p)

        G2 = R[0] + np.sum(AR * R[1:])  
        g2_values.append(G2)
    return g2_values

def peaks_analysis(x):
    peaks, properties = find_peaks(x, height=np.max(x)/5)
    results_half = peak_widths(x, peaks, rel_height=0.5)
    return peaks, results_half[0]
    # FWHM IZRAČUN


# ================================================================================
# 2. NALOGA
def predict_signal(signal, ak):
    """
    Napove signal na podlagi danih koeficientov a_k.
    """
    p = len(ak)
    n = len(signal//2)
    prediction = np.zeros(n)
    
    for i in range(p, n):
        prediction[i] = -np.sum(ak * signal[i-p:i][::-1])
        # prediction[i] = -np.sum(signal[:-p-1:-1])
    # for i in range(n):
    #         y = -np.sum(signal[:-p-1:-1] * ak)
    #         prediction = np.hstack([signal, y])
    return prediction


def calculate_errors(original, predicted):
    """
    Izračuna absolutno napako v vsaki točki in povprečno napako.
    """
    abs_error = np.abs(original - predicted)
    avg_error = np.mean(abs_error)
    return abs_error, avg_error
