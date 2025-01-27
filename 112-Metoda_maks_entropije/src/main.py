import numpy as np
import matplotlib.pyplot as plt

from utils.io_utils import load_data, load_co2, load_Wolf, load_lunar_data
from plotting import *
from config import testni_signal
from helpers import parabola, linearno, kubicno, co2_y_fit

# 0. NALOGA
# FFT + YW ZA DVE RAZLIČNI FREKVENCI
# P(delta_omega), da lahko prepoznaš oba vrhova
"""# TODO ZAKAJ MI VRNE DVA VRHOVA ZA TAKO VELIKE P"""
t = np.linspace(0, 5, 1000)
omega = 10
delta_omega = 10
p = 100
y = plot_test(t, omega, delta_omega, p, title=r"$\Delta \omega = 1$", save="test_omega10")

# 1. NALOGA
val2 = load_data("val2")
val3 = load_data("val3")
co2 = load_co2("co2")

# DATA VISUALIZATION
data1 = [val2, val3, co2, co2]
titles1 = ["val2.dat", "val3.dat", "co2.dat s fitom", "co2.dat z odšteto parabolo"]
plot_import_data(data1, titles1, save="Import_data")

# IZRAČUNAN G^2
p_max = 20
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 7))
plot_g2(val2, p_max, ax=ax1, title="val2", signal="val2")
plot_g2(val3, p_max, ax=ax2, title="val3", signal="val3")
plt.savefig("./Images/g2_val23")
plt.show()

# ANALIZA MODELA Z POLI
ps = [5, 10, 20]
plot_analysis(val3, ps=ps, save="Analiysis_val3")

# POKAŽI 1 PRIMER, KAKO IZGLEDA FILTER, ČE POLI ZUNAJ IN KAKO ČE POLI NOT
p = 40
poles_correction(val2, p, save="popravek_polov")

# NARIŠI VIŠINO VRHA V ODVISNOSTI OD P IN FWHM VRHA V ODVISNOSTI OD P
ps = np.arange(15, 30, 1)
plot_peak_analysis(val2, ps)

# CO2
co2 = co2_y_fit(co2, parabola)

# IZRAČUNAN G^2
p_max = 20
plot_g2(co2, p_max, signal="val2")
plt.savefig("./Images/g2_co2")
plt.show()

# # ANALIZA MODELA Z POLI
ps = [5, 10, 20]
plot_analysis(co2, ps=ps, save="Analisys co2")

# MODEL ZA KVADRATIČNI, LINEARNI in KUBIČNI TREND
p = 30
plot_co2_trends(co2, p=p)

# NAPAKA/RESIDUAL MODELA ZA VSAKO k-to MERITEV



""" 2. NALOGA """
borza = load_data("borza")
wolf = load_Wolf("Wolf_number")
dec, RA = load_lunar_data("luna")


# ZA LUNO, WOLF, BORZO -- PLOT DATA IN PSD
ps = [10, 20, 30]
plot_data_and_psd(RA, ps=ps, save="data_luna-RA")
max_order = 60
compare_orders = [5, 10, 15]
predict_analysis(RA, max_order, compare_orders, save="predict_luna-RA")