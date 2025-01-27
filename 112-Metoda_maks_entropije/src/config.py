from matplotlib import cm
import numpy as np

# IO CONSTANTS
DATA_ROOT = "./data/"
OUTPUT_ROOT = "./Images/"

# TESTNI SIGNAL
def testni_signal(t, omega, delta_omega):
    return t, np.sin(omega*t) + np.sin((omega + delta_omega) * t)

# PLOTTING
def viridis_colors(num):
    return [cm.viridis(i / num) for i in range(num)]

def standard_colors(num):
    color_sez = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
    if num <= len(color_sez):
        return color_sez[:num]
    else:
        raise ValueError("Color sez ni dovolj velik. Uporabi raje viridis_colors()")