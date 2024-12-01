import numpy as np
import matplotlib as plt
from time import time

# Sistem si prevedam na brezdimenzijskega (na papirju)
def diff_eqs(t, vec, k, s):
    a, a_star, b = vec
    dadt = -a**2 + k * a * a_star
    dasdt = a**2 - k * a * a_star - k * s * a_star
    dbdt = k * s * a_star

    return [dadt, dasdt, dbdt]
