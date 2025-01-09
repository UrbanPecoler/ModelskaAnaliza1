import numpy as np

from utils.io_utils import load_data, load_rel_data
from helper import generate_F, generate_Q
from config import data_path, control_path, rel_data_path, R, modify, ModifyParams, H_rel
from plotting import plot_data, plot_kalman_spec_n, plot_kalman_mul_n, err_spektrogram, plot_rel_kalman, cov_comparison

# TODO
# - Popravi dinamično klicanje parametrov delta_n
# - Dodaj analiza plots za relative data v main analiza funkcijo
# - Dodaj možnost save=True/False, show=True/False

# PODATKI
t, measurements, control, a_x, a_y = load_data(data_path, control_path)
rel_data, at, ar = load_rel_data(rel_data_path)

# Časovni korak
dt = t[1] - t[0]

# GENERACIJA MATRIK
F = generate_F(dt)
Q = generate_Q(dt)
c_ = np.column_stack((np.zeros_like(measurements[:, 0]), 
                      np.zeros_like(measurements[:, 0]), 
                      a_x * dt, a_y * dt))


"""# 1) PLOT DATA """
plot_data(t, measurements, control, a_x, a_y)

""" 2) ANALIZA OSNOVNI KALMAN """
modify = "H"
params = ModifyParams(delta_n = 0)
plot_kalman_spec_n(t, measurements, control, c_, F, Q, R, params=params, modify=modify, title="basic_kal")

""" FIKSNI DELTA N ZA LE H IN H+c"""
params = ModifyParams(delta_n = 10)

modify = "H"
plot_kalman_spec_n(t, measurements, control, c_, F, Q, R, params=params, modify=modify, title="modify_H_10")

modify = "both"
plot_kalman_spec_n(t, measurements, control, c_, F, Q, R, params=params, modify=modify, title="modify_both_10")

""" 3) PLOT KALMAN Z NEKIM RANGE DELTA N
IN PLOT RESIDUALI + KORELACIJA ZA SPECIFIČNE DELTA N"""
modify = "H"
plot = "H"
params = ModifyParams(delta_n=[0, 5, 10])
plot_kalman_mul_n(t, measurements, control, c_, F, Q, R, modify=modify, params=params, plot=plot, title="delta_0-5-10")

"""# 4) PLOT KALMAN PRVIČ BREZ MERITEV HITROSTI"""

"""# BREZ MERITEV POTI"""
modify = "x"
params = ModifyParams(delta_n=0, delta_x=100000, delta_v=0)
plot_kalman_spec_n(t, measurements, control, c_, F, Q, R, params=params, modify=modify, title="brez_poti")
plot = "H"
params = ModifyParams(delta_x=100000, delta_v=[0, 5, 10])
plot_kalman_mul_n(t, measurements, control, c_, F, Q, R, modify=modify, params=params, plot=plot, title="brez_poti_0-5-10")


"""# BREZ MERITEV HITROSTI"""
# # TODO KOVARIANČNE MATRIKE MI NE IZPIŠE PRAV
modify = "v"
params = ModifyParams(delta_n=0, delta_x=0, delta_v=1000000)
plot_kalman_spec_n(t, measurements, control, c_, F, Q, R, params=params, modify=modify, title="brez_hitrosti")
plot = "H"
params = ModifyParams(delta_x=[0, 5, 10])
plot_kalman_mul_n(t, measurements, control, c_, F, Q, R, modify=modify, params=params, plot=plot, title="brez_hitrosti_0-5-10")

# 5) SPEKTROGRAM NAPAKE POTI IN HITROSTI DELTA N IN DELTA V
modify = "v"
params = ModifyParams(delta_n=0, delta_v=np.arange(1, 50), delta_x=np.arange(1, 50))
err_spektrogram(t, measurements, control, c_, F, Q, R, modify=modify, params=params)

""" 3. NALOGA """

"""# GENERACIJA MATRIK"""
# SIMPLE Q
method="simple"
plot_rel_kalman(t, measurements, rel_data, at, ar, control, F, H_rel, R, method=method)

# # NATANČEN Q
method="hard"
plot_rel_kalman(t, measurements, rel_data, at, ar, control, F, H_rel, R, method=method)

# # PRIMERJAVA KOVARIANČNE MATRIKE OBEH METOD
cov_comparison(t, rel_data, at, ar, control, F, H_rel, R)
