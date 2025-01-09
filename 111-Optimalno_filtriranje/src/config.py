import numpy as np
from dataclasses import dataclass
from typing import Union, List



# DATA PATH
data_path = "./data/kalman_cartesian_data.dat"
control_path = "./data/kalman_cartesian_kontrola.dat"
rel_data_path = "./data/kalman_relative_data.dat"

# PARAMETRI
sigma_xy = 25.
sigma_ax = 0.05
sigma_ay = 0.05
sigma_v = 1/3.6 # vrednost potem računam sproti

# MATRIKE SISTEMA
R = np.diag([sigma_xy**2, sigma_xy**2, sigma_v**2, sigma_v**2])
H = np.eye(4)

# ZAČETNA STANJA
# KARTEZIČNO
x = np.array([0, 0, 0, 0])  # začetno stanje [x, y, vx, vy]
P = np.eye(4)  # začetna kovariančna matrika

# RELATIVNO
vx_rel = 0.001
vy_rel = 0.001
P_rel = np.diag((sigma_xy, sigma_xy, 1000, 1000))
H_rel = np.diag((1, 1, 0, 0))
rotacija = np.array([[0, -1],
                    [1, 0]])   # Rotacija za pi/2

# ANALIZA - MODIFY PARAMETRI
modify = "none"
plot = "none"
method = "none"
title = "none"


@dataclass
class ModifyParams:
    delta_n: Union[int, List[int]] = 0
    delta_v: Union[int, List[int]] = 0
    delta_x: Union[int, List[int]] = 0


# =================================================================
# PLOTTING CONFIG
traj_figsize = (15, 6)