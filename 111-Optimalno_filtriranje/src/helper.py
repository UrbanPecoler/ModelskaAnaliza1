import numpy as np

from config import sigma_ax, sigma_ay, H, ModifyParams

# CONFIG HELPERS
def generate_F(dt):
    return np.array([[1, 0, dt, 0], 
                     [0, 1, 0, dt], 
                     [0, 0, 1, 0], 
                     [0, 0, 0, 1]])

def generate_Q(dt):
    return np.diag([0, 0, sigma_ax**2 * dt**2, sigma_ay**2 * dt**2])


def generate_H_and_c(t, c_, params=None, modify="none"):
    if params is None or not isinstance(params, ModifyParams):
        params = ModifyParams()
    delta_n = params.delta_n
    delta_v = params.delta_v
    delta_x = params.delta_x

    Hs = np.full((len(t), 4, 4), H)  

    # Handle modification based on 'modify'
    if modify in ["H", "h", "both"]:
        for i in range(0, len(t), delta_n+1):  
            Hs[i+1:i+delta_n+1] = np.zeros((4, 4)) 
    
    if modify in ["c", "both"]:
        for i in range(0, len(c_), delta_n+1):  
            c_[i+1:i+delta_n+1] = np.zeros(4) 
    
    if modify in ["v", "hitrost", "r", "x", "pot"]:
        # IZGUBIMO MERITEV HITROSTI
        if delta_v < len(t):
            if delta_v == 0:
                pass
                # # Hs = np.full((len(t), 4, 4), H)
                # Hs -= np.array([[0, 0, 0, 0],
                #                 [0, 0, 0, 0],
                #                 [0, 0, 1, 0],
                #                 [0, 0, 0, 1]])
            else:
                for i in range(0, len(t), delta_v+1):  
                    # print(i)
                    Hs[i+1:i+delta_v+1] -= np.diag((1, 1, 0, 0))
        elif delta_v >= len(t):
            Hs -= np.diag((1, 1, 0, 0))
        
        # IZGUBIMO MERITEV LEGE
        if delta_x < len(t):
            if delta_x == 0:
               pass
            else:    
                for i in range(0, len(t), delta_x+1):  
                    Hs[i+1:i+delta_x+1] -= np.diag((0, 0, 1, 1))
        elif delta_x >= len(t):
            Hs -= np.diag((0, 0, 1, 1))

    return Hs, c_

def unpack_params(params):
    if params is None or not isinstance(params, ModifyParams):
        params = ModifyParams()
    return params.delta_n, params.delta_v, params.delta_x


# =====================================================================
# CALCULATION HELPERS
def calculate_error(data, exact):
    x, y, vx, vy = data
    x_kon, y_kon, vx_kon, vy_kon = exact
    err_pot = np.sqrt((x-x_kon)**2 + (y-y_kon)**2)
    err_v = np.sqrt((vx - vx_kon)**2 + (vy - vy_kon)**2)
    return np.array([err_pot, err_v])

def calculate_distance(data, exact):
    x, y, vx, vy = data
    x_kon, y_kon, vx_kon, vy_kon = exact
    x_dist = x-x_kon
    y_dist = y-y_kon
    vx_speed = vx - vx_kon
    vy_speed = vy - vy_kon
    return np.array([x_dist, y_dist, vx_speed, vy_speed])


def hist_norm(lst):
    # mean_err = np.mean(lst)  # Povprečje napak
    # std_err = np.std(lst)    # Standardni odklon napak
    # norm_err = (lst - mean_err) / std_err
    norm_err = lst
    return norm_err




# =====================================================================
# PLOTTING HELPERS

def add_zoom(ax):
    # inset Axes....
    x1, x2, y1, y2 = 815, 1484, -2204, -1068  # območje ki ga zoomiram

    axins = ax.inset_axes(
        [0.33, 0.15, 0.47, 0.57], # Pozicija grafa
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    
    return axins
