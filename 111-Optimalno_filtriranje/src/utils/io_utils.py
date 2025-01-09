import numpy as np


def load_data(data_path, control_path):
    
    # Import data
    data = np.loadtxt(data_path)
    control = np.loadtxt(control_path)
    
    # Ekstrakcija podatkov
    t = data[:, 0]
    x_coord = data[:, 1]
    y_coord = data[:, 2]
    vx = data[:, 3]
    vy = data[:, 4]
    a_x = data[:, 5]
    a_y = data[:, 6]

    x_kon = control[:, 1]
    y_kon = control[:, 2]
    vx_kon = control[:, 3]
    vy_kon = control[:, 4]

    measurements = np.column_stack((x_coord, y_coord, vx, vy))
    control = np.column_stack((x_kon, y_kon, vx_kon, vy_kon))
    
    return t, measurements, control, a_x, a_y

def load_rel_data(path):
    rel_data = np.loadtxt(path)
    x_rel = rel_data[:, 1]
    y_rel = rel_data[:, 2]
    ax_rel = rel_data[:, 3]
    ay_rel = rel_data[:, 4]

    rel_data = np.column_stack((x_rel, y_rel))

    return rel_data, ax_rel, ay_rel