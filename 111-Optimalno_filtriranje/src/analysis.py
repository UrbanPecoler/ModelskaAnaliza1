import numpy as np

from config import modify, sigma_xy, sigma_v, ModifyParams, method
from helper import generate_H_and_c, calculate_error, calculate_distance
from config import x, P, P_rel, vx_rel, vy_rel, sigma_ax, rotacija

def data_analysis(t, measurements, control):    
    N = len(measurements[:, 1])  # število meritev
    err_measurements = np.zeros((N, 2)) 
    dist_measurements = np.zeros((N,4))
    for i in range(N):
        err_measurements[i] = calculate_error(measurements[i], control[i])
        dist_measurements[i] = calculate_distance(measurements[i], control[i])
    
    return err_measurements, dist_measurements



def kalman_filter(z, x, c, P, F, H, Q, R):
    # Napoved
    x_pred = F @ x + c
    P_pred = F @ P @ F.T + Q
    
    # Posodobitev
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    x = x_pred + K @ (z - H @ x_pred)
    P = (np.eye(len(K)) - K @ H) @ P_pred
    
    resid = np.linalg.norm(z - H @ x_pred)
    
    return x, P, resid

def kalman_rel_simulation(t, rel_data, at, ar, control, F, H, R, method=method):
    N = len(control[:, 1])  # število meritev
    traj = np.zeros((N, 4))  # za shranjevanje stanj
    dt = t[1] - t[0]
    x = np.array([rel_data[0, 0], rel_data[0, 1], vx_rel, vy_rel])
    P = P_rel
    
    Ps = np.zeros((N, 4, 4))
    err_measurments = np.zeros((N, 2)) 
    err_kalman = np.zeros((N,2))
    dist_kalman = np.zeros((N,4))
    for i in range(N):
        _,  _, vx, vy = x
        v = np.sqrt(vx**2 + vy**2)
        u = np.array([at[i]*dt**2, ar[i]*dt**2, at[i]*dt, ar[i]*dt])
        B = np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0, 0, vx/v, -vy/v],
                      [0,0,vy/v,vx/v]])
        c = B @ u
        if method == "simple":
            Q = dt**2 * sigma_ax * np.eye(4)
        else:
            v_vec = np.array([vx, vy])
            v_star = v_vec @ rotacija
            a_vec = np.array([at[i], ar[i]])
            a_star = a_vec @ rotacija
            # print(B[2:, 2:] @ a_star)
            Q = dt**2 * (sigma_ax*np.eye(2) + ((v_star @ P[:2, :2] @ v_star) / v**4) * np.tensordot((B[2:, 2:] @ a_star),(B[2:, 2:] @ a_star), axes=0))
            Q = np.block([[Q, np.zeros_like(Q)], [np.zeros_like(Q), Q]])
        z = np.array([rel_data[i, 0], rel_data[i, 1], 0, 0])
        x, P, resid = kalman_filter(z, x, c, P, F, H, Q, R)
        traj[i] = x

        err_measurments[i] = calculate_error(z, control[i])
        err_kalman[i] = calculate_error(x, control[i])

        dist_kalman[i] = calculate_distance(x, control[i])
        Ps[i] = P
    return traj, Ps, err_kalman, dist_kalman




def kalman_simulation(t, measurements, control, c_, F, Q, R, params=None, modify=modify):
    global x, P
    # print(f"Received params: {params}")
    if params is None or not isinstance(params, ModifyParams):
        params = ModifyParams()
    # print(f"Final params: {params}")  # Preveri, če se params prepiše

    delta_n = params.delta_n
    delta_v = params.delta_v
    delta_x = params.delta_x
    global N
    N = len(measurements[:, 1])  # število meritev
    traj = np.zeros((N, 4))  # za shranjevanje stanj
    resid_sez = np.zeros(len(measurements[:, 1]))
    Ps = np.zeros((N, 4, 4))
    err_measurments = np.zeros((N, 2)) 
    err_kalman = np.zeros((N,2))
    dist_kalman = np.zeros((N,4))
    Hs, c__ = generate_H_and_c(t, c_, params=params, modify=modify)
    # print(Hs)
    for i in range(N):
        vx, vy = measurements[i, 2], measurements[i, 3]
        sigma_v_i = 0.01 * np.sqrt(vx**2 + vy**2)
        sigma_v_max = max(sigma_v_i, sigma_v) 
        R = np.diag([sigma_xy**2, sigma_xy**2, sigma_v_max**2, sigma_v_max**2])
        c = c__[i]
        H = Hs[i]
        z = measurements[i]
        x, P, resid = kalman_filter(z, x, c, P, F, H, Q, R)
        traj[i] = x
        resid_sez[i] = resid

        err_measurments[i] = calculate_error(z, control[i])
        err_kalman[i] = calculate_error(x, control[i])

        dist_kalman[i] = calculate_distance(x, control[i])
        # print("")
        Ps[i] = P
    
    # Postavim nazaj na začetne pogoje
    x = np.array([0, 0, 0, 0])  
    P = np.eye(4) 

    return traj, Ps, resid_sez, err_kalman, dist_kalman