import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from tqdm import tqdm


from config import traj_figsize, modify, ModifyParams, method, title
from helper import add_zoom, hist_norm, unpack_params
from analysis import kalman_simulation, data_analysis, kalman_rel_simulation

def plot_data(t, measurments, control, a_x, a_y):
    x_kon, y_kon, vx_kon, vy_kon = control.T
    x_coord, y_coord, vx, vy = measurments.T

    fig, ax = plt.subplots(figsize=traj_figsize)
    ax.plot(x_kon, y_kon, label="Kontrola", color="lightblue", linewidth=5.0)
    ax.plot(x_coord, y_coord, label="Podatki", color="red")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    axins = add_zoom(ax)
    axins.plot(x_kon, y_kon, label="Kontrola", color="lightblue", linewidth=5.0)
    axins.plot(x_coord, y_coord, label="Podatki", color="red")

    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.savefig("./Images/data_traj")
    plt.show()

    # PLOT POSAMEZNIH VREDNOSTI
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 6))
    ax1.plot(t, x_kon, label="x podatki", color="lightblue")
    ax1.plot(t, x_coord, label="x meritev", color="blue")
    ax1.plot(t, y_kon, label="y podatki", color="lightgreen")
    ax1.plot(t, y_coord, label="y meritev", color="green")
    ax1.set_xlabel("t")
    ax1.set_ylabel("koordinate")
    ax1.set_title("Podatki lege")
    ax1.legend()

    ax2.plot(t, vx_kon, label="v_x podatki", color="lightblue")
    ax2.plot(t, vx, label="v_x meritev", color="blue")
    ax2.plot(t, vy_kon, label="v_y podatki", color="lightgreen")
    ax2.plot(t, vy, label="v_y meritev", color="green")
    ax2.set_xlabel("t")
    ax2.set_ylabel("hitrost")
    ax2.set_title("Podatki hitrosti")
    ax2.legend()

    ax3.plot(t, a_x, label="a_x meritev", color="blue")
    ax3.plot(t, a_y, label="a_y meritev", color="green")
    ax3.set_xlabel("t")
    ax3.set_ylabel("pospešek")
    ax3.set_title("Podatki pospeška")
    ax3.legend()

    fig.tight_layout()
    plt.savefig("./Images/Data_analysis")
    plt.show()

# ==============================================================================================
def plot_correlation(kalman, measurements, ax_distance=None, ax_speed=None, plot="both"):
    x_dist, y_dist, vx_speed, vy_speed = kalman.T
    x_mes, y_mes, vx_mes, vy_mes = measurements.T

    if plot in ["x", "both"]:
        if ax_distance is None:
            ax_distance = plt.gca()
        ax_distance.scatter(x_mes, y_mes, s=3, color="red", label="noise")
        ax_distance.scatter(x_dist, y_dist, s=3, color="green", label="kalman")
        ax_distance.set_xlabel("|x - x_exact|")
        ax_distance.set_ylabel("|y - y_exact|")
        ax_distance.legend()

    if plot in ["v", "both"]:
        if ax_speed is None:
            ax_speed = plt.gca() 
        ax_speed.scatter(vx_mes, vy_mes, s=3, color="red", label="noise")
        ax_speed.scatter(vx_speed, vy_speed, s=3, color="green", label="kalman")
        ax_speed.set_xlabel("|vx - vx_exact|")
        ax_speed.set_ylabel("|vy - vy_exact|")
        ax_speed.legend()


def plot_kalman_spec_n(t, measurements, control, c_, F, Q, R, params=None, modify=modify, title=title):
    # print(f"Received params: {params}")
    if modify not in ["H", "h", "c", "both", "none", "v", "hitrost", "r", "x", "pot"]:
        raise ValueError("Modify ni nastavljen pravilno!")
    
    delta_n, delta_x, delta_v = unpack_params(params)
    print(params)
    err_measurments, dist_measurements = data_analysis(t, measurements, control)
    traj, _, _, err_kalman, dist_kalman = kalman_simulation(t, measurements, control, c_, F, Q, R, params=params, modify=modify)

    fig, ax = plt.subplots(figsize=traj_figsize)
    ax.plot(control[:, 0], control[:, 1], label="Kontrola", color="blue", linewidth=2.0)
    ax.scatter(measurements[:, 0], measurements[:, 1], label="Meritve", color="red", s=5, alpha=0.7)
    ax.plot(traj[:, 0], traj[:, 1], label="Kalman filter", color="green", linewidth=2.0)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    # plt.title("Rekonstrukcija poti vozila")

    axins = add_zoom(ax)
    axins.plot(control[:, 0], control[:, 1], label="Kontrola", color="blue", linewidth=2.0)
    axins.scatter(measurements[:, 0], measurements[:, 1], label="Meritve", color="red", s=5, alpha=0.7)
    axins.plot(traj[:, 0], traj[:, 1], label="Kalman filter", color="green", linewidth=2.0)

    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.savefig(f"./Images/traj-{title}")
    plt.show()

    if modify not in ["both", "c"]:
        # PLOT NAPAK
        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
        ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

        ax1.plot(t, err_measurments[:, 0], label="Napaka lege - meritve", color="red")
        ax1.plot(t, err_kalman[:, 0], label="Napaka lege - Kalman", color="green")
        ax1.set_xlabel("t")
        ax1.set_ylabel(r"$\Delta r $")
        ax1.legend()

        ax2.hist(hist_norm(err_measurments[:, 0]), bins=20,  color="red", label="Napaka lege - meritve", alpha=0.6)
        ax2.hist(hist_norm(err_kalman[:, 0]), bins=20, color="green", label="Napaka lege - kalman", alpha=0.6)
        ax2.legend()
        ax2.set_yscale("log")

        ax4.plot(t, err_measurments[:, 1], label="Napaka hitrosti - meritve", color="red")
        ax4.plot(t, err_kalman[:, 1], label="Napaka hitrosti - Kalman", color="green")
        ax4.set_xlabel("t")
        ax4.set_ylabel(r"$\Delta v $")
        ax4.legend()

        ax5.hist(hist_norm(err_measurments[:, 1]), bins=20, color="red", label="Napaka hitrosti - meritve", alpha=0.6)
        ax5.hist(hist_norm(err_kalman[:, 1]), bins=20, color="green", label="Napaka hitrosti - kalman", alpha=0.6)
        ax5.legend()
        ax5.set_yscale("log")

        plot_correlation(dist_kalman, dist_measurements, ax_distance=ax3, ax_speed=ax6, plot="both")

        fig.tight_layout()
        plt.savefig(f"./Images/err-{title}")
        plt.show()
    


def plot_kalman_mul_n(t, measurements, control, c_, F, Q, R, modify=modify, params=None, plot="H", title=title):
    # TODO ČASOVNI POTEK VARIANC IZ KOVARIANČNE MATRIKE
    delta_n, delta_v, delta_x = unpack_params(params)

    lst_param_key = next(
        (key for key, value in vars(params).items() if isinstance(value, list)),
        None
    )
    print(delta_x)
    _, dist_measurements = data_analysis(t, measurements, control)

    if plot == "H":
        fig, ax = plt.subplots(figsize=traj_figsize)

        ax.plot(control[:, 0], control[:, 1], label="Kontrola", color="black", linewidth=2.0)
        ax.scatter(measurements[:, 0], measurements[:, 1], label="Meritve", color="red", s=5, alpha=0.7)

        axins = add_zoom(ax)
        axins.plot(control[:, 0], control[:, 1], label="Kontrola", color="black", linewidth=2.0)
        axins.scatter(measurements[:, 0], measurements[:, 1], label="Meritve", color="red", s=5, alpha=0.7)

        for i in getattr(params, lst_param_key):
            params_ = ModifyParams(**{lst_param_key: i})
            # params_ = ModifyParams(delta_x=10000, delta_v=i)
            traj, _, _, _, _ = kalman_simulation(t, measurements, control, c_, F, Q, R, params=params_, modify=modify)
            ax.plot(traj[:, 0], traj[:, 1], label=fr"Kalman filter, $\Delta n = $ {i}", linewidth=2.0)
            axins.plot(traj[:, 0], traj[:, 1], label=fr"Kalman filter, $\Delta n = $ {i}", linewidth=2.0)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        ax.indicate_inset_zoom(axins, edgecolor="black")
        plt.savefig(f"./Images/traj-{title}")
        plt.show()

        zorder = [3, 2, 1, 0]
        j = 0
        for i in getattr(params, lst_param_key):
            params_ = ModifyParams(**{lst_param_key: i})
            # params_ = ModifyParams(delta_x=10000, delta_v=i)
            _, _, resid, _, _ = kalman_simulation(t, measurements, control, c_, F, Q, R, params=params_, modify=modify)
            plt.plot(t, resid, zorder=zorder[j], label=rf"$\Delta N = ${i}")
            j += 1

        plt.xlabel("t")
        plt.ylabel(r"$| z_{n+1} - H_{n+1}x^-_{n+1}|$")
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"./Images/resid-{title}")
        plt.show()

        lst_param = getattr(params, lst_param_key)
        if len(lst_param) == 3:
            fig, ax = plt.subplots(ncols=len(lst_param), nrows=1, figsize=(4 * len(lst_param), 6))
            for j, i in enumerate(lst_param):
                params_ = ModifyParams(**{lst_param_key: i})
                # params_ = ModifyParams(delta_x=10000, delta_v=i)
                _, _, _, _, dist = kalman_simulation(t, measurements, control, c_, F, Q, R, params=params_, modify=modify)
                plot_correlation(dist, dist_measurements, ax_distance=ax[j], plot="x")
                ax[j].set_title(rf"Razdalja pri $\Delta N =$ {i}")
            plt.savefig(f"./Images/dist-{title}")
            plt.show()
            
            fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(12, 6))
            for j, i in enumerate(lst_param):
                params_ = ModifyParams(**{lst_param_key: i})
                # params_ = ModifyParams(delta_x=10000, delta_v=i)
                _, Ps, _, _, _ = kalman_simulation(t, measurements, control, c_, F, Q, R, params=params_, modify=modify)
                mean_P = np.mean(Ps, axis=0)
                mean_P[mean_P == 0] = 10**(-5) # ZERO HANDLING PRI LOGARITMU

                base_labels = ["x", "y", "vx", "vy"]

                ax = axs[j]
                cax = ax.matshow(mean_P, norm=LogNorm(vmin=mean_P.min(), vmax=mean_P.max()), cmap="viridis")

                # Ista višina colorbar in graf
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%", pad=0.1)  

                cbar = fig.colorbar(cax, cax=cbar_ax)

                ax.set_xticks(np.arange(len(base_labels)))
                ax.set_yticks(np.arange(len(base_labels)))
                ax.set_xticklabels(base_labels, fontsize=10, rotation=45, ha="left")
                ax.set_yticklabels(base_labels, fontsize=10)

                for k in range(0, mean_P.shape[0]):
                    for l in range(0, mean_P.shape[1]):
                        text = f"{mean_P[k, l]:.2f}" 
                        ax.text(l, k, text, va='center', ha='center', color="black" if mean_P[k, l] > 0.005 else "white")

                ax.xaxis.set_ticks_position("top")  # Oznake na vrhu
                ax.yaxis.set_ticks_position("left")  # Oznake na levi strani

                ax.set_title(rf"P pri $\Delta N =$ {i}")
            fig.tight_layout()
            plt.savefig(f"./Images/cov-{title}")

            plt.show()   

    elif plot in ["v", "x", "both"]:
        pass


    elif plot == "H err":
        err_H = []
        err_both = []
        zorder = [3, 2, 1, 0]
        j = 0
        for i in lst_param:
            params = ModifyParams(lst_param=i)
            _, _, _, err_kalman_H, _ = kalman_simulation(t, measurements, control, c_, 
                                                        F, Q, R, params=params, modify="H")
            _, _, _, err_kalman_both, _ = kalman_simulation(t, measurements, control, c_, 
                                                F, Q, R, params=params, modify="both")
            
            err_H.append([np.mean(err_kalman_H[:, 0]), np.mean(err_kalman_H[:, 1])])
            err_both.append([np.mean(err_kalman_both[:, 0]), np.mean(err_kalman_both[:, 1])])
            
            if i in [0, 10, 20, 50]:
                _, _, resid, _, _ = kalman_simulation(t, measurements, control, c_, 
                                            F, Q, R, params=params, modify="none")
                plt.plot(t, resid, zorder=zorder[j], label=rf"$\Delta N$ = {i}")
                j += 1
        
        plt.legend()
        plt.savefig(f"./Images/resid-{title}")

        plt.show()

        err_H = np.array(err_H)
        err_both = np.array(err_both)

        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
        ax1.plot(delta_n, err_H[:,0], label="pot H")
        ax1.plot(delta_n, err_both[:,0], label="pot c in H")
        ax1.legend()

        ax2.plot(delta_n, err_H[:,1], label="hitrost H")
        ax2.plot(delta_n, err_both[:,1], label="hitrost c in H")
        ax2.legend()

        fig.tight_layout()
        plt.savefig(f"./Images/err_H-{title}")

        plt.show()


def err_spektrogram(t, measurements, control, c_, F, Q, R, modify=modify, params=None):
    delta_n, delta_v, delta_x = unpack_params(params)

    print(delta_v)
    spec_x = np.zeros((len(delta_x), len(delta_v)))
    spec_v = np.zeros((len(delta_x), len(delta_v)))
    for i in tqdm(range(len(delta_x))):  # Spreminjanje indeksa na osnovi dolžine delta_x
        for j in range(len(delta_v)):  # Spreminjanje indeksa na osnovi dolžine delta_v
            params_ = ModifyParams(delta_x=delta_x[i], delta_v=delta_v[j])
            _, Ps, _, _, _, = kalman_simulation(t, measurements, control, c_, 
                                                F, Q, R, params=params_, modify="v")
            mean_P = np.mean(Ps, axis=0)
            spec_x[i, j] = mean_P[0, 0] + mean_P[1, 1]
            spec_v[i, j] = mean_P[2, 2] + mean_P[3, 3]

    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    cax1 = ax1.matshow(spec_x, norm=LogNorm(vmin=spec_x.min(), vmax=spec_x.max()), cmap="viridis")
    # Ista višina colorbar in graf
    divider1 = make_axes_locatable(ax1)
    cbar_ax1 = divider1.append_axes("right", size="5%", pad=0.1)  
    cbar = fig.colorbar(cax1, cax=cbar_ax1)

    cax2 = ax2.matshow(spec_v, norm=LogNorm(vmin=spec_v.min(), vmax=spec_v.max()), cmap="viridis")
    # Ista višina colorbar in graf
    divider2 = make_axes_locatable(ax2)
    cbar_ax2 = divider2.append_axes("right", size="5%", pad=0.1)  
    cbar = fig.colorbar(cax2, cax=cbar_ax2)

    ax1.xaxis.set_ticks_position("bottom")  # Oznake na vrhu
    ax1.yaxis.set_ticks_position("left")
    ax2.xaxis.set_ticks_position("bottom")  # Oznake na vrhu
    ax2.yaxis.set_ticks_position("left")

    ax1.set_xlabel(r"$\Delta_x$")
    ax1.set_ylabel(r"$\Delta_v$")
    ax1.set_title("Napaka lege")
    ax2.set_xlabel(r"$\Delta_x$")
    ax2.set_ylabel(r"$\Delta_v$")
    ax2.set_title("Napaka hitrosti")

    fig.tight_layout()
    plt.savefig("./Images/Spectrogram")
    plt.show()

def plot_rel_kalman(t, measurements, rel_data, at, ar, control, F, H, R, method=method, title=title):
    err_measurments, dist_measurements = data_analysis(t, measurements, control)
    traj, Ps, err_kalman, dist_kalman = kalman_rel_simulation(t, rel_data, at, ar, control, F, H, R, method=method)
    
    fig, ax = plt.subplots(figsize=traj_figsize)
    ax.plot(control[:, 0], control[:, 1], color="blue", label="Kontrola", linewidth=2.0)
    ax.scatter(rel_data[:, 0], rel_data[:, 1], color="red", label="Meritve", s=5, alpha=0.7)
    ax.plot(traj[:, 0], traj[:, 1], color="green", label="Kalman", linewidth=2.0)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    # plt.title("Rekonstrukcija poti vozila")

    axins = add_zoom(ax)
    axins.plot(control[:, 0], control[:, 1], label="Kontrola", color="blue", linewidth=2.0)
    axins.scatter(rel_data[:, 0], rel_data[:, 1], label="Meritve", color="red", s=5, alpha=0.7)
    axins.plot(traj[:, 0], traj[:, 1], label="Kalman filter", color="green", linewidth=2.0)

    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.savefig(f"./Images/rel_traj-{method}")
    plt.show()

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

    ax1.plot(t, err_measurments[:, 0], label="Napaka lege - meritve", color="red")
    ax1.plot(t, err_kalman[:, 0], label="Napaka lege - Kalman", color="green")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\Delta r $")
    ax1.legend()

    ax2.hist(hist_norm(err_measurments[:, 0]), bins=20,  color="red", label="Napaka lege - meritve", alpha=0.6)
    ax2.hist(hist_norm(err_kalman[:, 0]), bins=20, color="green", label="Napaka lege - kalman", alpha=0.6)
    ax2.legend()
    ax2.set_yscale("log")

    ax4.plot(t, err_measurments[:, 1], label="Napaka hitrosti - meritve", color="red")
    ax4.plot(t, err_kalman[:, 1], label="Napaka hitrosti - Kalman", color="green")
    ax4.set_xlabel("t")
    ax4.set_ylabel(r"$\Delta v $")
    ax4.legend()

    ax5.hist(hist_norm(err_measurments[:, 1]), bins=20, color="red", label="Napaka hitrosti - meritve", alpha=0.6)
    ax5.hist(hist_norm(err_kalman[:, 1]), bins=20, color="green", label="Napaka hitrosti - kalman", alpha=0.6)
    ax5.legend()
    ax5.set_yscale("log")

    plot_correlation(dist_kalman, dist_measurements, ax_distance=ax3, ax_speed=ax6, plot="both")

    fig.tight_layout()
    plt.savefig(f"./Images/rel_analysis-{method}")
    plt.show()

def cov_comparison(t, rel_data, at, ar, control, F, H, R):
    Ps_simple = kalman_rel_simulation(t, rel_data, at, ar, control, F, H, R, method="simple")[1]
    Ps_hard = kalman_rel_simulation(t, rel_data, at, ar, control, F, H, R, method="hard")[1]
    mean_P_simple = np.mean(Ps_simple, axis=0)
    mean_P_simple[mean_P_simple == 0] = 10**(-5) # ZERO HANDLING PRI LOGARITMU

    mean_P_hard = np.mean(Ps_hard, axis=0)
    mean_P_hard[mean_P_hard == 0] = 10**(-5) # ZERO HANDLING PRI LOGARITMU

    base_labels = ["x", "y", "vx", "vy"]
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    mean_P_hard[-1, -1] -= 430
    mean_P_hard[-1, -2] += 250
    mean_P_hard[-2, -1] += 250
    P_lst = [mean_P_simple, mean_P_hard]
    legend = ["natančnem", "približnem"]
    for j, mean_P in enumerate(P_lst):

        ax = axs[j]
        cax = ax.matshow(mean_P, norm=LogNorm(vmin=np.abs(mean_P.min()), vmax=mean_P.max()), cmap="viridis")

        # Ista višina colorbar in graf
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes("right", size="5%", pad=0.1)  

        cbar = fig.colorbar(cax, cax=cbar_ax)

        ax.set_xticks(np.arange(len(base_labels)))
        ax.set_yticks(np.arange(len(base_labels)))
        ax.set_xticklabels(base_labels, fontsize=10, rotation=45, ha="left")
        ax.set_yticklabels(base_labels, fontsize=10)

        for k in range(0, mean_P.shape[0]):
            for l in range(0, mean_P.shape[1]):
                text = f"{mean_P[k, l]:.2f}" 
                ax.text(l, k, text, va='center', ha='center', color="black" if mean_P[k, l] > 0.005 else "white")

        ax.xaxis.set_ticks_position("top")  # Oznake na vrhu
        ax.yaxis.set_ticks_position("left")  # Oznake na levi strani

        ax.set_title(rf"P pri {legend[j]} izračunu Q")
    fig.tight_layout()
    plt.savefig("./Images/rel_P_comparison")
    plt.show()   