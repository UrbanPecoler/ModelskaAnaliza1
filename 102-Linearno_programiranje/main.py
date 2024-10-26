import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import pandas as pd
import seaborn as sns

df = pd.read_excel('zivila.xlsx')
# print(df.head)

zivilo = df['zivilo'].values
energija = df['energija[kcal]'].values
mascobe = df['mascobe[g]'].values
ogljikovi_hidrati = df['ogljikovi hidrati[g]'].values
proteini = df['proteini[g]'].values
ca = df['Ca[mg]'].values
fe = df['Fe[mg]'].values
vitamin_c = df['Vitamin C[mg]'].values
kalij = df['Kalij[mg]'].values
natrij = df['Natrij [mg]'].values
cena = df['Cena(EUR)'].values


def linear_prog(c, A_ub, b_ub, bounds=False):
    if bounds:
        sol = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 2))

        # PRIMER OMEJITVE RADENSKE IN SOLI V PRVEM DELU
        # radenska_idx = np.where(zivilo == 'Radenska')[0][0]
        # sol_idx = np.where(zivilo == 'Sol')[0][0]
        # bounds_list = [(0, None) for _ in range(len(c))]  # Privzeto ni omejitve (0 do neskončnosti)
        # bounds_list[radenska_idx] = (0, 1.5)  # Radenska omejena na 150g
        # bounds_list[sol_idx] = (0, 0.05)        # Sol omejena na 5g
        
        # sol = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_list)
    else:
        sol = linprog(c, A_ub=A_ub, b_ub=b_ub)
    # print(sol)
    # print("Vrednost optimizirane funkcije: ", sol.fun)
    x = sol['x'][sol['x']>0.00001]
    labels = df.index.values
    labels = labels[sol['x']>0.00001]
    return labels, x, sol["x"]

# Histogram
def plot_hist(x, save=""):
    grams = x*100
    total_grams = sum(grams)
    percentages = [(g / total_grams) * 100 for g in grams]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax1 = plt.subplots()

    bars1 = ax1.bar(x - width/2, grams, width, label='Masa hrane (Gram)', color='b', alpha=0.7)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, percentages, width, label='Procent hrane v dieti (%)',
                     color='r', alpha=0.7)

    # ax1.set_xlabel('Food Item')
    ax1.set_ylabel('Masa hrane (Gram)', color='b')
    ax2.set_ylabel('Procent hrane v dieti (%)', color='r')

    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_xticks(x)
    ax1.set_xticklabels(zivilo[labels], rotation=45, ha='right')

    ax2.set_ylim(0, 100)

    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9), frameon=False)

    fig.tight_layout()
    if save != "":
        plt.savefig(f"./Images/hist_{save}")
    plt.show()

def total_var(var, full_x):
    return np.sum(full_x * var)

def plot_heatmap(labels, save=""):
    nutrients = ["masa", energija, mascobe, ogljikovi_hidrati, proteini, ca, fe, 
                 vitamin_c, kalij, natrij, cena]

    results = np.zeros(shape=(len(nutrients), len(labels)))
    for i, nutrient in enumerate(nutrients):
        if i == 0:
            results[0] = x/(sum(x))*100
        else:
            results[i] = (nutrient[labels]*x) / sum(nutrient[labels]*x) * 100

    heatmap_data = pd.DataFrame(results.T, index=zivilo[labels], 
                                columns=["Masa", "Energija", 'Maščobe', 'Ogljikovi hidrati', 'Proteini', 
                                         'Kalcij', 'Železo', 'Vitamin C', 'Kalij', 'Natrij', "Cena"])

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, vmin=0, vmax=100, cmap="YlGnBu", 
                linewidths=.5, cbar_kws={'label': 'Procent'})
    plt.xticks(rotation=45)

    plt.tight_layout()
    if save != "":
        plt.savefig(f"./Images/map_{save}")
    plt.show()

def latex_table(string_list, array1, array2):
    # Start table
    latex_code = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
    latex_code += "Item & Array1 &  Cena (EUR) \\\\\n\\hline\n"

    # Populate rows
    for i in range(len(string_list)):
        latex_code += f"{string_list[i]} & {array1[i]:.2f} & {array2[i]:.2f} \\\\\n"

    # End table
    latex_code += "\\hline\n\\end{tabular}\n\\caption{Your Caption}\n\\end{table}"

    return latex_code

# TODO Poglej še kaj se zgodi, če natrija ne upoštevaš v modelu (Dobiš ogromno količino soli)
# --> PRVI DEL --> Poglej še, če omejiš radenska consumption na 0.5 L in 
#                   sol na 5g (posebej za sol, radensko in za obe skupaj)
# --> Omejim posamezno hrano na 150g, za vse 3 minimizacije



# ==================================================================================
# NIČTI DEL -- MINIMIZACIJA KALORIJ Z GLAVNIMI NUTRIENTI
b_ub = np.array([-70., -310., -50., -1000., -18., 20])
A_ub = np.array([-mascobe, -ogljikovi_hidrati, -proteini, -ca, -fe, np.ones(len(df))])
c = energija

labels, x, full_x = linear_prog(c, A_ub, b_ub)

# plot_hist(x, save="nicta-kal")

# PRVI DEL -- MINIMIZACIJA KALORIJ

b_ub = np.array([-70., -310., -50., -1000., -18., -60., -3500., -500., 2400, 20])
A_ub = np.array([-mascobe, -ogljikovi_hidrati, -proteini, -ca, -fe, 
                 -vitamin_c, -kalij, -natrij, natrij, np.ones(len(df))])
c = energija

labels, x, full_x = linear_prog(c, A_ub, b_ub, bounds=True)

# Podatki za tabelo
# print(energija)
# print(full_x)
# print(f"Živila labels: {zivilo[labels]}")
# print(f"Energija:  {(full_x * energija)[np.nonzero(full_x * energija)]}")
# print(f"Cena: {(full_x * cena)[np.nonzero(full_x * cena)]}")
# print(f"Skupaj energija: {total_var(energija, full_x)}")
# print(f"Skupaj cena: {total_var(cena, full_x)}")
# print(f"skupaj masa: {sum(x)*100}")

sez1 = zivilo[labels]
sez2 = (full_x * energija)[labels]
sez3 = x * 100
sez4 = (full_x * cena)[labels]

# print(latex_table(sez1, sez2, sez3, sez4))


# plot_heatmap(labels, save="brez-prva-kal")
# plot_hist(x, save="brez-prva-kal")

# NOTE Turns out da nerabim navzdol omejit natrija, dobim še vseeno isti rezultat, 

# DRUGI DEL -- MINIMIZACIJA MAŠČOB
c = mascobe
b_ub = np.array([-2000., -310., -50., -1000., -18., -60., -3500., -500., 2400, 20])
A_ub[0] = -energija

labels, x, full_x = linear_prog(c, A_ub, b_ub)
# # Podatki za tabelo
# sez1 = zivilo[labels]
# sez2 = (full_x * energija)[labels]
# sez3 = (full_x * mascobe)[labels]
# sez4 = x * 100
# sez5 = (full_x * cena)[labels]

# # print(latex_table(sez1, sez2, sez3, sez4, sez5))
# print(f"Skupaj energija: {total_var(energija, full_x)}")
# print(f"Skupaj cena: {total_var(cena, full_x)}")
# print(f"skupaj masa: {sum(x)*100}")
# print(f"Skupaj maščobe: {total_var(mascobe, full_x)}")


# print(total_var(energija, full_x))
# plot_hist(x, save="prva-masc")
# plot_heatmap(labels, save="prva_masc")

# TRETJI DEL -- MINIMIZACIJA CENE
c = cena
b_ub = np.array([-2000., -70., -310., -50., -1000., -18., -60., -3500., -500., 2400, 20])
A_ub = np.array([-energija, -mascobe, -ogljikovi_hidrati, -proteini, -ca, -fe, 
                 -vitamin_c, -kalij, -natrij, natrij, np.ones(len(natrij))])

labels, x, full_x = linear_prog(c, A_ub, b_ub)

# # Podatki za tabelo
# sez1 = zivilo[labels]
# sez2 = (full_x * energija)[labels]
# sez4 = x * 100
# sez5 = (full_x * cena)[labels]

# print(latex_table(sez1, sez2, sez4, sez5))
# print(f"Skupaj energija: {total_var(energija, full_x)}")
# print(f"Skupaj cena: {total_var(cena, full_x)}")
# print(f"skupaj masa: {sum(x)*100}")
# print(f"Skupaj maščobe: {total_var(mascobe, full_x)}")

# plot_hist(x, save="prva-cena")
# plot_heatmap(labels, save="prva-cena")

# ==================================================================================================
# ČETRTI DEL
# Omejim food consumption na recimo 150g, pa pogledam za vse tri zgornje primere

# Energija
b_ub = np.array([-70., -310., -50., -1000., -18., -60., -3500., -500., 2400, 20])
A_ub = np.array([-mascobe, -ogljikovi_hidrati, -proteini, -ca, -fe, 
                 -vitamin_c, -kalij, -natrij, natrij, np.ones(len(df))])
c = energija

labels, x, full_x = linear_prog(c, A_ub, b_ub, bounds=True)

# plot_hist(x, save="druga-cal")
# plot_heatmap(labels, save="druga-cal")

# Mascobe
c = mascobe
b_ub = np.array([-2000., -310., -50., -1000., -18., -60., -3500., -500., 2400, 20])
A_ub = np.array([-energija, -ogljikovi_hidrati, -proteini, -ca, -fe, 
                 -vitamin_c, -kalij, -natrij, natrij, np.ones(len(df))])

labels, x, full_x = linear_prog(c, A_ub, b_ub, bounds=True)

# plot_hist(x, save="druga-masc")
# plot_heatmap(labels, save="druga-masc")

# Cena
c = cena
b_ub = np.array([-2000., -70., -310., -50., -1000., -18., -60., -3500., -500., 2400, 20])
A_ub = np.array([-energija, -mascobe, -ogljikovi_hidrati, -proteini, -ca, -fe, 
                 -vitamin_c, -kalij, -natrij, natrij, np.ones(len(natrij))])

labels, x, full_x = linear_prog(c, A_ub, b_ub, bounds=True)
print(total_var(cena, full_x, ))

# plot_hist(x, save="druga-cena")
# plot_heatmap(labels, save="druga-cena")

# PETI DEL -- CENA V ODVISNOSTI OD KALORIJ
# TODO --> POPRAVI GRAF Z CENAMI ZGLED TADEJKOOOO GITHUB (na koncu)
c = energija
A_ub = np.array([-mascobe, -ogljikovi_hidrati, -proteini, -ca, -fe, -vitamin_c, -kalij, -natrij, natrij, np.ones(len(natrij)), -cena, cena])

cene = np.arange(1.7, 18, 0.1)
energija_sez = []
masa_sez = []
for cena_ in cene:
    b_ub = np.array([-70., -310., -50., -1000., -18., -60., -3500., -500., 2400, 20, -cena_-0.1, cena_+0.1])

    labels, x, full_x = linear_prog(c, A_ub, b_ub, bounds=True)
    energija_sez.append(total_var(energija, full_x))
    masa_sez.append(sum(x))

# print(len(energy))
# print(len(energija_sez))
fig, ax1 = plt.subplots()

# Prvi scatter plot za maso (rdeča barva, levi y)
# ax1.scatter(cene, masa_sez, s=1, c="r", label="Masa")
# ax1.set_xlabel("Cena (EUR)")
# ax1.set_ylabel("Masa (g)", color="r")
# ax1.tick_params(axis='y', labelcolor="r")

# ax2 = ax1.twinx()
# ax2.scatter(cene, energija_sez, s=1, c="b", label="Energija")
# ax2.set_ylabel("Energija [kcal]", color="b")
# ax2.tick_params(axis='y', labelcolor="b")

# fig.tight_layout()
# plt.savefig("./Images/energija_cena")
# plt.show()


# ŠESTI DEL -- NOVA DIETA
df = pd.read_excel('LowCarb.xlsx')
# print(df.head)

zivilo = df['zivilo'].values
energija = df['energija[kcal]'].values
mascobe = df['mascobe[g]'].values
ogljikovi_hidrati = df['ogljikovi hidrati[g]'].values
proteini = df['proteini[g]'].values
ca = df['Ca[mg]'].values
fe = df['Fe[mg]'].values
vitamin_c = df['Vitamin C[mg]'].values
kalij = df['Kalij[mg]'].values
natrij = df['Natrij [mg]'].values
cena = df['Cena(EUR)'].values

c = cena
b_ub = np.array([-2500., -111., -156., -218., -1000., -18., -60., -3500., -500., 2400, 20])
A_ub = np.array([-energija, -mascobe, -ogljikovi_hidrati, -proteini, -ca, -fe, 
                 -vitamin_c, -kalij, -natrij, natrij, np.ones(len(natrij))])

labels, x, full_x = linear_prog(c, A_ub, b_ub)
# print(total_var(cena, full_x, ))

sez1 = zivilo[labels]
sez2 = x * 100
sez3 = (full_x * cena)[labels]
print(latex_table(sez1, sez2, sez3))

print(f"Skupaj energija: {total_var(energija, full_x)}")
print(f"Skupaj cena: {total_var(cena, full_x)}")
print(f"skupaj masa: {sum(x)*100}")


# plot_hist(x, save="low_carb-cena")
# plot_heatmap(labels, save="low_carb-cena")