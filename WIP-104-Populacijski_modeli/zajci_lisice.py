import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# Parametri modela
alpha = 1.0  # Rastni faktor zajcev
beta = 0.1   # Stopnja plenjenja zajcev s strani lisic
gamma = 1.5   # Umiranje lisic brez plena
delta = 0.075  # Rast lisic zaradi plena

# Lotka-Volterrovih model
def lotka_volterra(t, y, alpha, beta, gamma, delta):
    Z, L = y
    dZdt = alpha * Z - beta * Z * L
    dLdt = -gamma * L + delta * Z * L
    return [dZdt, dLdt]

# Z0 = 40  
# L0 = 9  

# t_span = (0, 30)
# t_eval = np.linspace(t_span[0], t_span[1], 1000)

# solution = solve_ivp(lotka_volterra, t_span, [Z0, L0], t_eval=t_eval, args=(alpha, beta, gamma, delta))

# # Risanje faznega diagrama
# Z = solution.y[0]
# L = solution.y[1]
# t = solution.t

# plt.plot(Z, L, '-b', label="Fazni diagram")
# plt.xlabel("Populacija zajcev (Z)")
# plt.ylabel("Populacija lisic (L)")
# plt.title("Fazni diagram populacij zajcev in lisic")
# plt.legend()
# plt.show()

# # Risanje časovnega poteka populacij
# t_span = (0, 200)
# t_eval = np.linspace(t_span[0], t_span[1], 1000)

# solution = solve_ivp(lotka_volterra, t_span, [Z0, L0], t_eval=t_eval, args=(alpha, beta, gamma, delta))
# Z = solution.y[0]
# L = solution.y[1]
# t = solution.t

# plt.plot(t, Z, '-g', label="Zajci (Z)")
# plt.plot(t, L, '-r', label="Lisice (L)")
# plt.xlabel("Čas")
# plt.ylabel("Populacija")
# plt.title("Časovni razvoj populacij zajcev in lisic")
# plt.legend()

# plt.show()


# =====================================================================
# PREPIŠEM V BREZDIMENZIJSKO OBLIKO
def model(t, y):
    z = y[0]
    l = y[1]
    dldt = p*z*(1-l)
    dzdt = (l/p)*(z-1)
    return [dzdt, dldt]


Z0 = 40  
L0 = 9 
p = 2  
zac_p = np.array([Z0, L0])

t_span = (0, 30)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

print("HA")
solution = odeint(model, zac_p, t_eval)
Z = solution.y[0]
L = solution.y[1]
t = solution.t

print("Z")
plt.plot(Z, L, '-b', label="Fazni diagram")
plt.xlabel("Populacija zajcev (Z)")
plt.ylabel("Populacija lisic (L)")
plt.title("Fazni diagram populacij zajcev in lisic")
plt.legend()
plt.show()
