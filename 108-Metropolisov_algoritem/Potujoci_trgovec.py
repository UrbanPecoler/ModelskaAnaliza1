import numpy as np
import matplotlib.pyplot as plt

# Parametri
np.random.seed(42)
num_points = 100
num_iterations = 10000
initial_temperature = 1.0
cooling_rate = 0.5

points = np.random.rand(num_points, 2)

def path_length(points, order):
    ordered_points = points[order]
    return np.sum(np.sqrt(np.sum(np.diff(ordered_points, axis=0)**2, axis=1))) + \
           np.sqrt(np.sum((ordered_points[-1] - ordered_points[0])**2))

order = np.arange(num_points)  
np.random.shuffle(order)
current_length = path_length(points, order)
lengths = [current_length]

temperature = initial_temperature

for step in range(num_iterations):
    # Naključno zamenjaj dve točki na poti
    i, j = np.random.choice(num_points, 2, replace=False)
    new_order = order.copy()
    new_order[i], new_order[j] = new_order[j], new_order[i]
    new_length = path_length(points, new_order)
    
    if new_length < current_length or np.random.rand() < np.exp((current_length - new_length) / temperature):
        order = new_order
        current_length = new_length

    lengths.append(current_length)
    temperature *= cooling_rate  # Zmanjšuj temperaturo

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 1. Vizualizacija poti
axs[0].scatter(points[:, 0], points[:, 1], c='red', label='Mesta')
# for i in range(num_points):
#     axs[0].text(points[i, 0], points[i, 1], f'{i+1}', fontsize=10, ha='right')
path_points = points[np.append(order, order[0])]  # Povežemo zadnjo in prvo točko
axs[0].plot(path_points[:, 0], path_points[:, 1], c='blue', label='Pot')
axs[0].set_title("Pot trgovskega potnika")
axs[0].legend()
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)
axs[0].set_aspect('equal')

# 2. Graf dolžine poti skozi iteracije
axs[1].plot(lengths, c='green')
axs[1].set_title("Dolžina poti v odvisnosti od števila korakov")
axs[1].set_xlabel("Število korakov")
axs[1].set_ylabel("Dolžina poti")
axs[1].grid(True)

plt.tight_layout()
plt.savefig(f"./Images/trgovec_{num_points}")
plt.show()
