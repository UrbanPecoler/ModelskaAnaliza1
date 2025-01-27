import numpy as np
import matplotlib.pyplot as plt

from scipy.signal.windows import hann, hamming, lanczos, bartlett
from skimage import color, data, restoration



# Definirajmo dimenzije in nastavitev za polnjenje
nx, ny = 512, 512
N_pad = 200

kernels = ['kernel1.pgm', 'kernel2.pgm', 'kernel3.pgm']
lena_files = [
    ('lena_k1_n0.pgm', 'lena_k1_n4.pgm', 'lena_k1_n8.pgm', 'lena_k1_n16.pgm', 'lena_k1_nx.pgm'),
    ('lena_k2_n0.pgm', 'lena_k2_n4.pgm', 'lena_k2_n8.pgm', 'lena_k2_n16.pgm', 'lena_k2_nx.pgm'),
    ('lena_k3_n0.pgm', 'lena_k3_n4.pgm', 'lena_k3_n8.pgm', 'lena_k3_n16.pgm', 'lena_k3_nx.pgm')
]

def load_image(file_path):
    return np.pad(np.loadtxt(file_path, skiprows=3).reshape((nx, ny)), N_pad, mode='edge')

kernels_data = {f'k{i+1}': load_image(f'./Data/lena_slike/{kernels[i]}') for i in range(len(kernels))}

lena_data = {
    f'lena_k{i+1}': {f'n{j}': load_image(f'./Data/lena_slike/lena_k{i+1}_n{j}.pgm') for j in [0, 4, 8, 16, "x"]}
    for i in range(3)
}

# print(kernels_data.keys())
# print(lena_data)

# ==================================================================================================
# JEDRA IN NJIHOVE FT
k1 = kernels_data["k1"]
k2 = kernels_data["k2"]
k3 = kernels_data["k3"]

k1_f = np.fft.fft2(k1)
k2_f = np.fft.fft2(k2)
k3_f = np.fft.fft2(k3)

def plot_kernels():
    kernels = [k1, k2, k3]
    Kernels_FT = [k1_f, k2_f, k3_f]
    titles = ['kernel1.pgm', 'kernel2.pgm', 'kernel3.pgm']

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for i, (kernel, kernel_ft, title) in enumerate(zip(kernels, Kernels_FT, titles)):
        # Surova slika
        ax = axs[0, i]
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(kernel[N_pad:-N_pad, N_pad:-N_pad], cmap='Greys_r')
        
        # Fourierov prostor
        ax = axs[1, i]
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(np.abs(kernel_ft[N_pad:-N_pad, N_pad:-N_pad]), cmap='Greys_r')
        
    axs[0, 0].set_ylabel('Surova slika')
    axs[1, 0].set_ylabel('Fourierov prostor')

    plt.tight_layout()
    plt.savefig("./Images/tretja_jedra")
    plt.show()

# plot_kernels()

def primerjava_window(data, N_pad=N_pad):
    image, kernel, title = data
    image_size = image[N_pad:-N_pad, N_pad:-N_pad].shape

    no_window = 1
    window_hann = np.outer(hann(image_size[0]), hann(image_size[1]))
    window_hamming = np.outer(hamming(image_size[0]), hamming(image_size[1]))
    window_lanczos = np.outer(lanczos(image_size[0]), lanczos(image_size[1]))
    window_bartlett = np.outer(bartlett(image_size[0]), bartlett(image_size[1]))

    windows = [no_window, window_hann, window_hamming, window_lanczos, window_bartlett]
    titles = ["No window", "Hann", "Hamming", "Lanczos", "Parzen"]
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 5, 3)
    plt.title(f'Original - {title}')
    plt.imshow(image[N_pad:-N_pad, N_pad:-N_pad], cmap='gray')
    plt.axis('off')
    for i, window_ in enumerate(windows):
        window = window_
        image_windowed = image[N_pad:-N_pad, N_pad:-N_pad] * window
        kernel_windowed = kernel[N_pad:-N_pad, N_pad:-N_pad] * window

        deconvolved = restoration.wiener(image_windowed, kernel_windowed, balance=5000000)
        
        # originalne slike (prva vrsta)
        plt.subplot(1, 5, i + 1)
        plt.title(f'{titles[i]}')
        plt.imshow(deconvolved, cmap='gray')
        plt.axis('off')

    # plt.tight_layout()
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)
    plt.savefig("./Images/tretja_primerjavaOken")
    plt.show()

data = (lena_data["lena_k2"]["n4"], k2, 'lena_k2_n4')
# primerjava_window(data)

def plot_lanczos(N_pad=N_pad):
    window_lanczos = np.outer(lanczos(nx+N_pad), lanczos(ny+N_pad))
    plt.title("Lanczos window")
    plt.imshow(window_lanczos)
    plt.savefig("./Images/Lanczos_okno")
    plt.show()
# plot_lanczos()

def plot_images(data, j, N_pad=N_pad):
    image_size = data[0][0][N_pad:-N_pad, N_pad:-N_pad].shape

    # Different Window functions
    window_hann = np.outer(hann(image_size[0]), hann(image_size[1]))
    window_hamming = np.outer(hamming(image_size[0]), hamming(image_size[1]))
    window_lanczos = np.outer(lanczos(image_size[0]), lanczos(image_size[1]))


    # Izbira okna
    # window = window_hann
    window = window_lanczos
    plt.figure(figsize=(12, 6))

    for i, (image, kernel, title) in enumerate(data):
        image_windowed = image[N_pad:-N_pad, N_pad:-N_pad] * window
        kernel_windowed = kernel[N_pad:-N_pad, N_pad:-N_pad] * window

        deconvolved = restoration.wiener(image_windowed, kernel_windowed, balance=5000000)
        
        # originalne slike (prva vrsta)
        plt.subplot(2, 4, i + 1)
        plt.title(f'Original: {title}')
        plt.imshow(image[N_pad:-N_pad, N_pad:-N_pad], cmap='gray')
        plt.axis('off')
        
        # dekonvolucirane slike (druga vrsta)
        plt.subplot(2, 4, i + 5)
        plt.title(f'Deconvolved: {title}')
        plt.imshow(deconvolved, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"./Images/tretja_images{j}")
    plt.show()

images_and_kernels1 = [
    (lena_data["lena_k1"]["n0"], k1, 'lena_k1_n0'),
    (lena_data["lena_k1"]["n4"], k1, 'lena_k1_n4'),
    (lena_data["lena_k1"]["n8"], k1, 'lena_k1_n8'),
    (lena_data["lena_k1"]["n16"], k1, 'lena_k1_n16'),
]

images_and_kernels2 = [
    (lena_data["lena_k2"]["n0"], k2, 'lena_k2_n0'),
    (lena_data["lena_k2"]["n4"], k2, 'lena_k2_n4'),
    (lena_data["lena_k2"]["n8"], k2, 'lena_k2_n8'),
    (lena_data["lena_k2"]["n16"], k2, 'lena_k2_n16'),
]

images_and_kernels3 = [
    (lena_data["lena_k3"]["n0"], k3, 'lena_k3_n0'),
    (lena_data["lena_k3"]["n4"], k3, 'lena_k3_n4'),
    (lena_data["lena_k3"]["n8"], k3, 'lena_k3_n8'),
    (lena_data["lena_k3"]["n16"], k3, 'lena_k3_n16'),
]

plot_images(images_and_kernels1, 1)
plot_images(images_and_kernels2, 2)
plot_images(images_and_kernels3, 3)



