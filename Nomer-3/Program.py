import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def applyLpf(image_path, radius):

    img = cv2.imread(image_path)

    if img is None:
        print("Gambar tidak ditemukan!")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

    def low_pass_filter(Color, radius):
        rows, cols = Color.shape
        rowMid, colsMid = rows // 2, cols // 2 

        dft = np.fft.fft2(Color)
        dft_shift = np.fft.fftshift(dft)

        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (colsMid, rowMid), radius, 1, -1)

        filtered_shift = dft_shift * mask

        filtered_ishift = np.fft.ifftshift(filtered_shift)
        filtered_img = np.fft.ifft2(filtered_ishift)
        return np.abs(filtered_img)

    R, G, B = cv2.split(img)

    R_filtered = low_pass_filter(R, radius)
    G_filtered = low_pass_filter(G, radius)
    B_filtered = low_pass_filter(B, radius)

    filtered_img = cv2.merge((R_filtered, G_filtered, B_filtered))
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Gambar Asli")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_img)
    plt.title("Hasil LPF")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

base_folder = "Nomer-3"
input_folder = os.path.join(base_folder, "Image", "hutao.jpeg")
applyLpf(input_folder, radius=30)
