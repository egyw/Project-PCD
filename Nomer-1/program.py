import cv2
import numpy as np
import os
import shutil

# Konfigurasi folder
base_folder = "Nomer-1"
input_folder = os.path.join(base_folder, "Image")
output_folder = os.path.join(base_folder, "Output")

# Membersihkan atau membuat folder output
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Menampilkan daftar gambar yang tersedia
print("Available images:")
for filename in os.listdir(input_folder):
    print(f"- {filename}")

# Meminta input file dari pengguna
input_file = input("Enter the name of the image file (e.g., 'image1.png'): ").strip()
input_path = os.path.join(input_folder, input_file)

# Validasi input file
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Image '{input_file}' does not exist in folder '{input_folder}'.")

def hitung_prosentase_isi_bejana(image_path, output_folder):
    # Membaca gambar
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mengaburkan gambar untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepian menggunakan Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Temukan kontur pada citra
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Bejana tidak ditemukan.")
        return

    # Pilih kontur terbesar (diasumsikan sebagai bejana)
    bejana_contour = max(contours, key=cv2.contourArea)

    # Gambarkan kontur bejana pada gambar asli
    image_with_contour = image.copy()
    cv2.drawContours(image_with_contour, [bejana_contour], -1, (0, 255, 0), 2)

    # Hitung area bejana
    luas_bejana = cv2.contourArea(bejana_contour)

    # Dapatkan bounding box untuk menentukan isi bejana
    x, y, w, h = cv2.boundingRect(bejana_contour)

    # Asumsikan isi bejana >50% dari tinggi
    batas_isi_y = y + int(h / 2)

    # Buat mask untuk menghitung area isi bejana
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [bejana_contour], -1, 255, -1)
    mask[batas_isi_y:, :] = 0

    # Hitung area isi bejana
    luas_isi_bejana = cv2.countNonZero(mask)

    # Hitung prosentase isi bejana
    luas_bejana = luas_bejana - luas_isi_bejana
    prosentase_isi = (luas_isi_bejana / luas_bejana) * 100

    # Buat gambar hasil mask isi bejana
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = [0, 0, 255]
    hasil_image = cv2.addWeighted(image, 1, mask_colored, 0.3, 0)

    # Menyimpan hasil gambar ke folder output
    cv2.imwrite(os.path.join(output_folder, f"{input_file}_original.png"), image)
    cv2.imwrite(os.path.join(output_folder, f"{input_file}_gray.png"), gray)
    cv2.imwrite(os.path.join(output_folder, f"{input_file}_edges.png"), edges)
    cv2.imwrite(os.path.join(output_folder, f"{input_file}_contour.png"), image_with_contour)
    cv2.imwrite(os.path.join(output_folder, f"{input_file}_filled.png"), hasil_image)

    # Menyimpan hasil teks
    result_file = os.path.join(output_folder, f"{input_file}_result.txt")
    with open(result_file, 'w') as f:
        f.write(f"Luas isi bejana: {luas_isi_bejana:.1f}\n")
        f.write(f"Luas total bejana: {luas_bejana:.1f}\n")
        f.write(f"Prosentase isi: {prosentase_isi:.2f}%\n")

    print("Processing complete. Results saved in output folder.")

# Memproses gambar
hitung_prosentase_isi_bejana(input_path, output_folder)
