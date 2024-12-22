import cv2
import numpy as np
import os
import shutil

# Struktur folder
base_folder = "Nomer-1"
input_folder = os.path.join(base_folder, "Image")
output_folder = os.path.join(base_folder, "Output")

# Membersihkan folder output jika ada
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

# Daftar gambar yang tersedia
print("Available images:")
for filename in os.listdir(input_folder):
    print(f"- {filename}")

# Input gambar
input_file = input("Enter the name of the image file (e.g., 'image1.png'): ").strip()
input_path = os.path.join(input_folder, input_file)

if not os.path.exists(input_path):
    raise FileNotFoundError(f"Image '{input_file}' does not exist in folder '{input_folder}'.")

# Membaca gambar
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"Image '{input_file}' could not be loaded. Please check the file format and integrity.")

# Langkah 1: Konversi ke grayscale dan Gaussian Blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Langkah 2: Threshold adaptif
adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
adaptive_thresh_path = os.path.join(output_folder, f"adaptive_thresh_{input_file}")
cv2.imwrite(adaptive_thresh_path, adaptive_thresh)

# Langkah 3: Deteksi tepian menggunakan Canny dengan threshold rendah
edges = cv2.Canny(adaptive_thresh, 10, 50)  # Gunakan threshold rendah
edge_image_path = os.path.join(output_folder, f"edges_{input_file}")
cv2.imwrite(edge_image_path, edges)

# Langkah 4: Deteksi kontur terbesar untuk area bejana
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    raise ValueError("No contours found. Please check the input image.")

# Kontur terbesar dianggap sebagai bejana
bejana_contour = max(contours, key=cv2.contourArea)
bejana_mask = np.zeros_like(gray)
cv2.drawContours(bejana_mask, [bejana_contour], -1, 255, -1)

# Simpan gambar bejana mask
bejana_mask_path = os.path.join(output_folder, f"bejana_mask_{input_file}")
cv2.imwrite(bejana_mask_path, bejana_mask)

# Langkah 5: Deteksi isi bejana
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask_fill = cv2.inRange(hsv_image, (10, 100, 100), (40, 255, 255))  # Sesuaikan warna isi
isi_bejana_mask = cv2.bitwise_and(bejana_mask, mask_fill)

# Simpan gambar isi bejana
filled_image_path = os.path.join(output_folder, f"filled_{input_file}")
cv2.imwrite(filled_image_path, isi_bejana_mask)

# Langkah 6: Hitung area total dan isi
total_area = cv2.countNonZero(bejana_mask)
filled_area = cv2.countNonZero(isi_bejana_mask)

# Hitung persentase isi
percentage = (filled_area / total_area) * 100 if total_area > 0 else 0

# Simpan hasil akhir dengan anotasi
output_image = image.copy()
cv2.putText(output_image, f"Isi: {filled_area}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(output_image, f"Total: {total_area}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(output_image, f"Persentase: {percentage:.2f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Simpan hasil gambar akhir
output_image_path = os.path.join(output_folder, f"result_{input_file}")
cv2.imwrite(output_image_path, output_image)

# Simpan hasil ke file teks
output_text_path = os.path.join(output_folder, f"result_{os.path.splitext(input_file)[0]}.txt")
with open(output_text_path, "w") as text_file:
    text_file.write(f"Isi: {filled_area}\n")
    text_file.write(f"Total: {total_area}\n")
    text_file.write(f"Persentase: {percentage:.2f}%\n")

# Informasi ke pengguna
print(f"Processing complete. Results saved in:")
print(f"- Adaptive Threshold: {adaptive_thresh_path}")
print(f"- Edge Image: {edge_image_path}")
print(f"- Bejana Mask: {bejana_mask_path}")
print(f"- Filled Area Image: {filled_image_path}")
print(f"- Final Image: {output_image_path}")
print(f"- Text: {output_text_path}")
