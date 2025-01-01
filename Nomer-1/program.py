import cv2
import numpy as np

def hitung_prosentase_isi_bejana(image_path):
    # Membaca gambar
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mengaburkan gambar untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepian menggunakan Canny
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Tepian Bejana", edges)  # Menampilkan hasil deteksi tepian

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
    prosentase_isi = (luas_isi_bejana / luas_bejana) * 100

    # Buat gambar hasil mask isi bejana
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = [0, 0, 255]
    hasil_image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

    hasil_teks_1 = f"Isi = {luas_isi_bejana:.1f}"
    hasil_teks_2 = f"Total = {luas_bejana:.1f}"
    hasil_teks_3 = f"{prosentase_isi:.2f}%"
    
    (w_teks1, h_teks1), _ = cv2.getTextSize(hasil_teks_1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)
    (w_teks2, h_teks2), _ = cv2.getTextSize(hasil_teks_2, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)
    (w_teks3, h_teks3), _ = cv2.getTextSize(hasil_teks_3, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 2)

    pos_teks1 = ((image.shape[1] - w_teks1) // 2, (image.shape[0] - (h_teks1 + h_teks2 + h_teks3)) // 2)
    pos_teks2 = ((image.shape[1] - w_teks2) // 2, (image.shape[0] - (h_teks2 + h_teks3)) // 2 + h_teks1)
    pos_teks3 = ((image.shape[1] - w_teks3) // 2, (image.shape[0] - h_teks3) // 2 + h_teks1 + h_teks2)

    cv2.putText(hasil_image, hasil_teks_1, pos_teks1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
    cv2.putText(hasil_image, hasil_teks_2, pos_teks2, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)
    cv2.putText(hasil_image, hasil_teks_3, pos_teks3, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)

    # Membuat gambar hasil
    mask_bejana = np.zeros_like(image)
    cv2.drawContours(mask_bejana, [bejana_contour], -1, (0, 255, 0), thickness=cv2.FILLED)

    center_point_image = image.copy()
    M = cv2.moments(bejana_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(center_point_image, (cX, cY), 5, (255, 0, 0), -1)

    # Menampilkan semua tahap
    r_image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2)) 
    r_hasil_image = cv2.resize(hasil_image, (image.shape[1] * 2, image.shape[0] * 2))
    r_center_point_image = cv2.resize(center_point_image, (image.shape[1] * 2, image.shape[0] * 2))
    r_mask_bejana = cv2.resize(mask_bejana, (image.shape[1] * 2, image.shape[0] * 2))
    r_edges = cv2.resize(edges, (image.shape[1] * 2, image.shape[0] * 2))

    # Menampilkan gambar-gambar
    cv2.imshow("Gambar Asli", r_image)
    cv2.imshow("Tepian Bejana", r_edges)
    cv2.imshow("Center Point", r_center_point_image)
    cv2.imshow("Luas Bejana", r_mask_bejana)
    cv2.imshow("Luas Isi Bejana", r_hasil_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"{hasil_teks_1}\n{hasil_teks_2}\n{hasil_teks_3}")

# Contoh penggunaan
image_path = "Image/image1.png"
hitung_prosentase_isi_bejana(image_path)
