import cv2
import numpy as np

print("Rontgen Yeniden Boyutlandiriliyor ve Normalize Ediliyor...\n")

# 1. Kaydettigimiz ornek rontgeni tekrar siyah-beyaz okuyoruz
resim_adi = "ornek_rontgen.jpg"
orijinal_resim = cv2.imread(resim_adi, cv2.IMREAD_GRAYSCALE)

# 2. YENIDEN BOYUTLANDIRMA (Resizing)
# Resmi 150 genislik ve 150 yukseklik olacak sekilde kucultuyoruz
yeni_boyut = (150, 150)
kucuk_resim = cv2.resize(orijinal_resim, yeni_boyut)

# 3. NORMALIZASYON (0-255 arasini 0-1 arasina cekme)
# Pikselleri 255'e bolersek, en parlak yer (255) 1.0 olur, en karanlik yer (0) 0.0 olur.
normalize_resim = kucuk_resim / 255.0

print("--- ISLEM ONCESI ---")
print("Orijinal Boyut:", orijinal_resim.shape)
print("Orijinal Piksel Degerleri (Ilk 3x3):\n", orijinal_resim[:3, :3])

print("\n--- ISLEM SONRASI ---")
print("Yeni Boyut:", kucuk_resim.shape)
print("Normalize Piksel Degerleri (Ilk 3x3):\n", np.round(normalize_resim[:3, :3], 2))

# Farkliliklari gormek icin iki resmi de ekrana verelim
print("\nOrijinal ve Kucultulmus resimler acildi. Gorev cubuguna bakin.")
print("Pencereleri kapatmak icin resmin uzerine tiklayip klavyeden BIR TUSA BASIN.")

# Iki pencereyi ayni anda aciyoruz
cv2.imshow("Orijinal Dev Rontgen", orijinal_resim)
cv2.imshow("Yapay Zeka Icin Kucultulmus (150x150)", kucuk_resim)

cv2.waitKey(0)
cv2.destroyAllWindows()