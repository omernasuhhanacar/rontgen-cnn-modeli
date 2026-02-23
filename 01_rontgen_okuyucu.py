import urllib.request
import cv2

print("Goruntu Isleme Laboratuvarina Hos Geldin!\n")

url = "https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg"
resim_adi = "ornek_rontgen.jpg"

# AJAN MASKESI: Sitenin bizi engellememesi icin kendimizi "Windows'taki bir tarayici" gibi tanitiyoruz.
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
urllib.request.install_opener(opener)

print("Rontgen internetten indiriliyor...")
urllib.request.urlretrieve(url, resim_adi)

# Resmi OpenCV ile okuma (Siyah-Beyaz olarak)
resim_matrisi = cv2.imread(resim_adi, cv2.IMREAD_GRAYSCALE)

# Bilgisayarin resmi nasil gordugunu ekrana basalim
print("\nResmin Boyutlari (Yukseklik, Genislik):", resim_matrisi.shape)
print("Iste bilgisayarin gordugu sayilar (Ilk 5 satir ve sutun):\n", resim_matrisi[:5, :5])

# Resmi ekranda gosterme
print("\nRontgen penceresi acildi! Lutfen gorev cubugunu kontrol edin.")
print("DIKKAT: Pencereyi kapatmak icin X tusuna BASMAYIN! Resmin uzerine tiklayip klavyeden HERHANGI BIR TUSA basin.")

cv2.imshow("Normal Akciger Rontgeni (Kapatmak icin bir tusa bas)", resim_matrisi)
cv2.waitKey(0) # Sen bir tusa basana kadar pencereyi acik tutar
cv2.destroyAllWindows()