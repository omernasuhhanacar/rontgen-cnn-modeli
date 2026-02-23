import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("  Gercek Hastane Verileri Yukleniyor...\n")

# Veri setimizin bulundugu klasorun yolu
egitim_klasoru = 'chest_xray/train'

# 1. BINLERCE FOTOGRAFI OTOMATIK OKUMA VE HAZIRLAMA
# Tensorflow bu komutla klasordeki tum resimleri bulur, 150x150 yapar ve 32'serli gruplar halinde hazirlar.
egitim_verisi = tf.keras.utils.image_dataset_from_directory(
    egitim_klasoru,
    color_mode='grayscale', # Siyah-beyaz okuyoruz
    image_size=(150, 150),  # Yapay zekanin standart boyutu
    batch_size=32           # Ekran kartina ayni anda 32 resim gonderilecek
)

# Pikselleri 0-255'ten 0-1 arasina (Normalizasyon) cekiyoruz
normalizasyon_katmani = tf.keras.layers.Rescaling(1./255)
egitim_verisi = egitim_verisi.map(lambda x, y: (normalizasyon_katmani(x), y))

print("\n  Yapay Zeka Beyni Insa Ediliyor...\n")

# 2. CNN MİMARİSİ (Ayni guclu doktor beyni)
model = models.Sequential([
    layers.Input(shape=(150, 150, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid') # 0: Normal, 1: Zaturre
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. BUYUK ANTRENMAN BASLIYOR!
print("-" * 50)
print("  GERCEK VERILERLE EGITIM BASLADI! (Ekran kartiniz calismaya basliyor)")
print("-" * 50)

# 5216 adet gercek egitim fotografi var. 3 tur (epoch) dondurecegiz.
gecmis = model.fit(egitim_verisi, epochs=3)

print("\n  Egitim Tamamlandi! Modeli basariyla gercek verilerle egitiniz.")