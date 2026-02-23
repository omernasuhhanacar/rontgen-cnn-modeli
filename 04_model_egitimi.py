import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

print("Hastane Verileri (Simulasyon) Hazirlaniyor...\n")

# 1. UYDURMA (SENTETİK) RONTGENLER OLUSTURUYORUZ
# 100 adet Normal Rontgen (Genelde siyah/karanlik)
normal_rontgenler = np.random.rand(100, 150, 150, 1) * 0.3

# 100 adet Zaturre Rontgeni (Icine beyaz, parlak hastalik lekeleri eklenmis)
zaturre_rontgenler = np.random.rand(100, 150, 150, 1) * 0.8

# Verileri ve cevap anahtarlarini birlestiriyoruz
# Normal = 0, Zaturre = 1
X_egitim = np.vstack((normal_rontgenler, zaturre_rontgenler))
y_egitim = np.array([0]*100 + [1]*100)

print("Egitim icin toplam 200 adet rontgen hazir!\n")

# 2. YAPAY ZEKA BEYNINI (CNN) TEKRAR KURUYORUZ (Ayni 18.816 parametrelik beyin)
model = models.Sequential()
model.add(layers.Input(shape=(150, 150, 1)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. VE EGITIM BASLIYOR!
print("-" * 50)
print("ANTRENMAN BASLADI! (Lutfen ciktidaki 'accuracy' degerlerini izleyin)")
print("-" * 50)

# epochs=5 demek, yapay zeka bu 200 resme 5 defa bastan sona bakip calisacak demektir.
gecmis = model.fit(X_egitim, y_egitim, epochs=5, batch_size=10)

# 4. FINAL TESTI 
print("\n" + "-" * 50)
print("Doktor Yapay Zeka Hazir! Simdi hic gormedigi bir hastayi test edelim.")

# Yeni bir zaturre hastasi (parlak lekeli) olusturuyoruz
yeni_hasta = np.random.rand(1, 150, 150, 1) * 0.8
tahmin = model.predict(yeni_hasta)

# Tahmin 0.5'ten buyukse Zaturre (1), kucukse Normal (0) sayilir
if tahmin[0][0] > 0.5:
    print(f"\nTESHIS: ZATURRE (Eminlik Orani: %{tahmin[0][0]*100:.2f})")
else:
    print(f"\nTESHIS: NORMAL CIGER (Eminlik Orani: %{(1-tahmin[0][0])*100:.2f})")