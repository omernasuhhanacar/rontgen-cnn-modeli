import tensorflow as tf
from tensorflow.keras import layers, models

print("Yapay Zeka (CNN) Beyni Insa Ediliyor...\n")

# 1. Modelin Temelini Atiyoruz (Siralik Katmanlar)
model = models.Sequential()

# 2. GOZ KATMANLARI (Evrisim - Concolutional)
# 32 adet farkli buyutec (filtre) kullaniyoruz.
# input_shape=(150, 150, 1) -> 150x150 boyutunda, 1 renk kanalli (Siyah-Beyaz) resim girecek
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
model.add(layers.MaxPooling2D((2, 2))) # Goruntuyu kucultup onemli yerleri alir

# Ikinci bir goz katmani (Daha derine, daha detayli bakmak icin)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 3. KARAR KATMANLARI (Dense)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model basariyla olusturuldu!\n")
print("Iste Yapay Zekamizin Anatomisi (Model Ozeti):")
print("-" * 50)
model.summary()