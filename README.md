# 🩻 Akciğer Röntgenlerinden Zatürre (Pneumonia) Teşhisi Yapan CNN Modeli

Bu proje, Derin Öğrenme (Deep Learning) ve Evrişimli Sinir Ağları (CNN) kullanılarak, hastane röntgen görüntülerinden (X-Ray) zatürre hastalığının otomatik olarak teşhis edilmesini sağlayan bir yapay zeka sistemidir.

## 🚀 Projenin Amacı
Görüntü işleme (Computer Vision) teknikleri kullanılarak doktorların teşhis süreçlerini hızlandırmak ve karar destek mekanizması oluşturmak hedeflenmiştir. Model, sağlıklı akciğerler ile zatürreli akciğerler arasındaki yapısal farkları piksel matrisleri üzerinden öğrenir.

## 🧠 Model Mimarisi ve Teknolojiler
Projede **TensorFlow** ve **Keras** altyapısı kullanılarak özel bir CNN mimarisi inşa edilmiştir.

* **Kullanılan Kütüphaneler:** `TensorFlow`, `Keras`, `OpenCV`, `NumPy`, `Matplotlib`
* **Model Tipi:** Sequential (Sıralı) CNN
* **Parametre Sayısı:** Toplam 18,816 eğitilebilir nöron bağlantısı
* **Görüntü Ön İşleme:** Tüm röntgenler 150x150 piksel boyutuna küçültülmüş ve (0-1) aralığında normalize edilmiştir.

## 📊 Veri Seti ve Başarı Oranı
Modelin eğitiminde **Guangzhou Kadın ve Çocuk Sağlığı Merkezi**'nden alınan gerçek hasta verileri (Kaggle Chest X-Ray Pneumonia Dataset) kullanılmıştır.

* **Veri Seti Büyüklüğü:** 5.216 adet eğitim (train) röntgen görüntüsü.
* **Model Başarı Oranı (Accuracy):** **%97.51** 🏆

1.2 GB boyutundaki bu devasa veri seti, donanım hızlandırması (GPU) kullanılarak kısa sürede işlenmiş ve modelin eğitim turları (Epochs) başarıyla tamamlanmıştır.

## ⚙️ Nasıl Çalıştırılır?
*Projedeki ana veri seti boyutu nedeniyle GitHub'a yüklenmemiştir (`.gitignore` ile hariç tutulmuştur).*

1. Projeyi bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/rontgen-cnn-modeli.git](https://github.com/KULLANICI_ADIN/rontgen-cnn-modeli.git)

2. Gerekli kütüphaneleri yükleyin:
pip install tensorflow opencv-python matplotlib numpy

3. Kaggle üzerinden veri setini indirip chest_xray klasörünü proje dizinine ekleyin ve modeli çalıştırın
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

4. İndirdiğiniz arşivin içindeki chest_xray klasörünü proje dizininin ana dizinine yerleştirin.

5. başlatmak için 05_gercek_veri_cnn.py dosyasını çalıştırın.
-----------------------------------------------------

# ---

# 🩻 CNN Model for Pneumonia Diagnosis from Chest X-Rays

This project is an artificial intelligence system that automatically diagnoses pneumonia from hospital X-ray images using Deep Learning and Convolutional Neural Networks (CNN).

## 🚀 Project Goal
It aims to accelerate the diagnostic processes of doctors and create a decision support mechanism using Computer Vision techniques. The model learns the structural differences between healthy and pneumonic lungs through pixel matrices.

## 🧠 Model Architecture & Technologies
A custom CNN architecture was built using **TensorFlow** and **Keras** infrastructure.

* **Libraries Used:** `TensorFlow`, `Keras`, `OpenCV`, `NumPy`, `Matplotlib`
* **Model Type:** Sequential CNN
* **Parameters:** Total 18,816 trainable neuron connections
* **Image Preprocessing:** All X-rays are resized to 150x150 pixels and normalized to the (0-1) range.

## 📊 Dataset & Accuracy
Real patient data from the **Guangzhou Women and Children's Medical Center** was used to train the model. The dataset is open-source and available via Kaggle.

* **Dataset Source:** [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Dataset Size:** Total 1.2 GB (5,216 training X-ray images).
* **Model Accuracy:** **97.51%** 🏆

This massive 1.2 GB dataset was processed rapidly using hardware acceleration (GPU), and the model's training epochs were successfully completed.

## ⚙️ How to Run
*Due to the large size of the main dataset (1.2 GB), it is not uploaded to GitHub (excluded via `.gitignore`).*

1. Clone the project to your local machine:
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/rontgen-cnn-modeli.git](https://github.com/KULLANICI_ADIN/rontgen-cnn-modeli.git)

2. Install the required libraries:
pip install tensorflow opencv-python matplotlib numpy

3. Download the dataset from kaggle from this link https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

4. Extract the archive and place the chest_xray folder directly into the project's root directory.

5. Run the 05_gercek_veri_cnn.py file to start the training process.
