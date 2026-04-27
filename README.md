# Yapay Sinir Ağları Projesi
**Multi-Layer Perceptron ile Karakter Tanıma ve Perceptron ile Lojik Kapı Sınıflandırma**

## Proje Özeti
Bu projede, Python programlama dili kullanılarak sıfırdan (from scratch) yapay sinir ağı algoritmaları implemente edilmiştir. Scikit-learn, TensorFlow veya PyTorch gibi hazır kütüphaneler kullanılmadan, yalnızca NumPy ile matris operasyonları gerçekleştirilmiştir.

## Özellikler

### 1. Algoritmalar
- **Single Layer Perceptron**: Delta kuralı ile ağırlık güncelleme
- **Multi-Layer Perceptron (MLP)**: Backpropagation algoritması ile eğitim
- **Aktivasyon Fonksiyonları**: Sigmoid, Tanh, ReLU, Step
- **Optimizasyon**: Gradient Descent

### 2. Veri Setleri
- **Karakter Tanıma**: 5 farklı harf (A, B, C, D, E) × 3 farklı font (Standart, İtalik, Kalın) = 15 örnek
- **Lojik Kapılar**: AND, OR, XOR doğruluk tabloları

### 3. Grafiksel Arayüz (GUI)
- **CustomTkinter** ile modern, karanlık tema arayüz
- **Karakter Çizim Alanı**: 7×5 piksel grid üzerinde çizim
- **Parametre Kontrolü**: Learning rate, epoch, gizli katman boyutu ayarlanabilir
- **Model Yönetimi**: Eğitilmiş modelleri kaydetme/yükleme (.npz formatı)
- **Görselleştirme**: Matplotlib ile eğitim grafikleri ve karar sınırları

## Kurulum

### Gereksinimler
pip install numpy matplotlib customtkinter pillow

### Çalıştırma
python main.py

### Proje Yapısı
ann_project/
├── main.py                 # Ana giriş noktası
├── README.md              # Bu dosya
├── requirements.txt       # Bağımlılıklar
├── src/                   # Kaynak kodlar
│   ├── utils.py          # Aktivasyon ve kayıp fonksiyonları
│   ├── perceptron.py     # Tek katmanlı algılayıcı
│   ├── mlp.py            # Çok katmanlı algılayıcı + Backpropagation
│   ├── data_loader.py    # Veri seti oluşturma ve yükleme
│   └── visualizer.py     # Matplotlib görselleştirmeleri
├── gui/                   # Arayüz modülü
│   ├── __init__.py
│   └── main_gui.py       # CustomTkinter ana pencere
└── tests/                 # Test modülleri
    ├── test_networks.py  # Ağ testleri
    └── test_visualization.py # Görselleştirme testleri

## Kullanım

### Karakter Tanıma
1- "Karakter Tanıma" sekmesine geçin
2- Learning Rate (örn: 0.3), Epoch (örn: 2000) ve Gizli Katman Nöron Sayısını (örn: 25) girin
3- "MLP Modelini Eğit" butonuna basın
4- Sağ taraftaki 7×5 grid üzerine bir harf çizin (piksel aç/kapat için tıklayın)
5- "Tahmin Et" butonuna basın

### Lojik Kapılar
1- "Lojik Kapılar" sekmesine geçin
2- AND, OR veya XOR seçin
3- "Perceptron Eğit" butonuna basın
4- "Karar Sınırını Göster" ile 2D görselleştirmeyi görün

## Matematiksel Altyapı

### Perceptron Güncelleme Kuralı
w_i = w_i + η * (target - output) * x_i

### Backpropagation (MLP)
1- İleri Besleme: z = W·a + b, a = σ(z)
2- Hata Hesaplama: δ = (a - y) ⊙ σ'(z)
3- Geri Yayılım: δ_l = (W_{l+1}^T · δ_{l+1}) ⊙ σ'(z_l)
4- Ağırlık Güncelleme: W = W - η * ∇W

## Sonuçlar

### Doğruluk Oranları
AND Kapısı: %100 (Doğrusal ayrılabilir)
OR Kapısı: %100 (Doğrusal ayrılabilir)
XOR Kapısı: ~%50 (Tek katmanlı Perceptron ile çözülemez, MLP ile %100)
Karakter Tanıma: %100 (Eğitim seti üzerinde)

## Geliştirici
Öğrenci: Berk Yağız KALAFAT
Numara: 220501003
Teslim Tarihi: 26 Nisan 2026

## Lisans

Bu proje eğitim amaçlı geliştirilmiştir.