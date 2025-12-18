# 🎓 MLTools - Veri Madenciliği Eğitim Platformu

Veri madenciliği ve makine öğrenmesi algoritmalarını **sıfırdan öğrenmek** için tasarlanmış interaktif web platformu. Tüm algoritmalar **vanilla Python** ile yazılmıştır (sklearn kullanılmamıştır).

## ✨ Özellikler

### 🌳 Karar Ağaçları (Decision Trees)
- **ID3** (Information Gain - Entropy)
- **C4.5** (Gain Ratio)
- **CART** (Gini Index & Twoing)
- ✅ İnteraktif D3.js görselleştirme
- ✅ Adım adım entropy hesaplamaları
- ✅ Information Gain gösterimi

### 🎯 Sınıflandırma Algoritmaları
- **KNN** (K-Nearest Neighbors)
  - Euclidean, Manhattan, Minkowski mesafe metrikleri
  - Interaktif k değeri seçimi

### 🔗 Kümeleme Algoritmaları
- **K-Means** Clustering
  - Silhouette score
  - Inertia ölçümü
  - Görsel küme analizi

### 🛒 Birliktelik Kuralları
- **Apriori** Algorithm
  - Market sepeti analizi
  - Support, Confidence, Lift metrikleri

## 🚀 Kurulum

```bash
# Repository'yi klonla
git clone https://github.com/[username]/mltools.git
cd mltools

# Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı çalıştır
python app.py