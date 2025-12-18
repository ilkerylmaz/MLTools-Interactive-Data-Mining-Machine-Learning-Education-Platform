# Veri Madenciliği Eğitim Platformu

Yerel ortamda çalışan, Flask tabanlı bir veri madenciliği eğitim platformu.

## Özellikler

### Algoritmalar
- **Sınıflandırma:** KNN, Decision Tree (ID3, C4.5, CART)
- **Kümeleme:** K-Means
- **Birliktelik Kuralları:** Apriori

### Teknik Özellikler
- Tüm algoritmalar Vanilla Python ile yazılmıştır (sklearn YOK)
- Stateless mimari (veritabanı yok)
- Modern ve responsive UI (Bootstrap 5)
- İnteraktif grafikler (Chart.js)

## Kurulum

```bash
# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı başlat
python app.py
```

## Kullanım

1. Tarayıcıda `http://localhost:5000` adresine gidin
2. Sol sidebar'dan algoritma seçin
3. CSV dosyanızı yükleyin
4. Parametreleri ayarlayın
5. "Çalıştır" butonuna tıklayın

## CSV Format Gereksinimleri

### Sınıflandırma (KNN, Decision Tree)
- Son sütun hedef değişken (label) olmalı
- Örnek:
```csv
feature1,feature2,feature3,label
5.1,3.5,1.4,setosa
4.9,3.0,1.4,setosa
```

### Kümeleme (K-Means)
- Tüm sütunlar numerik olmalı
- Örnek:
```csv
x,y
1.0,2.0
1.5,1.8
```

### Birliktelik (Apriori)
- Her satır bir transaction
- Örnek:
```csv
items
bread,milk,eggs
milk,diapers,beer
```

## Proje Yapısı

```
mltools/
├── algorithms/          # Vanilla Python algoritmaları
├── templates/           # Jinja2 templates
├── static/             # CSS, JS dosyaları
├── memory-bank/        # Proje dokümantasyonu
├── uploads/            # Geçici CSV dosyaları
├── app.py              # Flask uygulaması
└── requirements.txt    # Python bağımlılıkları
```

## Lisans
MIT
