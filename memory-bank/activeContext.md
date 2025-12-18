# Active Context: Güncel Çalışma Durumu

## Şu Anki Odak
**Phase 3: Tamamlandı** - Proje kullanıma hazır

### Aktif Görevler
- [x] Memory Bank dosyaları oluşturuldu
- [x] Proje klasör yapısı oluşturuldu
- [x] Core algoritma class'ları yazıldı
- [x] Flask app kuruldu
- [x] Frontend templates hazırlandı

## Son Değişiklikler
- **11 Aralık 2025:** Proje başlatıldı
- Memory Bank sistemi kuruldu
- Teknik kararlar alındı
- Tüm algoritmalar vanilla Python ile implementa edildi
- Flask app ve UI tamamlandı

## Proje Durumu: TAMAMLANDI ✅

### Mevcut Dosya Yapısı
```
mltools/
├── AGENTS.md
├── README.md
├── requirements.txt
├── app.py                    # Flask application
├── algorithms/
│   ├── __init__.py
│   ├── knn.py               # K-Nearest Neighbors
│   ├── decision_tree.py     # ID3/C4.5/CART
│   ├── kmeans.py            # K-Means Clustering
│   ├── apriori.py           # Association Rules
│   └── metrics.py           # Tüm metrikler
├── templates/
│   ├── base.html            # Base template
│   ├── index.html           # Ana sayfa
│   ├── knn.html             # KNN UI
│   ├── decision_tree.html   # Decision Tree UI
│   ├── kmeans.html          # K-Means UI
│   └── apriori.html         # Apriori UI
├── memory-bank/
│   └── [tüm context dosyaları]
└── uploads/                  # CSV yükleme klasörü
```

## Çalıştırma
```bash
pip install -r requirements.txt
python app.py
# http://localhost:5000 adresinden erişin
```

## Aktif Kararlar

### Onaylanan Kararlar
| Karar | Seçim | Tarih |
|-------|-------|-------|
| Memory Bank konumu | `memory-bank/` klasörü | 11 Ara 2025 |
| Decision Tree yapısı | Tek class, parametrik | 11 Ara 2025 |
| CSV formatı | Son sütun label (sınıflandırma) | 11 Ara 2025 |
| Apriori formatı | Transaction-based | 11 Ara 2025 |
| Test/Train split | UI'dan seçilebilir (%50-%90) | 11 Ara 2025 |
| Twoing kriteri | Evet, dahil edildi | 11 Ara 2025 |

## Önemli Kalıplar ve Tercihler

### Kod Stili
- Class-based algoritmalar
- Detaylı docstring ve yorumlar (Türkçe)
- Type hints kullanımı
- Matematiksel formüller yorumlarda

### Dosya Organizasyonu
```
algorithms/
├── __init__.py
├── knn.py
├── decision_tree.py
├── kmeans.py
├── apriori.py
└── metrics.py
```

## Öğrenilen Dersler
- Henüz yok (proje yeni başladı)

## Riskler ve Dikkat Edilecekler
1. **Performans:** Büyük veri setlerinde vanilla Python yavaş olabilir
2. **Edge Cases:** Eksik veri, kategorik değişkenler
3. **Numerik Stabilite:** Division by zero, overflow
