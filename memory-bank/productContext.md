# Product Context: Veri Madenciliği Eğitim Platformu

## Neden Bu Proje Var?

### Problem Tanımı
1. **Kara Kutu Sorunu:** Öğrenciler sklearn gibi kütüphaneleri kullanıyor ama algoritmaların iç işleyişini anlamıyor
2. **Teorik-Pratik Uçurumu:** Matematiksel formüller ile gerçek kod arasında bağlantı kuramıyorlar
3. **Erişim Zorluğu:** Çoğu eğitim platformu online ve karmaşık kurulum gerektiriyor

### Çözümümüz
Algoritmaların matematiksel temellerini saf Python ile implement ederek, öğrencilerin:
- Her adımı kod seviyesinde görmesini
- Kendi verileriyle deney yapmasını
- Metriklerin nasıl hesaplandığını anlamasını sağlıyoruz

## Nasıl Çalışmalı?

### Kullanıcı Akışı
```
1. Ana sayfa → Sidebar'dan algoritma seç
2. Algoritma sayfası → CSV yükle
3. Parametreleri ayarla (k değeri, split oranı, vb.)
4. "Çalıştır" butonuna bas
5. Sonuçları gör: Metrikler, Grafikler, Tablolar
```

### Algoritma Kategorileri
| Kategori | Algoritmalar | Veri Formatı |
|----------|-------------|--------------|
| Sınıflandırma | KNN, Decision Tree | Son sütun = label |
| Kümeleme | K-Means | Tüm sütunlar numerik |
| Birliktelik | Apriori | Transaction-based |

## Kullanıcı Deneyimi Hedefleri

### Öğrenme Odaklı
- Her algoritma sayfasında kısa teori açıklaması
- Formüllerin görsel gösterimi
- Adım adım hesaplama logları (opsiyonel)

### Kullanım Kolaylığı
- Drag & drop CSV yükleme
- Sezgisel parametre kontrolleri
- Responsive tasarım (mobil uyumlu)

### Görsel Zenginlik
- Chart.js ile interaktif grafikler
- Confusion matrix heatmap
- Cluster görselleştirme (scatter plot)
- Decision tree yapısı (opsiyonel)

## Hedef Kitle
1. **Üniversite Öğrencileri:** Veri madenciliği dersi alanlar
2. **Bootcamp Katılımcıları:** ML öğrenenler
3. **Meraklı Geliştiriciler:** Algoritmaları derinlemesine anlamak isteyenler

## Başarı Metrikleri
- Algoritma doğruluğu: sklearn ile karşılaştırılabilir sonuçlar
- Sayfa yüklenme: < 2 saniye
- CSV işleme: < 5 saniye (10K satıra kadar)
