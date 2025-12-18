# Project Brief: Veri Madenciliği Eğitim Platformu

## Proje Özeti
Yerel ortamda çalışan, veritabanı gerektirmeyen, Flask tabanlı bir veri madenciliği eğitim platformu.

## Temel Gereksinimler

### Fonksiyonel Gereksinimler
1. **Algoritma Eğitimi:** Kullanıcılar temel ML algoritmalarını interaktif olarak öğrenebilmeli
2. **CSV Yükleme:** Kullanıcı kendi veri setini yükleyebilmeli
3. **Görselleştirme:** Sonuçlar grafik ve tablolarla sunulmalı
4. **Metrik Hesaplama:** Her algoritma için uygun metrikler gösterilmeli

### Teknik Gereksinimler
1. **Vanilla Python:** Tüm algoritmalar sıfırdan yazılmalı (sklearn YOK)
2. **Stateless:** Veritabanı kullanılmayacak
3. **Local:** Sadece yerel ortamda çalışacak

## Hedef Algoritmalar

### Sınıflandırma
- KNN (K-Nearest Neighbors) - Öklid mesafesi
- Decision Tree - ID3, C4.5, CART (Gini, Twoing) parametrik

### Kümeleme
- K-Means - Manuel centroid iterasyonu

### Birliktelik Kuralları
- Apriori - Support, Confidence, Lift hesaplamaları

## Başarı Kriterleri
1. Tüm algoritmalar matematiksel doğrulukla çalışmalı
2. UI kullanıcı dostu ve responsive olmalı
3. Eğitim amaçlı açıklamalar içermeli
4. Performans metrikleri doğru hesaplanmalı

## Kapsam Dışı
- Gerçek zamanlı veri akışı
- Kullanıcı kimlik doğrulama
- Veritabanı entegrasyonu
- Cloud deployment
- Hazır ML kütüphaneleri (sklearn, etc.)

## Zaman Çizelgesi
- Phase 1: Memory Bank + Proje yapısı
- Phase 2: Core algoritmalar
- Phase 3: Flask routes + API
- Phase 4: Frontend UI
- Phase 5: Test ve refinement
