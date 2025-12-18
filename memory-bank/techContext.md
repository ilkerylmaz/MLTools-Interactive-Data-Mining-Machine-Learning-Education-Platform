# Tech Context: Teknoloji Stack ve Kısıtlamalar

## Kullanılan Teknolojiler

### Backend
| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| Python | 3.9+ | Ana programlama dili |
| Flask | 2.x | Web framework |
| Pandas | 2.x | CSV okuma, DataFrame işlemleri |
| NumPy | 1.x | Matematiksel hesaplamalar |

### Frontend
| Teknoloji | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| Jinja2 | (Flask ile) | Template engine |
| Bootstrap | 5.x | CSS framework, responsive tasarım |
| Chart.js | 4.x | Grafik görselleştirme |
| JavaScript | ES6+ | Interaktivite |

### Geliştirme Araçları
| Araç | Kullanım Amacı |
|------|----------------|
| pip | Paket yönetimi |
| venv | Sanal ortam |
| Git | Versiyon kontrolü |

## Geliştirme Ortamı Kurulumu

### 1. Sanal Ortam
```bash
cd mltools
python -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 2. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 3. Uygulamayı Çalıştır
```bash
python app.py
# veya
flask run --debug
```

### 4. Tarayıcıda Aç
```
http://localhost:5000
```

## Teknik Kısıtlamalar

### KRİTİK: Kütüphane Kuralları
```
✅ İZİN VERİLEN (Veri İşleme)
├── pandas          # CSV okuma, DataFrame
├── numpy           # Array, matematik
├── math            # sqrt, log, exp
├── collections     # Counter, defaultdict
└── itertools       # combinations, permutations

❌ YASAK (ML/İstatistik)
├── sklearn         # Tüm modüller
├── scipy.stats     # İstatistiksel fonksiyonlar
├── statsmodels     # Modeller
├── tensorflow      # Deep learning
├── pytorch         # Deep learning
└── xgboost/lightgbm # Gradient boosting
```

### Veritabanı
- **Kısıtlama:** Veritabanı YOK
- **Alternatif:** Stateless mimari, her istek bağımsız
- **Geçici Depolama:** `uploads/` klasörü (session bazlı)

### Performans Hedefleri
| Metrik | Hedef |
|--------|-------|
| Sayfa yüklenme | < 2 saniye |
| CSV işleme (10K satır) | < 5 saniye |
| Algoritma çalışma (10K satır) | < 30 saniye |

## Bağımlılıklar (requirements.txt)

```
flask>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
werkzeug>=2.0.0
```

## Tool Kullanım Kalıpları

### Flask Route Pattern
```python
@app.route('/api/knn', methods=['POST'])
def run_knn():
    # 1. Dosya al
    file = request.files['file']
    
    # 2. Parametreleri al
    k = int(request.form.get('k', 3))
    
    # 3. Veriyi işle
    df = pd.read_csv(file)
    
    # 4. Algoritmayı çalıştır
    model = KNN(k=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # 5. Metrikleri hesapla
    metrics = calculate_metrics(y_test, predictions)
    
    # 6. JSON döndür
    return jsonify(metrics)
```

### Frontend AJAX Pattern
```javascript
async function runAlgorithm(formData) {
    const response = await fetch('/api/knn', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    displayResults(result);
}
```

## Ortam Değişkenleri

```env
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=dev-secret-key-change-in-production
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216  # 16MB
```

## Dosya Upload Kısıtlamaları

| Parametre | Değer |
|-----------|-------|
| Max dosya boyutu | 16 MB |
| İzin verilen uzantılar | .csv |
| Geçici depolama süresi | Session süresi |

## Tarayıcı Desteği
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## API Response Formatı

### Başarılı Response
```json
{
    "success": true,
    "algorithm": "knn",
    "metrics": {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.935
    },
    "confusion_matrix": [[50, 2], [3, 45]],
    "chart_data": {...}
}
```

### Hata Response
```json
{
    "success": false,
    "error": "Geçersiz dosya formatı",
    "type": "validation"
}
```
