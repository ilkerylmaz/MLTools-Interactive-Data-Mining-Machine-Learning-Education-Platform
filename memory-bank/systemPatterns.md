# System Patterns: Mimari ve Tasarım Kararları

## Sistem Mimarisi

### Genel Yapı: Flask Monolitik
```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Jinja2    │  │  Bootstrap  │  │  Chart.js   │     │
│  │  Templates  │  │     CSS     │  │   Graphs    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    FLASK APP                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   Routes                          │   │
│  │  /              → index                          │   │
│  │  /knn           → KNN page                       │   │
│  │  /decision-tree → Decision Tree page             │   │
│  │  /kmeans        → K-Means page                   │   │
│  │  /apriori       → Apriori page                   │   │
│  │  /api/run/<alg> → Algorithm execution            │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  ALGORITHMS LAYER                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │   KNN    │ │ DecTree  │ │ K-Means  │ │ Apriori  │  │
│  │ (vanilla)│ │(vanilla) │ │(vanilla) │ │(vanilla) │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                      │                                  │
│              ┌───────┴───────┐                         │
│              │    Metrics    │                         │
│              │   (vanilla)   │                         │
│              └───────────────┘                         │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   DATA LAYER                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Pandas / NumPy                       │   │
│  │         (Sadece veri okuma/işleme)               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Dosya Yapısı Standardı

```
mltools/
├── memory-bank/                 # Memory Bank (dokümantasyon)
│   ├── projectbrief.md
│   ├── productContext.md
│   ├── activeContext.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   └── progress.md
│
├── algorithms/                  # Vanilla Python algoritmaları
│   ├── __init__.py
│   ├── base.py                 # Base class (opsiyonel)
│   ├── knn.py                  # K-Nearest Neighbors
│   ├── decision_tree.py        # Decision Tree (ID3/C4.5/CART)
│   ├── kmeans.py               # K-Means Clustering
│   ├── apriori.py              # Apriori Algorithm
│   └── metrics.py              # Metrik hesaplamaları
│
├── static/                      # Statik dosyalar
│   ├── css/
│   │   └── style.css           # Custom styles
│   └── js/
│       └── main.js             # Custom JavaScript
│
├── templates/                   # Jinja2 templates
│   ├── base.html               # Ana layout (sidebar dahil)
│   ├── index.html              # Ana sayfa
│   ├── knn.html                # KNN sayfası
│   ├── decision_tree.html      # Decision Tree sayfası
│   ├── kmeans.html             # K-Means sayfası
│   ├── apriori.html            # Apriori sayfası
│   └── components/
│       ├── upload_modal.html   # CSV yükleme modalı
│       └── results.html        # Sonuç gösterimi
│
├── uploads/                     # Geçici CSV dosyaları
│   └── .gitkeep
│
├── app.py                       # Flask ana uygulama
├── requirements.txt             # Python bağımlılıkları
├── AGENTS.md                    # Memory Bank kuralları
└── README.md                    # Proje dokümantasyonu
```

## Tasarım Kalıpları

### 1. Algorithm Base Pattern
Tüm algoritmalar ortak bir interface'e sahip:
```python
class BaseAlgorithm:
    def fit(self, X, y=None):
        """Modeli eğit"""
        raise NotImplementedError
    
    def predict(self, X):
        """Tahmin yap"""
        raise NotImplementedError
    
    def get_params(self):
        """Parametreleri döndür"""
        raise NotImplementedError
```

### 2. Strategy Pattern (Decision Tree)
Criterion seçimi için strategy pattern:
```python
class DecisionTree:
    def __init__(self, criterion='entropy', algorithm='id3'):
        self.criterion = criterion  # 'entropy', 'gini', 'twoing'
        self.algorithm = algorithm  # 'id3', 'c45', 'cart'
        
    def _calculate_impurity(self, y):
        if self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'twoing':
            return self._twoing(y)
```

### 3. Factory Pattern (Route Handler)
Algoritma oluşturma için factory:
```python
def create_algorithm(algo_type, **params):
    algorithms = {
        'knn': KNN,
        'decision_tree': DecisionTree,
        'kmeans': KMeans,
        'apriori': Apriori
    }
    return algorithms[algo_type](**params)
```

## Kritik Uygulama Yolları

### CSV İşleme Akışı
```
1. Upload → Flask route alır
2. Pandas ile oku → DataFrame
3. Validasyon → Tip kontrolü, eksik veri
4. Split (gerekirse) → Train/Test
5. Algoritmaya gönder → fit/predict
6. Metrik hesapla → metrics.py
7. JSON olarak döndür → Frontend'e
```

### Hata Yönetimi
```python
try:
    result = algorithm.fit(X_train, y_train)
except ValueError as e:
    return jsonify({'error': str(e), 'type': 'validation'})
except Exception as e:
    return jsonify({'error': 'Beklenmeyen hata', 'type': 'system'})
```

## Bileşen İlişkileri

```
┌─────────────┐     uses      ┌─────────────┐
│   app.py    │──────────────▶│ algorithms/ │
└─────────────┘               └─────────────┘
       │                             │
       │ renders                     │ uses
       ▼                             ▼
┌─────────────┐               ┌─────────────┐
│ templates/  │               │  metrics.py │
└─────────────┘               └─────────────┘
       │
       │ includes
       ▼
┌─────────────┐
│   static/   │
└─────────────┘
```

## Vanilla Python Kısıtlamaları

### İZİN VERİLEN
- `pandas` - Veri okuma/yazma
- `numpy` - Array işlemleri, matematik
- `math` - Matematiksel fonksiyonlar
- `collections` - Counter, defaultdict
- `itertools` - Kombinasyonlar (Apriori için)

### YASAK
- `sklearn` - Hiçbir modülü
- `scipy.stats` - İstatistiksel fonksiyonlar
- `statsmodels` - Hazır modeller
- Herhangi bir ML kütüphanesi
