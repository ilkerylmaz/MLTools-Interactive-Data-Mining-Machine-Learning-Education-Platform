"""
Veri Madenciliği Eğitim Platformu - Algoritmalar Paketi

Bu paket, sıfırdan yazılmış (Vanilla Python) makine öğrenmesi algoritmalarını içerir.
Hiçbir hazır ML kütüphanesi (sklearn vb.) kullanılmamıştır.

Algoritmalar:
- KNN: K-Nearest Neighbors (Sınıflandırma)
- DecisionTree: Karar Ağacı (ID3, C4.5, CART)
- KMeans: K-Means Kümeleme
- Apriori: Birliktelik Kuralları

Yardımcı Modüller:
- metrics: Performans metrikleri hesaplama
"""

from .knn import KNN
from .decision_tree import DecisionTree
from .kmeans import KMeans
from .apriori import Apriori
from .metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
    inertia
)

__all__ = [
    'KNN',
    'DecisionTree',
    'KMeans',
    'Apriori',
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'silhouette_score',
    'inertia'
]

__version__ = '1.0.0'
