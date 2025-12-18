"""
Metrik Hesaplamaları - Vanilla Python Implementasyonu

Bu modül, makine öğrenmesi modellerinin performansını değerlendirmek için
gerekli metrikleri içerir. Tüm hesaplamalar sıfırdan yapılmıştır.

Sınıflandırma Metrikleri:
- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1-Score
- Confusion Matrix

Kümeleme Metrikleri:
- Inertia (WCSS)
- Silhouette Score
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Set


# =============================================================================
# SINIFLANDIRMA METRİKLERİ
# =============================================================================

def confusion_matrix(y_true: List[Any], y_pred: List[Any]) -> Tuple[List[List[int]], List[Any]]:
    """
    Karmaşıklık matrisi hesapla.
    
    Confusion Matrix:
                    Predicted
                    Neg    Pos
    Actual  Neg     TN     FP
            Pos     FN     TP
    
    - TN (True Negative): Doğru negatif tahmin
    - FP (False Positive): Yanlış pozitif (Type I Error)
    - FN (False Negative): Yanlış negatif (Type II Error)
    - TP (True Positive): Doğru pozitif tahmin
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
    
    Returns:
        matrix: Confusion matrix (n_classes x n_classes)
        labels: Sınıf etiketleri sıralı listesi
    
    Örnek:
    >>> y_true = ['cat', 'cat', 'dog', 'dog', 'dog']
    >>> y_pred = ['cat', 'dog', 'dog', 'dog', 'cat']
    >>> cm, labels = confusion_matrix(y_true, y_pred)
    >>> # labels = ['cat', 'dog']
    >>> # cm = [[1, 1], [1, 2]]
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ve y_pred uzunlukları eşleşmiyor: {len(y_true)} != {len(y_pred)}")
    
    # Benzersiz etiketleri sıralı olarak al
    labels = sorted(list(set(y_true) | set(y_pred)))
    n_labels = len(labels)
    
    # Etiket -> indeks eşlemesi
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Matrisi oluştur
    matrix = [[0] * n_labels for _ in range(n_labels)]
    
    for true, pred in zip(y_true, y_pred):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        matrix[true_idx][pred_idx] += 1
    
    return matrix, labels


def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """
    Doğruluk (Accuracy) hesapla.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
             = Doğru tahmin sayısı / Toplam tahmin sayısı
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
    
    Returns:
        accuracy: 0 ile 1 arasında doğruluk değeri
    
    Not:
    - Dengesiz veri setlerinde yanıltıcı olabilir
    - Örn: %95 negatif veri setinde her zaman "negatif" tahmini %95 accuracy verir
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ve y_pred uzunlukları eşleşmiyor")
    
    if len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def precision_score(y_true: List[Any], y_pred: List[Any], average: str = 'macro') -> float:
    """
    Kesinlik (Precision) hesapla.
    
    Precision = TP / (TP + FP)
              = Doğru pozitif / Tüm pozitif tahminler
    
    "Pozitif dediğimizin ne kadarı gerçekten pozitif?"
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        average: 'macro' (sınıf ortalama), 'micro' (global), 'weighted' (ağırlıklı)
    
    Returns:
        precision: 0 ile 1 arasında kesinlik değeri
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    n_labels = len(labels)
    
    precisions = []
    supports = []
    
    for i in range(n_labels):
        # TP: matrix[i][i]
        tp = cm[i][i]
        # FP: i. sütundaki diğer değerler toplamı
        fp = sum(cm[j][i] for j in range(n_labels)) - tp
        
        # Support: Bu sınıfın gerçek örnek sayısı
        support = sum(cm[i])
        supports.append(support)
        
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)
    
    if average == 'macro':
        # Basit ortalama
        return sum(precisions) / n_labels if n_labels > 0 else 0.0
    elif average == 'micro':
        # Global TP / (Global TP + Global FP)
        total_tp = sum(cm[i][i] for i in range(n_labels))
        total_predicted = sum(sum(cm[j][i] for j in range(n_labels)) for i in range(n_labels))
        return total_tp / total_predicted if total_predicted > 0 else 0.0
    elif average == 'weighted':
        # Support ile ağırlıklı ortalama
        total_support = sum(supports)
        if total_support == 0:
            return 0.0
        return sum(p * s for p, s in zip(precisions, supports)) / total_support
    else:
        raise ValueError(f"Geçersiz average: {average}")


def recall_score(y_true: List[Any], y_pred: List[Any], average: str = 'macro') -> float:
    """
    Duyarlılık (Recall / Sensitivity / True Positive Rate) hesapla.
    
    Recall = TP / (TP + FN)
           = Doğru pozitif / Tüm gerçek pozitifler
    
    "Gerçek pozitiflerin ne kadarını yakaladık?"
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        average: 'macro', 'micro', 'weighted'
    
    Returns:
        recall: 0 ile 1 arasında duyarlılık değeri
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    n_labels = len(labels)
    
    recalls = []
    supports = []
    
    for i in range(n_labels):
        # TP: matrix[i][i]
        tp = cm[i][i]
        # FN: i. satırdaki diğer değerler toplamı
        fn = sum(cm[i]) - tp
        
        support = sum(cm[i])
        supports.append(support)
        
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)
    
    if average == 'macro':
        return sum(recalls) / n_labels if n_labels > 0 else 0.0
    elif average == 'micro':
        total_tp = sum(cm[i][i] for i in range(n_labels))
        total_actual = sum(sum(cm[i]) for i in range(n_labels))
        return total_tp / total_actual if total_actual > 0 else 0.0
    elif average == 'weighted':
        total_support = sum(supports)
        if total_support == 0:
            return 0.0
        return sum(r * s for r, s in zip(recalls, supports)) / total_support
    else:
        raise ValueError(f"Geçersiz average: {average}")


def f1_score(y_true: List[Any], y_pred: List[Any], average: str = 'macro') -> float:
    """
    F1-Score hesapla.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Harmonik ortalama olduğu için:
    - Precision veya Recall düşükse F1 de düşük olur
    - İkisi de yüksek olmalı ki F1 yüksek olsun
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        average: 'macro', 'micro', 'weighted'
    
    Returns:
        f1: 0 ile 1 arasında F1 değeri
    """
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    
    if p + r > 0:
        return 2 * (p * r) / (p + r)
    else:
        return 0.0


def classification_report(y_true: List[Any], y_pred: List[Any]) -> Dict:
    """
    Kapsamlı sınıflandırma raporu oluştur.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
    
    Returns:
        report: Sınıf bazlı ve genel metrikler içeren sözlük
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    n_labels = len(labels)
    
    report = {
        'classes': {},
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_avg': {
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1_score': f1_score(y_true, y_pred, average='macro')
        },
        'weighted_avg': {
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    }
    
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n_labels)) - tp
        fn = sum(cm[i]) - tp
        support = sum(cm[i])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        report['classes'][label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
    
    return report


# =============================================================================
# KÜMELEME METRİKLERİ
# =============================================================================

def inertia(X: List[List[float]], labels: List[int], centers: List[List[float]]) -> float:
    """
    Inertia (Within-Cluster Sum of Squares - WCSS) hesapla.
    
    Inertia = Σ Σ ||x - μk||²
              k x∈Ck
    
    Her noktanın kendi küme merkezine olan mesafelerinin karesi toplamı.
    Düşük inertia = Kümeler daha kompakt.
    
    Args:
        X: Veri noktaları
        labels: Küme etiketleri
        centers: Küme merkezleri
    
    Returns:
        inertia_value: Inertia değeri
    """
    inertia_value = 0.0
    
    for i, x in enumerate(X):
        center = centers[labels[i]]
        # ||x - μk||² hesapla
        squared_dist = sum((xi - ci) ** 2 for xi, ci in zip(x, center))
        inertia_value += squared_dist
    
    return inertia_value


def silhouette_score(X: List[List[float]], labels: List[int]) -> float:
    """
    Ortalama Silhouette Score hesapla.
    
    Her nokta için:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    Burada:
    - a(i): i noktasının kendi kümesindeki diğer noktalara ortalama mesafesi (cohesion)
    - b(i): i noktasının en yakın diğer kümedeki noktalara ortalama mesafesi (separation)
    
    Silhouette değeri:
    - s = 1: İdeal kümeleme (nokta doğru kümede)
    - s = 0: Küme sınırında
    - s = -1: Yanlış kümede
    
    Args:
        X: Veri noktaları
        labels: Küme etiketleri
    
    Returns:
        silhouette: Ortalama silhouette score (-1 ile 1 arasında)
    """
    n_samples = len(X)
    if n_samples < 2:
        return 0.0
    
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return 0.0
    
    silhouette_values = []
    
    for i in range(n_samples):
        # a(i): Kendi kümesindeki diğer noktalara ortalama mesafe
        a_i = _calculate_a(X, labels, i)
        
        # b(i): En yakın diğer kümedeki noktalara ortalama mesafe
        b_i = _calculate_b(X, labels, i, unique_labels)
        
        # s(i) hesapla
        max_ab = max(a_i, b_i)
        if max_ab > 0:
            s_i = (b_i - a_i) / max_ab
        else:
            s_i = 0.0
        
        silhouette_values.append(s_i)
    
    return sum(silhouette_values) / n_samples


def _calculate_a(X: List[List[float]], labels: List[int], i: int) -> float:
    """
    a(i): Nokta i'nin kendi kümesindeki diğer noktalara ortalama mesafesi.
    """
    cluster_i = labels[i]
    same_cluster_points = [j for j in range(len(X)) if labels[j] == cluster_i and j != i]
    
    if len(same_cluster_points) == 0:
        return 0.0
    
    total_dist = sum(_euclidean_distance(X[i], X[j]) for j in same_cluster_points)
    return total_dist / len(same_cluster_points)


def _calculate_b(X: List[List[float]], labels: List[int], i: int, unique_labels: List[int]) -> float:
    """
    b(i): Nokta i'nin en yakın diğer kümedeki noktalara ortalama mesafesi.
    """
    cluster_i = labels[i]
    min_avg_dist = float('inf')
    
    for cluster in unique_labels:
        if cluster == cluster_i:
            continue
        
        other_cluster_points = [j for j in range(len(X)) if labels[j] == cluster]
        
        if len(other_cluster_points) == 0:
            continue
        
        avg_dist = sum(_euclidean_distance(X[i], X[j]) for j in other_cluster_points) / len(other_cluster_points)
        min_avg_dist = min(min_avg_dist, avg_dist)
    
    return min_avg_dist if min_avg_dist != float('inf') else 0.0


def _euclidean_distance(p: List[float], q: List[float]) -> float:
    """Öklid mesafesi hesapla."""
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))


def silhouette_samples(X: List[List[float]], labels: List[int]) -> List[float]:
    """
    Her örnek için silhouette değerini hesapla.
    
    Args:
        X: Veri noktaları
        labels: Küme etiketleri
    
    Returns:
        silhouette_values: Her örnek için silhouette değeri
    """
    n_samples = len(X)
    unique_labels = list(set(labels))
    
    silhouette_values = []
    
    for i in range(n_samples):
        a_i = _calculate_a(X, labels, i)
        b_i = _calculate_b(X, labels, i, unique_labels)
        
        max_ab = max(a_i, b_i)
        if max_ab > 0:
            s_i = (b_i - a_i) / max_ab
        else:
            s_i = 0.0
        
        silhouette_values.append(s_i)
    
    return silhouette_values


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def train_test_split(
    X: List[List[float]],
    y: List[Any],
    test_size: float = 0.2,
    random_state: int = None,
    shuffle: bool = True
) -> Tuple[List[List[float]], List[List[float]], List[Any], List[Any]]:
    """
    Veriyi eğitim ve test setlerine ayır.
    
    Args:
        X: Özellikler
        y: Etiketler
        test_size: Test seti oranı (0 ile 1 arasında)
        random_state: Rastgelelik için seed
        shuffle: Veriyi karıştır mı?
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    import random as rnd
    
    if len(X) != len(y):
        raise ValueError("X ve y uzunlukları eşleşmiyor")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size 0 ile 1 arasında olmalı")
    
    n_samples = len(X)
    indices = list(range(n_samples))
    
    if random_state is not None:
        rnd.seed(random_state)
    
    if shuffle:
        rnd.shuffle(indices)
    
    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test
