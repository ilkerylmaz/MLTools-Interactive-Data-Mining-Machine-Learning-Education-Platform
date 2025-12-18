"""
K-Nearest Neighbors (KNN) Algoritması - Vanilla Python Implementasyonu

Matematiksel Temel:
==================
KNN, bir noktanın sınıfını belirlemek için en yakın k komşusuna bakar.
Tahmin, bu komşuların çoğunluk oylaması ile yapılır.

Öklid Mesafesi:
d(p, q) = sqrt(Σ(pi - qi)²)

Burada:
- p ve q iki nokta
- pi ve qi sırasıyla p ve q'nun i. boyuttaki değerleri

Algoritma Adımları:
1. Test noktası ile tüm eğitim noktaları arasındaki mesafeyi hesapla
2. En küçük k mesafeye sahip noktaları seç
3. Bu k noktanın sınıflarının çoğunluğunu tahmin olarak döndür
"""

import math
from collections import Counter
from typing import List, Tuple, Any


class KNN:
    """
    K-Nearest Neighbors Sınıflandırıcısı
    
    Parametreler:
    ------------
    k : int, default=3
        Komşu sayısı. Tek sayı olması önerilir (beraberlik durumunu önlemek için).
    
    Öznitelikler:
    ------------
    X_train : List[List[float]]
        Eğitim özellikleri
    y_train : List[Any]
        Eğitim etiketleri
    
    Örnek Kullanım:
    --------------
    >>> knn = KNN(k=3)
    >>> knn.fit(X_train, y_train)
    >>> predictions = knn.predict(X_test)
    """
    
    def __init__(self, k: int = 3):
        """
        KNN sınıflandırıcısını başlat.
        
        Args:
            k: Komşu sayısı (varsayılan: 3)
        """
        if k < 1:
            raise ValueError("k değeri 1'den küçük olamaz")
        
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: List[List[float]], y: List[Any]) -> 'KNN':
        """
        Modeli eğitim verileriyle eğit.
        
        KNN'de gerçek bir "eğitim" yoktur. Sadece verileri hafızada tutarız.
        Bu yüzden KNN "lazy learner" (tembel öğrenici) olarak adlandırılır.
        
        Args:
            X: Eğitim özellikleri (n_samples, n_features)
            y: Eğitim etiketleri (n_samples,)
        
        Returns:
            self: Eğitilmiş model
        
        Raises:
            ValueError: X ve y uzunlukları eşleşmiyorsa
        """
        if len(X) != len(y):
            raise ValueError(f"X ve y uzunlukları eşleşmiyor: {len(X)} != {len(y)}")
        
        if len(X) < self.k:
            raise ValueError(f"Eğitim verisi sayısı ({len(X)}) k değerinden ({self.k}) küçük olamaz")
        
        # Verileri sakla
        self.X_train = X
        self.y_train = y
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[Any]:
        """
        Test verileri için tahmin yap.
        
        Her test noktası için:
        1. Tüm eğitim noktalarına olan mesafeyi hesapla
        2. En yakın k komşuyu bul
        3. Çoğunluk oylaması ile sınıf belirle
        
        Args:
            X: Test özellikleri (n_samples, n_features)
        
        Returns:
            predictions: Tahmin edilen sınıflar
        
        Raises:
            ValueError: Model henüz eğitilmemişse
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model henüz eğitilmedi. Önce fit() metodunu çağırın.")
        
        predictions = []
        for x in X:
            # Her test noktası için tahmin yap
            prediction = self._predict_single(x)
            predictions.append(prediction)
        
        return predictions
    
    def _predict_single(self, x: List[float]) -> Any:
        """
        Tek bir test noktası için tahmin yap.
        
        Args:
            x: Test noktası özellikleri
        
        Returns:
            prediction: Tahmin edilen sınıf
        """
        # 1. Tüm eğitim noktalarına olan mesafeleri hesapla
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[i]))
        
        # 2. Mesafeye göre sırala ve en yakın k komşuyu al
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # 3. Çoğunluk oylaması
        k_nearest_labels = [label for _, label in k_nearest]
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]
    
    def _euclidean_distance(self, p: List[float], q: List[float]) -> float:
        """
        İki nokta arasındaki Öklid mesafesini hesapla.
        
        Formül: d(p, q) = sqrt(Σ(pi - qi)²)
        
        Args:
            p: Birinci nokta
            q: İkinci nokta
        
        Returns:
            distance: Öklid mesafesi
        
        Raises:
            ValueError: Noktaların boyutları eşleşmiyorsa
        """
        if len(p) != len(q):
            raise ValueError(f"Nokta boyutları eşleşmiyor: {len(p)} != {len(q)}")
        
        # Σ(pi - qi)² hesapla
        squared_sum = 0
        for pi, qi in zip(p, q):
            squared_sum += (pi - qi) ** 2
        
        # Karekök al
        return math.sqrt(squared_sum)
    
    def get_params(self) -> dict:
        """
        Model parametrelerini döndür.
        
        Returns:
            params: Parametre sözlüğü
        """
        return {'k': self.k}
    
    def set_params(self, **params) -> 'KNN':
        """
        Model parametrelerini ayarla.
        
        Args:
            **params: Parametre anahtar-değer çiftleri
        
        Returns:
            self: Güncellenmiş model
        """
        if 'k' in params:
            self.k = params['k']
        return self
    
    def predict_proba(self, X: List[List[float]]) -> List[dict]:
        """
        Her sınıf için olasılık tahminleri döndür.
        
        Olasılık, k komşu içindeki sınıf oranı olarak hesaplanır.
        
        Args:
            X: Test özellikleri
        
        Returns:
            probabilities: Her örnek için sınıf olasılıkları sözlüğü
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        probabilities = []
        for x in X:
            # En yakın k komşuyu bul
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._euclidean_distance(x, x_train)
                distances.append((dist, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [label for _, label in distances[:self.k]]
            
            # Sınıf oranlarını hesapla
            label_counts = Counter(k_nearest_labels)
            proba = {label: count / self.k for label, count in label_counts.items()}
            probabilities.append(proba)
        
        return probabilities
    
    def get_neighbors(self, x: List[float]) -> List[Tuple[float, Any, int]]:
        """
        Bir nokta için en yakın k komşuyu ve bilgilerini döndür.
        
        Debug ve görselleştirme için kullanışlı.
        
        Args:
            x: Sorgu noktası
        
        Returns:
            neighbors: (mesafe, etiket, indeks) tuple'ları listesi
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._euclidean_distance(x, x_train)
            distances.append((dist, self.y_train[i], i))
        
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
