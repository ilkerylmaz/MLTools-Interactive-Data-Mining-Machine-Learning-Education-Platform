"""
K-Means Kümeleme Algoritması - Vanilla Python Implementasyonu

Matematiksel Temel:
==================

K-Means, verileri k adet kümeye ayıran bir kümeleme algoritmasıdır.
Amaç, küme içi varyansı (inertia) minimize etmektir.

Amaç Fonksiyonu (Inertia / WCSS):
J = Σ Σ ||x - μk||²
    k  x∈Ck

Burada:
- Ck: k. kümedeki noktalar
- μk: k. kümenin centroid'i (merkezi)
- ||x - μk||²: Öklid mesafesinin karesi

Algoritma Adımları:
1. Başlangıç: k adet centroid rastgele seç
2. Atama: Her noktayı en yakın centroid'e ata
3. Güncelleme: Her kümenin yeni centroid'ini hesapla (ortalama)
4. Yakınsama: Centroid'ler değişmiyorsa veya max iterasyona ulaşıldıysa dur
5. Değilse 2. adıma dön

K-Means++ Başlatma:
- İlk centroid rastgele seç
- Sonraki centroid'leri uzaklık orantılı olasılıkla seç
- Daha iyi yakınsama sağlar
"""

import math
import random
from typing import List, Tuple, Optional


class KMeans:
    """
    K-Means Kümeleme Algoritması
    
    Parametreler:
    ------------
    n_clusters : int, default=3
        Küme sayısı (k).
    
    max_iter : int, default=300
        Maksimum iterasyon sayısı.
    
    tol : float, default=1e-4
        Yakınsama toleransı. Centroid değişimi bu değerin altındaysa dur.
    
    init : str, default='kmeans++'
        Başlatma yöntemi: 'random' veya 'kmeans++'.
    
    random_state : int, default=None
        Rastgelelik için seed değeri.
    
    Öznitelikler:
    ------------
    cluster_centers_ : List[List[float]]
        Küme merkezleri (centroid'ler).
    
    labels_ : List[int]
        Her örnek için küme etiketi.
    
    inertia_ : float
        Küme içi kareler toplamı (WCSS).
    
    n_iter_ : int
        Yakınsama için gereken iterasyon sayısı.
    
    Örnek Kullanım:
    --------------
    >>> kmeans = KMeans(n_clusters=3)
    >>> kmeans.fit(X)
    >>> labels = kmeans.labels_
    >>> centers = kmeans.cluster_centers_
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = 'kmeans++',
        random_state: Optional[int] = None
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters 1'den küçük olamaz")
        
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        self.cluster_centers_: List[List[float]] = []
        self.labels_: List[int] = []
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
    
    def fit(self, X: List[List[float]]) -> 'KMeans':
        """
        K-Means algoritmasını çalıştır.
        
        Args:
            X: Veri noktaları (n_samples, n_features)
        
        Returns:
            self: Fit edilmiş model
        """
        if len(X) < self.n_clusters:
            raise ValueError(f"Örnek sayısı ({len(X)}) küme sayısından ({self.n_clusters}) az olamaz")
        
        # Rastgelelik ayarla
        if self.random_state is not None:
            random.seed(self.random_state)
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # 1. Centroid'leri başlat
        if self.init == 'kmeans++':
            self.cluster_centers_ = self._init_kmeans_plusplus(X)
        else:
            self.cluster_centers_ = self._init_random(X)
        
        # Ana döngü
        for iteration in range(self.max_iter):
            # 2. Atama adımı: Her noktayı en yakın centroid'e ata
            new_labels = self._assign_clusters(X)
            
            # 3. Güncelleme adımı: Yeni centroid'leri hesapla
            new_centers = self._update_centers(X, new_labels, n_features)
            
            # 4. Yakınsama kontrolü
            center_shift = self._calculate_center_shift(self.cluster_centers_, new_centers)
            
            self.cluster_centers_ = new_centers
            self.labels_ = new_labels
            self.n_iter_ = iteration + 1
            
            if center_shift < self.tol:
                break
        
        # Inertia hesapla
        self.inertia_ = self._calculate_inertia(X)
        
        return self
    
    def _init_random(self, X: List[List[float]]) -> List[List[float]]:
        """
        Rastgele centroid başlatma.
        
        Veri noktalarından rastgele k tanesi seçilir.
        """
        indices = random.sample(range(len(X)), self.n_clusters)
        return [X[i].copy() for i in indices]
    
    def _init_kmeans_plusplus(self, X: List[List[float]]) -> List[List[float]]:
        """
        K-Means++ başlatma.
        
        Algoritma:
        1. İlk centroid'i rastgele seç
        2. Her nokta için en yakın centroid'e olan mesafeyi hesapla
        3. Mesafe² ile orantılı olasılıkla yeni centroid seç
        4. k centroid seçilene kadar tekrarla
        
        Bu yöntem, centroid'lerin birbirinden uzak olmasını sağlar
        ve daha iyi yakınsama elde edilir.
        """
        n_samples = len(X)
        centers = []
        
        # İlk centroid'i rastgele seç
        first_idx = random.randint(0, n_samples - 1)
        centers.append(X[first_idx].copy())
        
        # Kalan centroid'leri seç
        for _ in range(1, self.n_clusters):
            # Her nokta için en yakın centroid'e olan mesafe²
            distances_sq = []
            for x in X:
                min_dist_sq = float('inf')
                for center in centers:
                    dist_sq = self._squared_distance(x, center)
                    min_dist_sq = min(min_dist_sq, dist_sq)
                distances_sq.append(min_dist_sq)
            
            # Mesafe² ile orantılı olasılık
            total_dist_sq = sum(distances_sq)
            if total_dist_sq == 0:
                # Tüm noktalar mevcut centroid'lerle örtüşüyor
                remaining = [i for i in range(n_samples) if X[i] not in centers]
                if remaining:
                    next_idx = random.choice(remaining)
                else:
                    next_idx = random.randint(0, n_samples - 1)
            else:
                probabilities = [d / total_dist_sq for d in distances_sq]
                next_idx = self._weighted_choice(probabilities)
            
            centers.append(X[next_idx].copy())
        
        return centers
    
    def _weighted_choice(self, probabilities: List[float]) -> int:
        """
        Olasılık dağılımına göre indeks seç.
        """
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probabilities):
            cumsum += p
            if r <= cumsum:
                return i
        return len(probabilities) - 1
    
    def _assign_clusters(self, X: List[List[float]]) -> List[int]:
        """
        Her noktayı en yakın centroid'e ata.
        
        Args:
            X: Veri noktaları
        
        Returns:
            labels: Küme etiketleri
        """
        labels = []
        for x in X:
            min_dist = float('inf')
            closest_cluster = 0
            
            for k, center in enumerate(self.cluster_centers_):
                dist = self._euclidean_distance(x, center)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = k
            
            labels.append(closest_cluster)
        
        return labels
    
    def _update_centers(
        self,
        X: List[List[float]],
        labels: List[int],
        n_features: int
    ) -> List[List[float]]:
        """
        Yeni centroid'leri hesapla (her kümenin ortalaması).
        
        μk = (1/|Ck|) Σ x
                      x∈Ck
        
        Args:
            X: Veri noktaları
            labels: Mevcut küme etiketleri
            n_features: Özellik sayısı
        
        Returns:
            new_centers: Güncellenmiş centroid'ler
        """
        new_centers = []
        
        for k in range(self.n_clusters):
            # k. kümedeki noktaları bul
            cluster_points = [X[i] for i in range(len(X)) if labels[i] == k]
            
            if len(cluster_points) == 0:
                # Boş küme - eski centroid'i koru
                new_centers.append(self.cluster_centers_[k].copy())
            else:
                # Ortalama hesapla
                center = []
                for j in range(n_features):
                    mean_j = sum(p[j] for p in cluster_points) / len(cluster_points)
                    center.append(mean_j)
                new_centers.append(center)
        
        return new_centers
    
    def _calculate_center_shift(
        self,
        old_centers: List[List[float]],
        new_centers: List[List[float]]
    ) -> float:
        """
        Centroid değişim miktarını hesapla (yakınsama kontrolü için).
        """
        total_shift = 0.0
        for old, new in zip(old_centers, new_centers):
            total_shift += self._euclidean_distance(old, new)
        return total_shift
    
    def _calculate_inertia(self, X: List[List[float]]) -> float:
        """
        Inertia (WCSS) hesapla.
        
        Inertia = Σ Σ ||x - μk||²
                  k x∈Ck
        
        Args:
            X: Veri noktaları
        
        Returns:
            inertia: Küme içi kareler toplamı
        """
        inertia = 0.0
        for i, x in enumerate(X):
            center = self.cluster_centers_[self.labels_[i]]
            inertia += self._squared_distance(x, center)
        return inertia
    
    def _euclidean_distance(self, p: List[float], q: List[float]) -> float:
        """
        Öklid mesafesi hesapla.
        """
        return math.sqrt(self._squared_distance(p, q))
    
    def _squared_distance(self, p: List[float], q: List[float]) -> float:
        """
        Öklid mesafesinin karesi (daha hızlı karşılaştırma için).
        """
        return sum((pi - qi) ** 2 for pi, qi in zip(p, q))
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Yeni veriler için küme tahmini yap.
        
        Args:
            X: Test noktaları
        
        Returns:
            labels: Tahmin edilen küme etiketleri
        """
        if not self.cluster_centers_:
            raise ValueError("Model henüz fit edilmedi.")
        
        return self._assign_clusters(X)
    
    def fit_predict(self, X: List[List[float]]) -> List[int]:
        """
        Fit ve predict'i tek adımda yap.
        
        Args:
            X: Veri noktaları
        
        Returns:
            labels: Küme etiketleri
        """
        self.fit(X)
        return self.labels_
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """
        Her nokta için centroid'lere olan mesafeleri döndür.
        
        Args:
            X: Veri noktaları
        
        Returns:
            distances: Her nokta için k mesafe
        """
        if not self.cluster_centers_:
            raise ValueError("Model henüz fit edilmedi.")
        
        distances = []
        for x in X:
            dists = [self._euclidean_distance(x, center) for center in self.cluster_centers_]
            distances.append(dists)
        
        return distances
    
    def get_params(self) -> dict:
        """Model parametrelerini döndür."""
        return {
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'init': self.init,
            'random_state': self.random_state
        }
    
    def get_cluster_sizes(self) -> List[int]:
        """Her kümedeki örnek sayısını döndür."""
        if not self.labels_:
            raise ValueError("Model henüz fit edilmedi.")
        
        sizes = [0] * self.n_clusters
        for label in self.labels_:
            sizes[label] += 1
        return sizes
