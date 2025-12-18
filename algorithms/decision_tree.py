"""
Decision Tree (Karar Ağacı) Algoritması - Vanilla Python Implementasyonu

Bu implementasyon üç farklı algoritmayı destekler:
- ID3: Information Gain (Entropy bazlı)
- C4.5: Gain Ratio (ID3'ün geliştirilmiş versiyonu)
- CART: Gini Index veya Twoing kriteri

Matematiksel Temeller:
=====================

1. ENTROPY (ID3 için)
   H(S) = -Σ p(c) * log2(p(c))
   
   Burada:
   - S: Veri seti
   - p(c): c sınıfının olasılığı
   - Entropy bilgi belirsizliğini ölçer (0 = tamamen homojen, max = tamamen heterojen)

2. INFORMATION GAIN (ID3 için)
   IG(S, A) = H(S) - Σ (|Sv|/|S|) * H(Sv)
   
   Burada:
   - A: Öznitelik
   - Sv: A özniteliğinin v değerine sahip alt küme
   - Hangi özniteliğin en çok bilgi kazandırdığını ölçer

3. GAIN RATIO (C4.5 için)
   GR(S, A) = IG(S, A) / SplitInfo(S, A)
   SplitInfo(S, A) = -Σ (|Sv|/|S|) * log2(|Sv|/|S|)
   
   - Information Gain'in çok değerli özniteliklere karşı bias'ını düzeltir

4. GINI INDEX (CART için)
   Gini(S) = 1 - Σ p(c)²
   
   - Entropy'ye alternatif, daha hızlı hesaplanır
   - 0 = tamamen saf, 0.5 = maksimum impurity (binary için)

5. TWOING CRITERION (CART alternatif)
   Twoing(S, A) = (PL * PR / 4) * [Σ |P(c|L) - P(c|R)|]²
   
   - Çok sınıflı problemlerde daha iyi performans
   - İki gruba bölme optimizasyonu
"""

import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Union


class TreeNode:
    """
    Karar ağacı düğümü.
    
    Attributes:
        feature_index: Bölme yapılan öznitelik indeksi (iç düğümler için)
        threshold: Bölme eşik değeri (numerik öznitelikler için)
        value: Düğüm değeri (kategorik öznitelikler için)
        left: Sol alt ağaç
        right: Sağ alt ağaç
        children: Alt düğümler sözlüğü (çoklu bölme için)
        prediction: Yaprak düğüm tahmini
        is_leaf: Yaprak düğüm mü?
        samples: Bu düğümdeki örnek sayısı
        class_distribution: Sınıf dağılımı
    """
    def __init__(self):
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.value: Optional[Any] = None
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None
        self.children: Dict[Any, 'TreeNode'] = {}
        self.prediction: Optional[Any] = None
        self.is_leaf: bool = False
        self.samples: int = 0
        self.class_distribution: Dict[Any, int] = {}
        # Eğitim amaçlı metrikler
        self.entropy: Optional[float] = None
        self.gini: Optional[float] = None
        self.information_gain: Optional[float] = None


class DecisionTree:
    """
    Karar Ağacı Sınıflandırıcısı (ID3, C4.5, CART)
    
    Parametreler:
    ------------
    criterion : str, default='entropy'
        Bölme kriteri: 'entropy' (ID3), 'gain_ratio' (C4.5), 'gini' (CART), 'twoing' (CART)
    
    max_depth : int, default=None
        Maksimum ağaç derinliği. None ise sınırsız.
    
    min_samples_split : int, default=2
        Bölme için gereken minimum örnek sayısı.
    
    min_samples_leaf : int, default=1
        Yaprak düğümde bulunması gereken minimum örnek sayısı.
    
    Örnek Kullanım:
    --------------
    >>> dt = DecisionTree(criterion='entropy')  # ID3
    >>> dt = DecisionTree(criterion='gain_ratio')  # C4.5
    >>> dt = DecisionTree(criterion='gini')  # CART
    >>> dt.fit(X_train, y_train)
    >>> predictions = dt.predict(X_test)
    """
    
    SUPPORTED_CRITERIA = ['entropy', 'gain_ratio', 'gini', 'twoing']
    
    def __init__(
        self,
        criterion: str = 'entropy',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        if criterion not in self.SUPPORTED_CRITERIA:
            raise ValueError(f"Desteklenmeyen kriter: {criterion}. Desteklenenler: {self.SUPPORTED_CRITERIA}")
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: Optional[TreeNode] = None
        self.n_features: int = 0
        self.n_classes: int = 0
        self.classes: List[Any] = []
        self.feature_importances_: List[float] = []
    
    def fit(self, X: List[List[float]], y: List[Any]) -> 'DecisionTree':
        """
        Karar ağacını eğit.
        
        Args:
            X: Eğitim özellikleri (n_samples, n_features)
            y: Eğitim etiketleri (n_samples,)
        
        Returns:
            self: Eğitilmiş model
        """
        if len(X) != len(y):
            raise ValueError(f"X ve y uzunlukları eşleşmiyor: {len(X)} != {len(y)}")
        
        if len(X) == 0:
            raise ValueError("Boş veri seti ile eğitim yapılamaz")
        
        self.n_features = len(X[0])
        self.classes = list(set(y))
        self.n_classes = len(self.classes)
        self.feature_importances_ = [0.0] * self.n_features
        
        # Ağacı oluştur
        self.root = self._build_tree(X, y, depth=0)
        
        # Feature importance'ı normalize et
        total_importance = sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ = [imp / total_importance for imp in self.feature_importances_]
        
        return self
    
    def _build_tree(self, X: List[List[float]], y: List[Any], depth: int) -> TreeNode:
        """
        Ağacı rekürsif olarak oluştur.
        
        Args:
            X: Özellikler
            y: Etiketler
            depth: Mevcut derinlik
        
        Returns:
            node: Oluşturulan düğüm
        """
        node = TreeNode()
        node.samples = len(y)
        node.class_distribution = dict(Counter(y))
        
        # Eğitim amaçlı: Bu düğümün entropy/gini değerini hesapla
        if self.criterion in ['entropy', 'gain_ratio']:
            node.entropy = self._entropy(y)
        elif self.criterion in ['gini', 'twoing']:
            node.gini = self._gini(y)
        
        # Durma koşulları
        # 1. Tüm örnekler aynı sınıfta
        if len(set(y)) == 1:
            node.is_leaf = True
            node.prediction = y[0]
            return node
        
        # 2. Maksimum derinliğe ulaşıldı
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            node.prediction = self._majority_vote(y)
            return node
        
        # 3. Minimum örnek sayısı sağlanmıyor
        if len(y) < self.min_samples_split:
            node.is_leaf = True
            node.prediction = self._majority_vote(y)
            return node
        
        # En iyi bölmeyi bul
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # 4. Hiçbir bölme bilgi kazancı sağlamıyor
        if best_gain <= 0 or best_feature is None:
            node.is_leaf = True
            node.prediction = self._majority_vote(y)
            return node
        
        # Feature importance güncelle
        self.feature_importances_[best_feature] += best_gain * len(y)
        
        # Eğitim amaçlı: Information gain'i kaydet
        node.information_gain = best_gain
        
        node.feature_index = best_feature
        node.threshold = best_threshold
        
        # Veriyi böl ve alt ağaçları oluştur
        if best_threshold is not None:
            # Numerik öznitelik - binary split
            left_mask = [X[i][best_feature] <= best_threshold for i in range(len(X))]
            right_mask = [not m for m in left_mask]
            
            X_left = [X[i] for i in range(len(X)) if left_mask[i]]
            y_left = [y[i] for i in range(len(y)) if left_mask[i]]
            X_right = [X[i] for i in range(len(X)) if right_mask[i]]
            y_right = [y[i] for i in range(len(y)) if right_mask[i]]
            
            # Minimum yaprak örnek sayısı kontrolü
            if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                node.is_leaf = True
                node.prediction = self._majority_vote(y)
                return node
            
            node.left = self._build_tree(X_left, y_left, depth + 1)
            node.right = self._build_tree(X_right, y_right, depth + 1)
        else:
            # Kategorik öznitelik - multi-way split
            feature_values = set(X[i][best_feature] for i in range(len(X)))
            for value in feature_values:
                mask = [X[i][best_feature] == value for i in range(len(X))]
                X_subset = [X[i] for i in range(len(X)) if mask[i]]
                y_subset = [y[i] for i in range(len(y)) if mask[i]]
                
                if len(y_subset) >= self.min_samples_leaf:
                    node.children[value] = self._build_tree(X_subset, y_subset, depth + 1)
        
        return node
    
    def _find_best_split(self, X: List[List[float]], y: List[Any]) -> Tuple[Optional[int], Optional[float], float]:
        """
        En iyi bölme noktasını bul.
        
        Args:
            X: Özellikler
            y: Etiketler
        
        Returns:
            best_feature: En iyi öznitelik indeksi
            best_threshold: En iyi eşik değeri (numerik için)
            best_gain: En iyi bilgi kazancı
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        current_impurity = self._calculate_impurity(y)
        
        for feature_idx in range(self.n_features):
            feature_values = [X[i][feature_idx] for i in range(len(X))]
            unique_values = sorted(set(feature_values))
            
            # Numerik öznitelik kontrolü
            is_numeric = all(isinstance(v, (int, float)) for v in unique_values)
            
            if is_numeric and len(unique_values) > 2:
                # Numerik öznitelik - potansiyel eşik değerlerini dene
                # Eşik: 2'den fazla farklı değer varsa numerik olarak işle
                thresholds = self._get_thresholds(unique_values)
                
                for threshold in thresholds:
                    gain = self._calculate_split_gain(X, y, feature_idx, threshold, current_impurity)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
            else:
                # Kategorik öznitelik veya az değerli numerik
                gain = self._calculate_split_gain(X, y, feature_idx, None, current_impurity)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = None
        
        return best_feature, best_threshold, best_gain
    
    def _get_thresholds(self, values: List[float]) -> List[float]:
        """
        Potansiyel eşik değerlerini hesapla (ardışık değerlerin ortalaması).
        """
        thresholds = []
        for i in range(len(values) - 1):
            thresholds.append((values[i] + values[i + 1]) / 2)
        return thresholds
    
    def _calculate_split_gain(
        self,
        X: List[List[float]],
        y: List[Any],
        feature_idx: int,
        threshold: Optional[float],
        current_impurity: float
    ) -> float:
        """
        Bölme için bilgi kazancını hesapla.
        """
        n = len(y)
        
        if threshold is not None:
            # Binary split
            left_mask = [X[i][feature_idx] <= threshold for i in range(len(X))]
            y_left = [y[i] for i in range(len(y)) if left_mask[i]]
            y_right = [y[i] for i in range(len(y)) if not left_mask[i]]
            
            if len(y_left) == 0 or len(y_right) == 0:
                return -float('inf')
            
            # Ağırlıklı impurity
            weighted_impurity = (len(y_left) / n) * self._calculate_impurity(y_left) + \
                               (len(y_right) / n) * self._calculate_impurity(y_right)
            
            gain = current_impurity - weighted_impurity
            
            if self.criterion == 'gain_ratio':
                split_info = self._split_info_binary(len(y_left), len(y_right), n)
                if split_info > 0:
                    gain = gain / split_info
            elif self.criterion == 'twoing':
                gain = self._twoing_gain(y_left, y_right)
        
        else:
            # Multi-way split
            feature_values = [X[i][feature_idx] for i in range(len(X))]
            unique_values = set(feature_values)
            
            subsets = {}
            for value in unique_values:
                subsets[value] = [y[i] for i in range(len(y)) if feature_values[i] == value]
            
            weighted_impurity = sum(
                (len(subset) / n) * self._calculate_impurity(subset)
                for subset in subsets.values()
            )
            
            gain = current_impurity - weighted_impurity
            
            if self.criterion == 'gain_ratio':
                split_info = self._split_info_multi(subsets, n)
                if split_info > 0:
                    gain = gain / split_info
        
        return gain
    
    def _calculate_impurity(self, y: List[Any]) -> float:
        """
        Impurity (safsızlık) hesapla - seçilen kritere göre.
        """
        if self.criterion in ['entropy', 'gain_ratio']:
            return self._entropy(y)
        elif self.criterion in ['gini', 'twoing']:
            return self._gini(y)
        return 0.0
    
    def _entropy(self, y: List[Any]) -> float:
        """
        Shannon Entropy hesapla.
        
        H(S) = -Σ p(c) * log2(p(c))
        
        Args:
            y: Sınıf etiketleri
        
        Returns:
            entropy: Entropy değeri (0 ile log2(n_classes) arasında)
        """
        n = len(y)
        if n == 0:
            return 0.0
        
        # Sınıf sayılarını hesapla
        counter = Counter(y)
        
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / n
                # log2(p) hesapla, p=0 için tanımsız olduğundan kontrol et
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _gini(self, y: List[Any]) -> float:
        """
        Gini Index hesapla.
        
        Gini(S) = 1 - Σ p(c)²
        
        Args:
            y: Sınıf etiketleri
        
        Returns:
            gini: Gini indeksi (0 ile 1-1/n_classes arasında)
        """
        n = len(y)
        if n == 0:
            return 0.0
        
        counter = Counter(y)
        
        gini = 1.0
        for count in counter.values():
            p = count / n
            gini -= p ** 2
        
        return gini
    
    def _twoing_gain(self, y_left: List[Any], y_right: List[Any]) -> float:
        """
        Twoing kriteri ile kazanç hesapla.
        
        Twoing(S, A) = (PL * PR / 4) * [Σ |P(c|L) - P(c|R)|]²
        
        Args:
            y_left: Sol alt küme etiketleri
            y_right: Sağ alt küme etiketleri
        
        Returns:
            twoing: Twoing değeri
        """
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        p_left = n_left / n_total
        p_right = n_right / n_total
        
        # Sınıf dağılımlarını hesapla
        counter_left = Counter(y_left)
        counter_right = Counter(y_right)
        all_classes = set(y_left + y_right)
        
        # |P(c|L) - P(c|R)| toplamı
        diff_sum = 0.0
        for c in all_classes:
            p_c_left = counter_left.get(c, 0) / n_left
            p_c_right = counter_right.get(c, 0) / n_right
            diff_sum += abs(p_c_left - p_c_right)
        
        twoing = (p_left * p_right / 4) * (diff_sum ** 2)
        return twoing
    
    def _split_info_binary(self, n_left: int, n_right: int, n: int) -> float:
        """
        Binary split için Split Information hesapla (C4.5).
        
        SplitInfo = -Σ (|Sv|/|S|) * log2(|Sv|/|S|)
        """
        split_info = 0.0
        
        if n_left > 0:
            p_left = n_left / n
            split_info -= p_left * math.log2(p_left)
        
        if n_right > 0:
            p_right = n_right / n
            split_info -= p_right * math.log2(p_right)
        
        return split_info
    
    def _split_info_multi(self, subsets: Dict[Any, List[Any]], n: int) -> float:
        """
        Multi-way split için Split Information hesapla.
        """
        split_info = 0.0
        
        for subset in subsets.values():
            if len(subset) > 0:
                p = len(subset) / n
                split_info -= p * math.log2(p)
        
        return split_info
    
    def _majority_vote(self, y: List[Any]) -> Any:
        """
        Çoğunluk oylaması ile sınıf belirle.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X: List[List[float]]) -> List[Any]:
        """
        Test verileri için tahmin yap.
        
        Args:
            X: Test özellikleri
        
        Returns:
            predictions: Tahmin edilen sınıflar
        """
        if self.root is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        return [self._predict_single(x, self.root) for x in X]
    
    def _predict_single(self, x: List[float], node: TreeNode) -> Any:
        """
        Tek bir örnek için tahmin yap (rekürsif).
        """
        if node.is_leaf:
            return node.prediction
        
        feature_value = x[node.feature_index]
        
        if node.threshold is not None:
            # Binary split
            if feature_value <= node.threshold:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)
        else:
            # Multi-way split
            if feature_value in node.children:
                return self._predict_single(x, node.children[feature_value])
            else:
                # Bilinmeyen değer - çoğunluk tahmini
                return node.prediction if node.prediction else self._majority_vote(
                    list(node.class_distribution.keys())
                )
    
    def predict_proba(self, X: List[List[float]]) -> List[Dict[Any, float]]:
        """
        Sınıf olasılıklarını tahmin et.
        """
        if self.root is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        return [self._predict_proba_single(x, self.root) for x in X]
    
    def _predict_proba_single(self, x: List[float], node: TreeNode) -> Dict[Any, float]:
        """
        Tek bir örnek için olasılık tahmini.
        """
        if node.is_leaf:
            total = sum(node.class_distribution.values())
            return {k: v / total for k, v in node.class_distribution.items()}
        
        feature_value = x[node.feature_index]
        
        if node.threshold is not None:
            if feature_value <= node.threshold:
                return self._predict_proba_single(x, node.left)
            else:
                return self._predict_proba_single(x, node.right)
        else:
            if feature_value in node.children:
                return self._predict_proba_single(x, node.children[feature_value])
            else:
                total = sum(node.class_distribution.values())
                return {k: v / total for k, v in node.class_distribution.items()}
    
    def get_params(self) -> dict:
        """Model parametrelerini döndür."""
        return {
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }
    
    def get_tree_structure(self) -> dict:
        """
        Ağaç yapısını sözlük olarak döndür (görselleştirme için).
        """
        if self.root is None:
            return {}
        return self._node_to_dict(self.root)
    
    def get_tree_for_d3(self) -> dict:
        """
        D3.js görselleştirmesi için ağaç yapısını döndür.
        
        Returns:
            D3.js hierarchy formatında ağaç
        """
        if self.root is None:
            return {}
        return self._node_to_d3_dict(self.root, depth=0)
    
    def _node_to_d3_dict(self, node: TreeNode, depth: int = 0) -> dict:
        """Düğümü D3.js formatına dönüştür."""
        # Sınıf dağılımından dominant class ve yüzde
        if node.class_distribution:
            total = sum(node.class_distribution.values())
            dominant_class = max(node.class_distribution, key=node.class_distribution.get)
            dominant_pct = (node.class_distribution[dominant_class] / total * 100) if total > 0 else 0
        else:
            dominant_class = None
            dominant_pct = 0
        
        result = {
            'name': '',  # Sonra doldurulacak
            'samples': node.samples,
            'depth': depth,
            'is_leaf': node.is_leaf,
            'entropy': round(node.entropy, 4) if node.entropy is not None else None,
            'gini': round(node.gini, 4) if node.gini is not None else None,
            'information_gain': round(node.information_gain, 4) if node.information_gain is not None else None
        }
        
        if node.is_leaf:
            # Leaf node
            result['name'] = f"Class: {node.prediction}"
            result['prediction'] = node.prediction
            result['class_distribution'] = node.class_distribution
            result['type'] = 'leaf'
        else:
            # Internal node
            if node.threshold is not None:
                result['name'] = f"Feature {node.feature_index} ≤ {node.threshold:.2f}"
                result['feature_index'] = node.feature_index
                result['threshold'] = node.threshold
                result['type'] = 'split'
                
                # Children
                result['children'] = [
                    {
                        **self._node_to_d3_dict(node.left, depth + 1),
                        'edge_label': f'≤ {node.threshold:.2f}'
                    },
                    {
                        **self._node_to_d3_dict(node.right, depth + 1),
                        'edge_label': f'> {node.threshold:.2f}'
                    }
                ]
            else:
                # Categorical split
                result['name'] = f"Feature {node.feature_index}"
                result['feature_index'] = node.feature_index
                result['type'] = 'split'
                result['children'] = [
                    {
                        **self._node_to_d3_dict(child, depth + 1),
                        'edge_label': str(value)
                    }
                    for value, child in node.children.items()
                ]
        
        # Her node için dominant class bilgisi
        if dominant_class:
            result['dominant_class'] = dominant_class
            result['dominant_pct'] = round(dominant_pct, 1)
        
        return result
    
    # ...existing code...
