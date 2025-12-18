"""
Apriori Algoritması - Vanilla Python Implementasyonu

Matematiksel Temel:
==================

Apriori, birliktelik kuralları (association rules) çıkarmak için kullanılan
bir algoritmadır. Market sepeti analizi için yaygın kullanılır.

Temel Kavramlar:
---------------

1. ITEMSET: Bir veya daha fazla öğeden oluşan küme
   Örnek: {ekmek, süt}, {ekmek, süt, yumurta}

2. SUPPORT (Destek):
   support(X) = |X içeren işlemler| / |Toplam işlem sayısı|
   
   Bir itemset'in ne kadar sık geçtiğini gösterir.
   Örnek: {ekmek, süt} 100 işlemden 30'unda varsa → support = 0.30

3. CONFIDENCE (Güven):
   confidence(X → Y) = support(X ∪ Y) / support(X)
   
   X alındığında Y'nin de alınma olasılığı.
   Örnek: confidence({ekmek} → {süt}) = support({ekmek, süt}) / support({ekmek})

4. LIFT (Kaldırma):
   lift(X → Y) = confidence(X → Y) / support(Y)
   
   - lift > 1: X ve Y pozitif ilişkili (birlikte alınma eğilimi var)
   - lift = 1: X ve Y bağımsız
   - lift < 1: X ve Y negatif ilişkili

Apriori Algoritması:
-------------------
1. Tüm 1-itemset'lerin support'unu hesapla
2. min_support'u karşılayanları seç (frequent 1-itemsets)
3. Frequent (k-1)-itemset'lerden aday k-itemset'ler oluştur
4. Adayların support'unu hesapla
5. min_support'u karşılayanları seç (frequent k-itemsets)
6. Yeni frequent itemset bulunamayana kadar tekrarla

Apriori Özelliği (Pruning):
"Bir itemset sık (frequent) değilse, onun üst kümeleri de sık olamaz."
Bu özellik, arama uzayını dramatik olarak azaltır.
"""

from itertools import combinations
from collections import defaultdict
from typing import List, Set, Dict, Tuple, FrozenSet


class Apriori:
    """
    Apriori Birliktelik Kuralları Algoritması
    
    Parametreler:
    ------------
    min_support : float, default=0.1
        Minimum destek eşiği (0 ile 1 arasında).
    
    min_confidence : float, default=0.5
        Minimum güven eşiği (0 ile 1 arasında).
    
    max_itemset_size : int, default=None
        Maksimum itemset boyutu. None ise sınırsız.
    
    Öznitelikler:
    ------------
    frequent_itemsets_ : Dict[int, Dict[FrozenSet, float]]
        Sık itemset'ler ve support değerleri. {boyut: {itemset: support}}
    
    rules_ : List[Dict]
        Çıkarılan birliktelik kuralları.
    
    n_transactions_ : int
        İşlem sayısı.
    
    Örnek Kullanım:
    --------------
    >>> transactions = [
    ...     ['ekmek', 'süt'],
    ...     ['ekmek', 'süt', 'yumurta'],
    ...     ['süt', 'yumurta'],
    ... ]
    >>> apriori = Apriori(min_support=0.3, min_confidence=0.6)
    >>> apriori.fit(transactions)
    >>> rules = apriori.rules_
    """
    
    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.5,
        max_itemset_size: int = None
    ):
        if not 0 < min_support <= 1:
            raise ValueError("min_support 0 ile 1 arasında olmalı")
        if not 0 < min_confidence <= 1:
            raise ValueError("min_confidence 0 ile 1 arasında olmalı")
        
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_itemset_size = max_itemset_size
        
        self.frequent_itemsets_: Dict[int, Dict[FrozenSet, float]] = {}
        self.rules_: List[Dict] = []
        self.n_transactions_: int = 0
        self._transaction_list: List[FrozenSet] = []
    
    def fit(self, transactions: List[List[str]]) -> 'Apriori':
        """
        Apriori algoritmasını çalıştır.
        
        Args:
            transactions: İşlem listesi. Her işlem bir öğe listesi.
                         Örnek: [['ekmek', 'süt'], ['süt', 'yumurta']]
        
        Returns:
            self: Fit edilmiş model
        """
        if len(transactions) == 0:
            raise ValueError("Boş işlem listesi")
        
        # İşlemleri frozenset olarak sakla (hızlı üyelik kontrolü için)
        self._transaction_list = [frozenset(t) for t in transactions]
        self.n_transactions_ = len(transactions)
        
        # 1. Frequent 1-itemset'leri bul
        self.frequent_itemsets_ = {}
        freq_1_itemsets = self._find_frequent_1_itemsets()
        
        if not freq_1_itemsets:
            return self  # Hiç frequent itemset yok
        
        self.frequent_itemsets_[1] = freq_1_itemsets
        
        # 2. Daha büyük frequent itemset'leri bul
        k = 2
        while True:
            if self.max_itemset_size is not None and k > self.max_itemset_size:
                break
            
            # Aday k-itemset'ler oluştur
            candidates = self._generate_candidates(k)
            
            if not candidates:
                break
            
            # Adayların support'unu hesapla ve filtrele
            freq_k_itemsets = self._filter_candidates(candidates)
            
            if not freq_k_itemsets:
                break
            
            self.frequent_itemsets_[k] = freq_k_itemsets
            k += 1
        
        # 3. Birliktelik kurallarını çıkar
        self.rules_ = self._generate_rules()
        
        return self
    
    def _find_frequent_1_itemsets(self) -> Dict[FrozenSet, float]:
        """
        Frequent 1-itemset'leri bul.
        
        Returns:
            frequent: {frozenset({item}): support} sözlüğü
        """
        # Her öğenin kaç işlemde geçtiğini say
        item_counts = defaultdict(int)
        for transaction in self._transaction_list:
            for item in transaction:
                item_counts[item] += 1
        
        # Support hesapla ve filtrele
        frequent = {}
        for item, count in item_counts.items():
            support = count / self.n_transactions_
            if support >= self.min_support:
                frequent[frozenset([item])] = support
        
        return frequent
    
    def _generate_candidates(self, k: int) -> Set[FrozenSet]:
        """
        Aday k-itemset'ler oluştur (Apriori-gen).
        
        Frequent (k-1)-itemset'lerden k-itemset adayları oluşturur.
        Apriori özelliğini kullanarak budama yapar.
        
        Args:
            k: Itemset boyutu
        
        Returns:
            candidates: Aday itemset'ler kümesi
        """
        if k - 1 not in self.frequent_itemsets_:
            return set()
        
        freq_prev = list(self.frequent_itemsets_[k - 1].keys())
        candidates = set()
        
        # Her frequent (k-1)-itemset çiftini birleştir
        for i in range(len(freq_prev)):
            for j in range(i + 1, len(freq_prev)):
                # İki itemset'i birleştir
                union = freq_prev[i] | freq_prev[j]
                
                # Boyut kontrolü
                if len(union) == k:
                    # Apriori pruning: Tüm (k-1)-alt kümeleri frequent olmalı
                    if self._all_subsets_frequent(union, k - 1):
                        candidates.add(union)
        
        return candidates
    
    def _all_subsets_frequent(self, itemset: FrozenSet, subset_size: int) -> bool:
        """
        Bir itemset'in tüm alt kümelerinin frequent olup olmadığını kontrol et.
        
        Apriori özelliği: "Eğer bir alt küme frequent değilse, 
        üst küme de frequent olamaz."
        
        Args:
            itemset: Kontrol edilecek itemset
            subset_size: Alt küme boyutu
        
        Returns:
            all_frequent: Tüm alt kümeler frequent mi?
        """
        freq_itemsets = self.frequent_itemsets_.get(subset_size, {})
        
        for subset in combinations(itemset, subset_size):
            if frozenset(subset) not in freq_itemsets:
                return False
        
        return True
    
    def _filter_candidates(self, candidates: Set[FrozenSet]) -> Dict[FrozenSet, float]:
        """
        Adayların support'unu hesapla ve min_support'u karşılayanları döndür.
        
        Args:
            candidates: Aday itemset'ler
        
        Returns:
            frequent: Frequent itemset'ler ve support değerleri
        """
        # Her adayın kaç işlemde geçtiğini say
        candidate_counts = defaultdict(int)
        
        for transaction in self._transaction_list:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1
        
        # Support hesapla ve filtrele
        frequent = {}
        for candidate, count in candidate_counts.items():
            support = count / self.n_transactions_
            if support >= self.min_support:
                frequent[candidate] = support
        
        return frequent
    
    def _generate_rules(self) -> List[Dict]:
        """
        Frequent itemset'lerden birliktelik kuralları çıkar.
        
        Her frequent itemset için tüm olası kuralları dene:
        - X → Y where X ∪ Y = itemset and X ∩ Y = ∅
        
        Returns:
            rules: Kural listesi
        """
        rules = []
        
        # En az 2 öğeli itemset'lerden kural çıkarılabilir
        for k in range(2, max(self.frequent_itemsets_.keys()) + 1 if self.frequent_itemsets_ else 0):
            if k not in self.frequent_itemsets_:
                continue
            
            for itemset, support_xy in self.frequent_itemsets_[k].items():
                # Itemset'in tüm boş olmayan alt kümelerini dene (antecedent olarak)
                items = list(itemset)
                
                for i in range(1, len(items)):
                    for antecedent_tuple in combinations(items, i):
                        antecedent = frozenset(antecedent_tuple)
                        consequent = itemset - antecedent
                        
                        if len(consequent) == 0:
                            continue
                        
                        # Confidence hesapla
                        support_x = self._get_support(antecedent)
                        if support_x == 0:
                            continue
                        
                        confidence = support_xy / support_x
                        
                        if confidence >= self.min_confidence:
                            # Lift hesapla
                            support_y = self._get_support(consequent)
                            lift = confidence / support_y if support_y > 0 else 0
                            
                            rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support_xy,
                                'confidence': confidence,
                                'lift': lift
                            })
        
        # Confidence'a göre sırala
        rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        return rules
    
    def _get_support(self, itemset: FrozenSet) -> float:
        """
        Bir itemset'in support değerini döndür.
        
        Args:
            itemset: Sorgulanacak itemset
        
        Returns:
            support: Support değeri
        """
        size = len(itemset)
        if size in self.frequent_itemsets_:
            return self.frequent_itemsets_[size].get(itemset, 0)
        return 0
    
    def get_frequent_itemsets(self) -> List[Dict]:
        """
        Tüm frequent itemset'leri düz liste olarak döndür.
        
        Returns:
            itemsets: [{'itemset': set, 'support': float, 'size': int}, ...]
        """
        itemsets = []
        for size, freq_dict in self.frequent_itemsets_.items():
            for itemset, support in freq_dict.items():
                itemsets.append({
                    'itemset': set(itemset),
                    'support': support,
                    'size': size
                })
        
        # Support'a göre sırala
        itemsets.sort(key=lambda x: x['support'], reverse=True)
        
        return itemsets
    
    def get_rules(self) -> List[Dict]:
        """
        Tüm birliktelik kurallarını döndür.
        
        Returns:
            rules: Kural listesi
        """
        return self.rules_
    
    def get_params(self) -> dict:
        """Model parametrelerini döndür."""
        return {
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'max_itemset_size': self.max_itemset_size
        }
    
    def print_rules(self, max_rules: int = 10) -> None:
        """
        Kuralları okunabilir formatta yazdır.
        
        Args:
            max_rules: Maksimum kural sayısı
        """
        print(f"\n{'='*60}")
        print(f"BİRLİKTELİK KURALLARI (Top {min(max_rules, len(self.rules_))})")
        print(f"{'='*60}")
        print(f"Min Support: {self.min_support}, Min Confidence: {self.min_confidence}")
        print(f"Toplam Kural Sayısı: {len(self.rules_)}")
        print(f"{'='*60}\n")
        
        for i, rule in enumerate(self.rules_[:max_rules]):
            antecedent = ', '.join(sorted(rule['antecedent']))
            consequent = ', '.join(sorted(rule['consequent']))
            print(f"{i+1}. {{{antecedent}}} → {{{consequent}}}")
            print(f"   Support: {rule['support']:.3f}, "
                  f"Confidence: {rule['confidence']:.3f}, "
                  f"Lift: {rule['lift']:.3f}")
            print()
    
    def get_summary(self) -> Dict:
        """
        Özet istatistikler döndür.
        
        Returns:
            summary: Özet sözlüğü
        """
        total_itemsets = sum(len(v) for v in self.frequent_itemsets_.values())
        
        return {
            'n_transactions': self.n_transactions_,
            'n_frequent_itemsets': total_itemsets,
            'n_rules': len(self.rules_),
            'max_itemset_size': max(self.frequent_itemsets_.keys()) if self.frequent_itemsets_ else 0,
            'itemsets_by_size': {k: len(v) for k, v in self.frequent_itemsets_.items()},
            'avg_confidence': sum(r['confidence'] for r in self.rules_) / len(self.rules_) if self.rules_ else 0,
            'avg_lift': sum(r['lift'] for r in self.rules_) / len(self.rules_) if self.rules_ else 0
        }
