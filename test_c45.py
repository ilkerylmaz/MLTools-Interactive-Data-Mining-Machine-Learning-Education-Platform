"""
C4.5 (Gain Ratio) Karar Ağacı Test Scripti
C45.csv üzerinde algoritma testi
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/ilker/Documents/mltools')

from algorithms.decision_tree import DecisionTree

# CSV'yi yükle
df = pd.read_csv('sample_data/C45.csv')

print("=" * 60)
print("C4.5 KARAR AĞACI TESTİ")
print("=" * 60)
print("\nVeri:")
print(df)
print(f"\nToplam örnek sayısı: {len(df)}")

# ====================================================================
# ÖN İŞLEME: Sayısal kolonları kategorik hale dönüştür (discretization)
# ====================================================================
print("\n" + "=" * 60)
print("ÖN İŞLEME: Sayısal Kolonların Kategorikleştirilmesi")
print("=" * 60)

df_processed = df.copy()

for col in df.columns[:-1]:  # Son kolon hariç (hedef değişken)
    if pd.api.types.is_numeric_dtype(df[col]):
        mean_val = df[col].mean()
        print(f"\n{col}:")
        print(f"  Ortalama: {mean_val:.2f}")
        print(f"  Orijinal değerler: {sorted(df[col].unique())}")
        
        # Ortalamaya göre kategorize et
        df_processed[col] = df[col].apply(
            lambda x: 'düşük' if x <= mean_val else 'yüksek'
        )
        
        print(f"  Dönüştürülmüş: {df_processed[col].unique()}")
        
        # Dağılımı göster
        low_count = (df[col] <= mean_val).sum()
        high_count = (df[col] > mean_val).sum()
        print(f"  Dağılım: düşük={low_count}, yüksek={high_count}")
    else:
        print(f"\n{col}: Zaten kategorik (değişiklik yok)")

print("\n" + "=" * 60)
print("Dönüştürülmüş Veri:")
print("=" * 60)
print(df_processed)

# Özellikler ve etiketler
X = df_processed.iloc[:, :-1].values.tolist()
y = df_processed.iloc[:, -1].values.tolist()

feature_names = df_processed.columns[:-1].tolist()
print(f"\nÖzellikler: {feature_names}")
print(f"Hedef: {df_processed.columns[-1]}")

# C4.5 modeli oluştur (Gain Ratio)
model = DecisionTree(criterion='gain_ratio', max_depth=None)
model.fit(X, y)

print("\n" + "=" * 60)
print("OLUŞAN KARAR AĞACI (C4.5 - Gain Ratio)")
print("=" * 60)

# Ağaç yapısını yazdır
def print_tree(node, depth=0, prefix="ROOT", feature_names=None):
    indent = "  " * depth
    
    if node.is_leaf:
        dist = node.class_distribution
        total = sum(dist.values())
        dist_str = ", ".join([f"{k}:{v}" for k, v in dist.items()])
        print(f"{indent}{prefix} → {node.prediction} (n={node.samples}, [{dist_str}])")
    else:
        if feature_names:
            feat_name = feature_names[node.feature_index]
        else:
            feat_name = f"Feature_{node.feature_index}"
        
        print(f"{indent}{prefix} [{feat_name}] (n={node.samples})")
        
        if node.threshold is not None:
            # Numeric split
            print_tree(node.left, depth+1, f"  ≤ {node.threshold:.2f}", feature_names)
            print_tree(node.right, depth+1, f"  > {node.threshold:.2f}", feature_names)
        else:
            # Categorical split
            for value, child in sorted(node.children.items()):
                print_tree(child, depth+1, f"  = {value}", feature_names)

print_tree(model.root, feature_names=feature_names)

# Test tahminleri
print("\n" + "=" * 60)
print("TAHMİNLER")
print("=" * 60)

predictions = model.predict(X)
for i, (true_label, pred_label) in enumerate(zip(y, predictions), 1):
    status = "✓" if true_label == pred_label else "✗"
    print(f"Örnek {i}: Gerçek={true_label}, Tahmin={pred_label} {status}")

# Accuracy
correct = sum(1 for t, p in zip(y, predictions) if t == p)
accuracy = correct / len(y) * 100
print(f"\nDoğruluk: {accuracy:.1f}% ({correct}/{len(y)})")

print("\n" + "=" * 60)
print("BEKLENTİ (Görüntüdeki Ağaç):")
print("=" * 60)
print("""
ROOT [NITELIK1]
  = a [NITELIK2]
    ≤ ek (bir eşik) → Sınıf1
    > ek → Sınıf2
  = b [NITELIK1+b]
    → Sınıf1
  = c [NITELIK3]
    = yanlış → Sınıf1
    = doğru → Sınıf2

NOT: Görüntüde tam detay yok, yaklaşık yapı bu şekilde
""")

# Ağaç derinliğini kontrol et
def get_depth(node):
    if node.is_leaf:
        return 0
    if node.threshold is not None:
        return 1 + max(get_depth(node.left), get_depth(node.right))
    else:
        return 1 + max(get_depth(child) for child in node.children.values())

depth = get_depth(model.root)
print(f"\nAğaç Derinliği: {depth}")
