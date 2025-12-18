"""
ID3 Karar Ağacı Test Scripti
id3.csv üzerinde algoritma testi
"""
import pandas as pd
import sys
sys.path.insert(0, '/Users/ilker/Documents/mltools')

from algorithms.decision_tree import DecisionTree

# CSV'yi yükle
df = pd.read_csv('sample_data/id3.csv')

print("=" * 60)
print("ID3 KARAR AĞACI TESTİ")
print("=" * 60)
print("\nVeri:")
print(df)

# Özellikler ve etiketler
X = df.iloc[:, 1:-1].values.tolist()  # borc, gelir, statu
y = df.iloc[:, -1].values.tolist()     # risk

feature_names = df.columns[1:-1].tolist()
print(f"\nÖzellikler: {feature_names}")
print(f"Hedef: {df.columns[-1]}")

# ID3 modeli oluştur
model = DecisionTree(criterion='entropy', max_depth=None)
model.fit(X, y)

print("\n" + "=" * 60)
print("OLUŞAN KARAR AĞACI")
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
            print_tree(node.left, depth+1, f"  ≤ {node.threshold}", feature_names)
            print_tree(node.right, depth+1, f"  > {node.threshold}", feature_names)
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
    print(f"Müşteri {i}: Gerçek={true_label}, Tahmin={pred_label} {status}")

# Accuracy
correct = sum(1 for t, p in zip(y, predictions) if t == p)
accuracy = correct / len(y) * 100
print(f"\nDoğruluk: {accuracy:.1f}% ({correct}/{len(y)})")

print("\n" + "=" * 60)
print("BEKLENTİ (Görüntüdeki Ağaç):")
print("=" * 60)
print("""
ROOT [BORÇ]
  = DÜŞÜK [GELİR]
    = DÜŞÜK [STATÜ]
      = İŞVEREN → KÖTÜ (müşteri 5,9)
      = ÜCRETLİ → İYİ (müşteri 4,8)
    = YÜKSEK → İYİ (müşteri 6,7,10)
  = YÜKSEK → KÖTÜ (müşteri 1,2,3)
""")
