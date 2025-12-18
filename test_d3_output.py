"""
D3.js Tree Output Test
"""
import pandas as pd
import json
import sys
sys.path.insert(0, '/Users/ilker/Documents/mltools')

from algorithms.decision_tree import DecisionTree

# CSV'yi yükle
df = pd.read_csv('sample_data/id3.csv')

# Özellikler ve etiketler
X = df.iloc[:, 1:-1].values.tolist()
y = df.iloc[:, -1].values.tolist()
feature_names = df.columns[1:-1].tolist()

# Model
model = DecisionTree(criterion='entropy', max_depth=None)
model.fit(X, y)

print("=" * 60)
print("D3.js TREE OUTPUT TEST")
print("=" * 60)

# D3 için tree structure
tree_d3 = model.get_tree_for_d3()

print("\nD3.js Tree Structure:")
print(json.dumps(tree_d3, indent=2, ensure_ascii=False))

print("\n" + "=" * 60)
print("TREE VALIDATION")
print("=" * 60)

def validate_tree(node, depth=0):
    indent = "  " * depth
    
    if node.get('is_leaf'):
        print(f"{indent}LEAF: {node.get('name')} (samples={node.get('samples')})")
    else:
        print(f"{indent}SPLIT: {node.get('name')} (samples={node.get('samples')})")
        if 'children' in node:
            print(f"{indent}  Children count: {len(node['children'])}")
            for i, child in enumerate(node['children']):
                print(f"{indent}  Child {i+1} edge_label: {child.get('edge_label')}")
                validate_tree(child, depth+1)

validate_tree(tree_d3)
