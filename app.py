"""
Veri Madenciliği Eğitim Platformu - Flask Application

Bu uygulama, veri madenciliği algoritmalarını interaktif olarak
öğrenmek için tasarlanmış bir web platformudur.

Tüm algoritmalar Vanilla Python ile yazılmıştır (sklearn kullanılmamıştır).
"""

import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np

# Vanilla Python algoritmaları
from algorithms import KNN, DecisionTree, KMeans, Apriori
from algorithms.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, train_test_split,
    silhouette_score, inertia
)

app = Flask(__name__)
app.secret_key = 'dev-secret-key-change-in-production'

# Konfigürasyon
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Upload klasörünü oluştur
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Dosya uzantısı kontrolü."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =============================================================================
# ANA SAYFALAR
# =============================================================================

@app.route('/')
def index():
    """Ana sayfa."""
    return render_template('index.html')


@app.route('/knn')
def knn_page():
    """KNN algoritması sayfası."""
    return render_template('knn.html')


@app.route('/decision-tree')
def decision_tree_page():
    """Decision Tree algoritması sayfası."""
    return render_template('decision_tree.html')


@app.route('/kmeans')
def kmeans_page():
    """K-Means algoritması sayfası."""
    return render_template('kmeans.html')


@app.route('/apriori')
def apriori_page():
    """Apriori algoritması sayfası."""
    return render_template('apriori.html')


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/knn', methods=['POST'])
def run_knn():
    """
    KNN algoritmasını çalıştır.
    
    Beklenen form data:
    - file: CSV dosyası
    - k: Komşu sayısı (varsayılan: 3)
    - test_size: Test seti oranı (varsayılan: 0.2)
    """
    try:
        # Dosya kontrolü
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Dosya yüklenmedi'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Dosya seçilmedi'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Sadece CSV dosyaları kabul edilir'})
        
        # Parametreleri al
        k = int(request.form.get('k', 3))
        test_size = float(request.form.get('test_size', 0.2))
        
        # CSV'yi oku
        df = pd.read_csv(file)
        
        # İlk sütun sayısal ID ise atla
        first_col = df.columns[0].lower()
        if first_col in ['id', 'musteri', 'customer', 'index', 'no']:
            X = df.iloc[:, 1:-1].values.tolist()
        else:
            X = df.iloc[:, :-1].values.tolist()
        
        y = df.iloc[:, -1].values.tolist()
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # KNN modeli
        model = KNN(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrikler
        cm, labels = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return jsonify({
            'success': True,
            'algorithm': 'KNN',
            'params': {'k': k, 'test_size': test_size},
            'metrics': {
                'accuracy': round(report['accuracy'], 4),
                'precision': round(report['macro_avg']['precision'], 4),
                'recall': round(report['macro_avg']['recall'], 4),
                'f1_score': round(report['macro_avg']['f1_score'], 4)
            },
            'confusion_matrix': cm,
            'labels': labels,
            'class_report': report['classes'],
            'sample_count': {
                'train': len(y_train),
                'test': len(y_test)
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/decision-tree', methods=['POST'])
def run_decision_tree():
    """
    Decision Tree algoritmasını çalıştır - Eğitim Modu.
    
    Beklenen form data:
    - file: CSV dosyası
    - criterion: 'entropy' (ID3), 'gain_ratio' (C4.5), 'gini' (CART), 'twoing'
    
    Not: Eğitim amaçlı olduğu için tüm veri kullanılır (test/train split yok)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Dosya yüklenmedi'})
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Geçersiz dosya'})
        
        # Parametreler
        criterion = request.form.get('criterion', 'entropy')
        
        # CSV'yi oku
        df = pd.read_csv(file)
        
        # İlk sütun sayısal ID ise (musteri, id, vb.) atla
        first_col = df.columns[0].lower()
        if first_col in ['id', 'musteri', 'customer', 'index', 'no']:
            X = df.iloc[:, 1:-1].values.tolist()  # İlk sütunu atla
            feature_names = df.columns[1:-1].tolist()
        else:
            X = df.iloc[:, :-1].values.tolist()
            feature_names = df.columns[:-1].tolist()
        
        y = df.iloc[:, -1].values.tolist()
        
        # Eğitim Modu: Tüm veriyi kullan (test/train split yok)
        model = DecisionTree(criterion=criterion, max_depth=None)
        model.fit(X, y)
        
        # Ağaç yapısı (D3.js için)
        tree_structure = model.get_tree_for_d3()
        
        # Algoritma adı
        algo_names = {
            'entropy': 'ID3',
            'gain_ratio': 'C4.5',
            'gini': 'CART (Gini)',
            'twoing': 'CART (Twoing)'
        }
        
        # Ağaç bilgileri hesapla
        def get_tree_depth(node):
            """Ağaç derinliğini hesapla"""
            if node.is_leaf:
                return 0
            if node.threshold is not None:
                return 1 + max(get_tree_depth(node.left), get_tree_depth(node.right))
            else:
                return 1 + max(get_tree_depth(child) for child in node.children.values())
        
        def count_leaves(node):
            """Yaprak düğüm sayısını hesapla"""
            if node.is_leaf:
                return 1
            if node.threshold is not None:
                return count_leaves(node.left) + count_leaves(node.right)
            else:
                return sum(count_leaves(child) for child in node.children.values())
        
        tree_depth = get_tree_depth(model.root)
        num_leaves = count_leaves(model.root)
        
        return jsonify({
            'success': True,
            'algorithm': f'Decision Tree ({algo_names.get(criterion, criterion)})',
            'params': {
                'criterion': criterion,
                'mode': 'Eğitim Modu (Tüm Veri)'
            },
            'tree_info': {
                'depth': tree_depth,
                'num_leaves': num_leaves,
                'num_samples': len(y),
                'num_features': len(feature_names)
            },
            'feature_importances': model.feature_importances_,
            'feature_names': feature_names,
            'tree_structure': tree_structure
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/kmeans', methods=['POST'])
def run_kmeans():
    """
    K-Means algoritmasını çalıştır.
    
    Beklenen form data:
    - file: CSV dosyası (tüm sütunlar numerik)
    - n_clusters: Küme sayısı
    - max_iter: Maksimum iterasyon
    - init: Başlatma yöntemi ('random' veya 'kmeans++')
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Dosya yüklenmedi'})
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Geçersiz dosya'})
        
        # Parametreler
        n_clusters = int(request.form.get('n_clusters', 3))
        max_iter = int(request.form.get('max_iter', 300))
        init = request.form.get('init', 'kmeans++')
        
        # CSV'yi oku
        df = pd.read_csv(file)
        X = df.values.tolist()
        
        # K-Means modeli
        model = KMeans(n_clusters=n_clusters, max_iter=max_iter, init=init, random_state=42)
        model.fit(X)
        
        # Metrikler
        sil_score = silhouette_score(X, model.labels_) if n_clusters > 1 else 0
        
        # Görselleştirme için veri hazırla (ilk 2 boyut)
        scatter_data = []
        for i, point in enumerate(X):
            scatter_data.append({
                'x': point[0] if len(point) > 0 else 0,
                'y': point[1] if len(point) > 1 else 0,
                'cluster': model.labels_[i]
            })
        
        # Centroid'ler
        centroid_data = []
        for i, center in enumerate(model.cluster_centers_):
            centroid_data.append({
                'x': center[0] if len(center) > 0 else 0,
                'y': center[1] if len(center) > 1 else 0,
                'cluster': i
            })
        
        return jsonify({
            'success': True,
            'algorithm': 'K-Means',
            'params': {
                'n_clusters': n_clusters,
                'max_iter': max_iter,
                'init': init
            },
            'metrics': {
                'inertia': round(model.inertia_, 4),
                'silhouette_score': round(sil_score, 4),
                'n_iterations': model.n_iter_
            },
            'cluster_sizes': model.get_cluster_sizes(),
            'cluster_centers': model.cluster_centers_,
            'labels': model.labels_,
            'scatter_data': scatter_data,
            'centroid_data': centroid_data,
            'sample_count': len(X)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/apriori', methods=['POST'])
def run_apriori():
    """
    Apriori algoritmasını çalıştır.
    
    Beklenen form data:
    - file: CSV dosyası (transaction-based)
    - min_support: Minimum destek
    - min_confidence: Minimum güven
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Dosya yüklenmedi'})
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Geçersiz dosya'})
        
        # Parametreler
        min_support = float(request.form.get('min_support', 0.1))
        min_confidence = float(request.form.get('min_confidence', 0.5))
        
        # CSV'yi oku - her satır bir transaction
        df = pd.read_csv(file, header=None)
        
        # Transaction listesi oluştur
        transactions = []
        for _, row in df.iterrows():
            # NaN değerleri filtrele ve string'e çevir
            items = [str(item).strip() for item in row.dropna().values if str(item).strip()]
            if items:
                transactions.append(items)
        
        if not transactions:
            return jsonify({'success': False, 'error': 'Geçerli transaction bulunamadı'})
        
        # Apriori modeli
        model = Apriori(min_support=min_support, min_confidence=min_confidence)
        model.fit(transactions)
        
        # Sonuçlar
        frequent_itemsets = model.get_frequent_itemsets()
        rules = model.get_rules()
        summary = model.get_summary()
        
        # JSON serializable hale getir
        for itemset in frequent_itemsets:
            itemset['itemset'] = list(itemset['itemset'])
        
        for rule in rules:
            rule['antecedent'] = list(rule['antecedent'])
            rule['consequent'] = list(rule['consequent'])
        
        return jsonify({
            'success': True,
            'algorithm': 'Apriori',
            'params': {
                'min_support': min_support,
                'min_confidence': min_confidence
            },
            'summary': summary,
            'frequent_itemsets': frequent_itemsets[:50],  # İlk 50
            'rules': rules[:50],  # İlk 50
            'n_transactions': len(transactions)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# =============================================================================
# HATA HANDLER'LARI
# =============================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'Dosya boyutu çok büyük (max 16MB)'}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Sunucu hatası'}), 500


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
