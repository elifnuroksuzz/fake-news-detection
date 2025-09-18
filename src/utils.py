"""
utils.py
Yardımcı fonksiyonlar ve genel kullanım araçları
"""

import os
import json
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class Logger:
    """Basit logger sınıfı"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        
    def log(self, message: str, level: str = "INFO"):
        """Log mesajı yaz"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        
        print(log_message)
        
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')

class FileManager:
    """Dosya yönetimi için yardımcı sınıf"""
    
    @staticmethod
    def create_directories(paths: List[str]):
        """Dizinleri oluştur"""
        for path in paths:
            os.makedirs(path, exist_ok=True)
            print(f"📁 Dizin oluşturuldu: {path}")
    
    @staticmethod
    def save_json(data: Dict, filepath: str):
        """JSON dosyası kaydet"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON kaydedildi: {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """JSON dosyası yükle"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str):
        """Pickle dosyası kaydet"""
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"💾 Pickle kaydedildi: {filepath}")
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """Pickle dosyası yükle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class MetricsCalculator:
    """Değerlendirme metriklerini hesapla"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Sınıflandırma metrikleri hesapla
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler
            y_pred_proba: Tahmin olasılıkları
            
        Returns:
            Dict: Metrikler sözlüğü
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix,
            classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict, model_name: str = "Model"):
        """Metrikleri yazdır"""
        print(f"\n📊 {model_name} Performans Metrikleri:")
        print(f"   🎯 Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   🔍 Precision: {metrics['precision']:.4f}")
        print(f"   📈 Recall:    {metrics['recall']:.4f}")
        print(f"   ⚖️ F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"   📊 ROC-AUC:   {metrics['roc_auc']:.4f}")

class Visualizer:
    """Görselleştirme fonksiyonları"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'whitegrid'):
        """
        Görselleştirici başlat
        
        Args:
            figsize: Figure boyutu
            style: Seaborn stili
        """
        plt.style.use('default')
        sns.set_style(style)
        self.figsize = figsize
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], title: str = "Confusion Matrix", save_path: Optional[str] = None):
        """
        Confusion matrix çiz
        
        Args:
            cm: Confusion matrix
            labels: Sınıf etiketleri
            title: Grafik başlığı
            save_path: Kaydetme yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_score: float, 
                      title: str = "ROC Curve", save_path: Optional[str] = None):
        """
        ROC Curve çiz
        
        Args:
            fpr: False positive rate
            tpr: True positive rate
            auc_score: AUC skoru
            title: Grafik başlığı
            save_path: Kaydetme yolu
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ROC Curve kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history: Dict, metrics: List[str], 
                            title: str = "Training History", save_path: Optional[str] = None):
        """
        Eğitim geçmişi grafiği çiz
        
        Args:
            history: Eğitim geçmişi
            metrics: Gösterilecek metrikler
            title: Grafik başlığı
            save_path: Kaydetme yolu
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in history:
                axes[i].plot(history[metric], label=f'Train {metric}')
                if f'val_{metric}' in history:
                    axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
                
                axes[i].set_title(f'{metric.capitalize()}', fontsize=14)
                axes[i].set_xlabel('Epoch', fontsize=12)
                axes[i].set_ylabel(metric.capitalize(), fontsize=12)
                axes[i].legend()
                axes[i].grid(alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray, 
                               top_n: int = 20, title: str = "Feature Importance", save_path: Optional[str] = None):
        """
        Özellik önemliliği grafiği
        
        Args:
            feature_names: Özellik isimleri
            importance_scores: Önemlilik skorları
            top_n: Gösterilecek özellik sayısı
            title: Grafik başlığı
            save_path: Kaydetme yolu
        """
        # En önemli özellikleri seç
        indices = np.argsort(importance_scores)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.barh(range(len(indices)), importance_scores[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'{title} (Top {top_n})', fontsize=16, fontweight='bold')
        
        # En önemli özellikleri vurgula
        for i, (idx, score) in enumerate(zip(indices, importance_scores[indices])):
            ax.text(score + score*0.01, i, f'{score:.4f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Feature importance kaydedildi: {save_path}")
        
        plt.show()

class ConfigManager:
    """Konfigürasyon yönetimi"""
    
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict:
        """Konfigürasyon dosyasını yükle"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"⚠️ Konfigürasyon dosyası bulunamadı: {config_path}")
            return {}
    
    @staticmethod
    def save_config(config: Dict, config_path: str = "config.yaml"):
        """Konfigürasyonu kaydet"""
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        print(f"💾 Konfigürasyon kaydedildi: {config_path}")

class DataSplitter:
    """Veri bölme işlemleri"""
    
    @staticmethod
    def stratified_train_test_split(X: pd.DataFrame, y: pd.Series, 
                                  test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Stratified train-test split
        
        Args:
            X: Özellikler
            y: Etiketler
            test_size: Test boyutu
            random_state: Random seed
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        return train_test_split(X, y, test_size=test_size, 
                              stratify=y, random_state=random_state)
    
    @staticmethod
    def create_k_fold_splits(X: pd.DataFrame, y: pd.Series, 
                           n_splits: int = 10, random_state: int = 42):
        """
        K-Fold cross validation splits oluştur
        
        Args:
            X: Özellikler
            y: Etiketler
            n_splits: Fold sayısı
            random_state: Random seed
            
        Returns:
            StratifiedKFold: Cross validation objesi
        """
        from sklearn.model_selection import StratifiedKFold
        
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

class ModelPersistence:
    """Model kaydetme ve yükleme"""
    
    @staticmethod
    def save_model(model, model_path: str, metadata: Optional[Dict] = None):
        """
        Model kaydet
        
        Args:
            model: Kaydedilecek model
            model_path: Kayıt yolu
            metadata: Model metadata'sı
        """
        import joblib
        
        # Model kaydet
        joblib.dump(model, model_path)
        print(f"💾 Model kaydedildi: {model_path}")
        
        # Metadata kaydet
        if metadata:
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            FileManager.save_json(metadata, metadata_path)
    
    @staticmethod
    def load_model(model_path: str):
        """
        Model yükle
        
        Args:
            model_path: Model yolu
            
        Returns:
            Loaded model
        """
        import joblib
        
        model = joblib.load(model_path)
        print(f"📥 Model yüklendi: {model_path}")
        
        return model
    
    @staticmethod
    def load_model_with_metadata(model_path: str) -> Tuple:
        """
        Model ve metadata'yı yükle
        
        Args:
            model_path: Model yolu
            
        Returns:
            Tuple: (model, metadata)
        """
        import joblib
        
        # Model yükle
        model = joblib.load(model_path)
        
        # Metadata yükle
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        metadata = None
        
        if os.path.exists(metadata_path):
            metadata = FileManager.load_json(metadata_path)
        
        print(f"📥 Model ve metadata yüklendi: {model_path}")
        
        return model, metadata

class PerformanceTracker:
    """Performans takibi"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, model_name: str, fold: int, metrics: Dict):
        """
        Sonuç ekle
        
        Args:
            model_name: Model adı
            fold: Fold numarası
            metrics: Metrikler
        """
        if model_name not in self.results:
            self.results[model_name] = []
        
        result = {'fold': fold, **metrics}
        self.results[model_name].append(result)
    
    def get_average_metrics(self, model_name: str) -> Dict:
        """
        Ortalama metrikler
        
        Args:
            model_name: Model adı
            
        Returns:
            Dict: Ortalama metrikler
        """
        if model_name not in self.results:
            return {}
        
        results = self.results[model_name]
        
        # Metrik isimlerini al (fold dışında)
        metric_names = [key for key in results[0].keys() if key != 'fold']
        
        avg_metrics = {}
        for metric in metric_names:
            if metric not in ['confusion_matrix', 'classification_report']:
                values = [result[metric] for result in results]
                avg_metrics[f'{metric}_mean'] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
        
        return avg_metrics
    
    def get_best_model(self, metric: str = 'f1_score') -> str:
        """
        En iyi modeli bul
        
        Args:
            metric: Karşılaştırma metriği
            
        Returns:
            str: En iyi model adı
        """
        best_model = None
        best_score = -1
        
        for model_name in self.results.keys():
            avg_metrics = self.get_average_metrics(model_name)
            metric_key = f'{metric}_mean'
            
            if metric_key in avg_metrics:
                score = avg_metrics[metric_key]
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model
    
    def print_summary(self):
        """Özet rapor yazdır"""
        print("\n📊 MODEL PERFORMANS ÖZETİ")
        print("=" * 50)
        
        for model_name in self.results.keys():
            print(f"\n🤖 {model_name}")
            avg_metrics = self.get_average_metrics(model_name)
            
            for metric, value in avg_metrics.items():
                if '_mean' in metric:
                    base_metric = metric.replace('_mean', '')
                    std_metric = f'{base_metric}_std'
                    mean_val = avg_metrics[metric]
                    std_val = avg_metrics.get(std_metric, 0)
                    print(f"   {base_metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
        best_model = self.get_best_model('f1_score')
        if best_model:
            print(f"\n🏆 EN İYİ MODEL: {best_model}")
    
    def save_results(self, filepath: str):
        """Sonuçları kaydet"""
        FileManager.save_json(self.results, filepath)
    
    def load_results(self, filepath: str):
        """Sonuçları yükle"""
        self.results = FileManager.load_json(filepath)

def setup_reproducibility(seed: int = 42):
    """
    Reproducibility için random seed ayarla
    
    Args:
        seed: Random seed değeri
    """
    import random
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # Scikit-learn içindeki random durumlar için
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🔒 Reproducibility ayarlandı (seed: {seed})")

def print_system_info():
    """Sistem bilgilerini yazdır"""
    import sys
    import platform
    
    print("💻 SİSTEM BİLGİLERİ")
    print("=" * 30)
    print(f"🐍 Python: {sys.version}")
    print(f"💻 Platform: {platform.platform()}")
    print(f"🖥️ Processor: {platform.processor()}")
    print(f"📁 Current Dir: {os.getcwd()}")
    
    # Paket versiyonları
    packages = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn']
    print("\n📦 PAKET VERSİYONLARI")
    print("=" * 30)
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"📚 {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")

def create_project_structure():
    """Proje klasör yapısını oluştur"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'models/trained_models',
        'models/model_configs',
        'results/figures',
        'results/reports',
        'results/metrics',
        'notebooks'
    ]
    
    FileManager.create_directories(directories)
    print("🎉 Proje yapısı oluşturuldu!")

# Test fonksiyonu
def test_utils():
    """Utils modülünü test et"""
    print("🧪 Utils modülü test ediliyor...")
    
    # System info
    print_system_info()
    print()
    
    # Reproducibility
    setup_reproducibility(42)
    print()
    
    # Performance tracker test
    tracker = PerformanceTracker()
    tracker.add_result('TestModel', 1, {'accuracy': 0.85, 'f1_score': 0.83})
    tracker.add_result('TestModel', 2, {'accuracy': 0.87, 'f1_score': 0.85})
    
    avg_metrics = tracker.get_average_metrics('TestModel')
    print("📊 Test metrics:", avg_metrics)
    
    print("✅ Utils modülü test tamamlandı!")

if __name__ == "__main__":
    test_utils()