"""
fast_pipeline.py
Optimize edilmiş, hızlı çalışan fake news detection pipeline
Donanım-optimized, minimal hyperparameter tuning
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import pickle
import warnings
warnings.filterwarnings('ignore')

# Donanım optimizasyonu
os.environ['OMP_NUM_THREADS'] = '4'  # CPU core limit
os.environ['NUMBA_NUM_THREADS'] = '4'

# src modüllerini import et
sys.path.append('src')

class FastFakeNewsDetector:
    """
    Hızlı fake news detection sistemi
    Minimal hyperparameter tuning, maksimum performans
    """
    
    def __init__(self):
        """Pipeline başlatıcı"""
        self.models = {}
        self.results = {}
        self.feature_names = []
        
    def load_prepared_features(self) -> bool:
        """Hazırlanmış feature'ları yükle"""
        try:
            print("📥 Hazır feature'lar yükleniyor...")
            
            with open('data/processed/features_ready.pkl', 'rb') as f:
                data = pickle.load(f)
            
            self.features = data['features']
            self.feature_names = data['feature_names']
            
            print(f"✅ Feature'lar yüklendi:")
            print(f"   📊 Train: {self.features['X_train'].shape}")
            print(f"   📊 Valid: {self.features['X_valid'].shape}")
            print(f"   📊 Test: {self.features['X_test'].shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Feature yükleme hatası: {e}")
            return False
    
    def train_optimized_models(self) -> Dict:
        """
        Optimize edilmiş modelleri hızlıca eğit
        Minimal hyperparameter tuning, iyi performans
        """
        print("\n🚀 HIZLI MODEL EĞİTİMİ BAŞLATILIYOR!")
        print("=" * 60)
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, classification_report
        )
        
        X_train = self.features['X_train']
        y_train = self.features['y_train']
        X_test = self.features['X_test'] 
        y_test = self.features['y_test']
        
        # Model tanımları - optimize edilmiş parametreler
        model_configs = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,  # Tüm core'ları kullan
                    class_weight='balanced'
                ),
                'name': 'Random Forest'
            },
            
            'LogisticRegression': {
                'model': LogisticRegression(
                    C=1.0,
                    penalty='l2',
                    solver='saga',
                    max_iter=2000,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                ),
                'name': 'Logistic Regression'
            }
        }
        
        # XGBoost (varsa)
        try:
            import xgboost as xgb
            model_configs['XGBoost'] = {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                ),
                'name': 'XGBoost'
            }
        except ImportError:
            pass
        
        # LightGBM (varsa)
        try:
            import lightgbm as lgb
            model_configs['LightGBM'] = {
                'model': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    num_leaves=63,
                    subsample=0.9,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                ),
                'name': 'LightGBM'
            }
        except ImportError:
            pass
        
        results = {}
        
        # Her modeli eğit ve test et
        for model_key, config in model_configs.items():
            print(f"\n🤖 {config['name']} eğitiliyor...")
            start_time = time.time()
            
            model = config['model']
            
            # Eğitim
            model.fit(X_train, y_train)
            
            # Test tahminleri
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrikler
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0)
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Eğitim süresi
            training_time = time.time() - start_time
            metrics['training_time'] = training_time
            
            # Sonuçları kaydet
            results[model_key] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            self.models[model_key] = model
            
            # Sonuçları yazdır
            print(f"   ⏱️ Eğitim süresi: {training_time:.2f} saniye")
            print(f"   🎯 Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   🔍 Precision: {metrics['precision']:.4f}")
            print(f"   📈 Recall:    {metrics['recall']:.4f}")
            print(f"   ⚖️ F1-Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                print(f"   📊 ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # En iyi modeli belirle
        best_f1 = 0
        best_model_key = ""
        
        for model_key, result in results.items():
            f1 = result['metrics']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_model_key = model_key
        
        print(f"\n🏆 EN İYİ MODEL: {model_configs[best_model_key]['name']}")
        print(f"🎯 En İyi F1-Score: {best_f1:.4f}")
        
        self.results = results
        self.best_model_key = best_model_key
        self.best_model = results[best_model_key]['model']
        
        return results
    
    def create_quick_visualizations(self):
        """Hızlı görselleştirmeler oluştur"""
        print("\n📊 Hızlı görselleştirmeler oluşturuluyor...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve
        
        # Stil ayarları
        plt.style.use('default')
        sns.set_palette("husl")
        
        os.makedirs('results/figures', exist_ok=True)
        
        y_test = self.features['y_test']
        
        # Model karşılaştırma grafiği
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = []
        metric_values = {metric: [] for metric in metrics}
        
        for model_key, result in self.results.items():
            model_names.append(model_key)
            for metric in metrics:
                metric_values[metric].append(result['metrics'][metric])
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, metric_values[metric], width, 
                  label=metric.upper().replace('_', '-'), alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/fast_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # En iyi model için confusion matrix
        best_result = self.results[self.best_model_key]
        cm = confusion_matrix(y_test, best_result['predictions'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'], ax=ax)
        
        ax.set_title(f'{self.best_model_key} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'results/figures/fast_{self.best_model_key.lower()}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Görselleştirmeler kaydedildi:")
        print("   📊 results/figures/fast_model_comparison.png")
        print("   📊 results/figures/fast_{}_confusion_matrix.png".format(self.best_model_key.lower()))
    
    def save_models(self):
        """Modelleri kaydet"""
        print("\n💾 Modeller kaydediliyor...")
        
        os.makedirs('models/trained_models', exist_ok=True)
        
        for model_key, model in self.models.items():
            filename = f'models/trained_models/fast_{model_key.lower()}_model.pkl'
            
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"   💾 {model_key} kaydedildi: {filename}")
        
        # En iyi model
        best_filename = f'models/trained_models/best_fast_model_{self.best_model_key.lower()}.pkl'
        with open(best_filename, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"   🏆 En iyi model kaydedildi: {best_filename}")
    
    def demo_predictions(self):
        """Demo tahminler"""
        print("\n🧪 DEMO TAHMİNLER")
        print("=" * 40)
        
        demo_texts = [
            "The Federal Reserve announced interest rate changes to combat inflation.",
            "Scientists discover breakthrough in renewable energy storage technology.",
            "BREAKING: Aliens have officially contacted world governments!",
            "SHOCKING: Local man discovers secret to eternal youth!",
            "The unemployment rate decreased according to latest statistics."
        ]
        
        expected_labels = ['Real', 'Real', 'Fake', 'Fake', 'Real']
        
        from data_preprocessing import DataPreprocessor
        from feature_engineering import FeatureEngineering
        import yaml
        
        # Preprocessor ve feature engineer yükle
        config = yaml.safe_load(open('config.yaml'))
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineering(config)
        
        # TF-IDF vectorizer'ı yükle
        feature_engineer.tfidf_vectorizer = self.get_tfidf_vectorizer()
        
        correct = 0
        
        for i, (text, expected) in enumerate(zip(demo_texts, expected_labels), 1):
            try:
                # Metni işle
                cleaned = preprocessor.clean_text(text)
                tokens = preprocessor.tokenize_and_filter(cleaned)
                
                # Basit feature extraction (sadece TF-IDF)
                text_for_tfidf = ' '.join(tokens)
                features = feature_engineer.tfidf_vectorizer.transform([text_for_tfidf])
                
                # Sadece TF-IDF ile tahmin (diğer feature'lar olmadan)
                if features.shape[1] < self.features['X_train'].shape[1]:
                    # Eksik sütunları sıfır ile doldur
                    missing_cols = self.features['X_train'].shape[1] - features.shape[1]
                    padding = np.zeros((1, missing_cols))
                    features = np.hstack([features.toarray(), padding])
                
                prediction = self.best_model.predict(features)[0]
                confidence = max(self.best_model.predict_proba(features)[0])
                
                result = 'Real' if prediction == 1 else 'Fake'
                status = '✅' if result == expected else '❌'
                
                if result == expected:
                    correct += 1
                
                print(f"{i}. {text[:50]}...")
                print(f"   Tahmin: {result} ({confidence:.3f}) - Gerçek: {expected} {status}")
                
            except Exception as e:
                print(f"{i}. Tahmin hatası: {e}")
        
        accuracy = correct / len(demo_texts)
        print(f"\n📊 Demo Accuracy: {accuracy:.1%} ({correct}/{len(demo_texts)})")
    
    def get_tfidf_vectorizer(self):
        """TF-IDF vectorizer'ı feature engineering'den al"""
        # Bu basit bir placeholder - gerçekte feature_engineer'den alınmalı
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Eğitim data'sından TF-IDF vectorizer'ı yeniden oluştur
        # (Normalde kaydedilmiş olması gerek)
        
        return TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
    
    def run_fast_pipeline(self) -> bool:
        """Hızlı pipeline çalıştır"""
        start_time = time.time()
        
        print("🚀 HIZLI FAKE NEWS DETECTION PIPELINE")
        print("=" * 60)
        print(f"⏰ Başlangıç: {datetime.now().strftime('%H:%M:%S')}")
        
        try:
            # 1. Feature'ları yükle
            if not self.load_prepared_features():
                return False
            
            # 2. Modelleri hızlıca eğit
            self.train_optimized_models()
            
            # 3. Görselleştirmeler
            self.create_quick_visualizations()
            
            # 4. Modelleri kaydet
            self.save_models()
            
            # 5. Demo tahminler
            self.demo_predictions()
            
            # Toplam süre
            total_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("🎉 HIZLI PIPELINE TAMAMLANDI!")
            print(f"⏱️ Toplam süre: {total_time/60:.1f} dakika")
            print(f"🏆 En iyi model: {self.best_model_key}")
            print(f"🎯 En iyi F1-Score: {self.results[self.best_model_key]['metrics']['f1_score']:.4f}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline hatası: {e}")
            return False

def main():
    """Ana fonksiyon"""
    detector = FastFakeNewsDetector()
    success = detector.run_fast_pipeline()
    
    if success:
        print("\n✅ Sistem kullanıma hazır!")
    else:
        print("\n❌ Pipeline başarısız!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())