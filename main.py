"""
main.py
Fake News Detection - Ana Pipeline
Tüm süreçleri koordine eden ana dosya
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# src modüllerini import et
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineering
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from utils import (
    ConfigManager, setup_reproducibility, print_system_info,
    Logger, FileManager, ModelPersistence
)

class FakeNewsDetectionPipeline:
    """
    Tam fake news detection pipeline'ı
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Pipeline başlatıcısı
        
        Args:
            config_path: Konfigürasyon dosyası yolu
        """
        self.config = ConfigManager.load_config(config_path)
        
        # Logger başlat
        log_file = self.config.get('output', {}).get('log_file', 'fake_news_detection.log')
        self.logger = Logger(log_file)
        
        # Reproducibility ayarla
        seed = self.config.get('reproducibility', {}).get('random_seed', 42)
        setup_reproducibility(seed)
        
        # Modülleri başlat
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineering(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        
        # Sonuç kayıt yolları
        self.model_save_path = self.config.get('models', {}).get('model_save_path', 'models/trained_models/')
        
        self.logger.log("🚀 FakeNewsDetectionPipeline başlatıldı!")
        
    def run_full_pipeline(self, force_preprocessing: bool = False,
                         force_feature_engineering: bool = False) -> bool:
        """
        Tam pipeline'ı çalıştır
        
        Args:
            force_preprocessing: Veri ön işlemeyi zorla
            force_feature_engineering: Feature engineering'i zorla
            
        Returns:
            bool: Pipeline başarılı mı
        """
        self.logger.log("🚀 TAM FAKE NEWS DETECTION PIPELINE BAŞLATILIYOR!")
        print("=" * 80)
        
        try:
            # 1. Veri Ön İşleme
            success = self._run_preprocessing_stage(force_preprocessing)
            if not success:
                return False
                
            # 2. Feature Engineering
            success = self._run_feature_engineering_stage(force_feature_engineering)
            if not success:
                return False
                
            # 3. Model Training
            success = self._run_training_stage()
            if not success:
                return False
                
            # 4. Model Evaluation
            success = self._run_evaluation_stage()
            if not success:
                return False
                
            # 5. Final Report
            self._create_final_report()
            
            self.logger.log("🎉 TAM PIPELINE BAŞARIYLA TAMAMLANDI!")
            return True
            
        except Exception as e:
            self.logger.log(f"❌ Pipeline hatası: {str(e)}", "ERROR")
            return False
    
    def _run_preprocessing_stage(self, force: bool = False) -> bool:
        """
        Veri ön işleme aşaması
        
        Args:
            force: Zorla çalıştır
            
        Returns:
            bool: Başarılı mı
        """
        self.logger.log("📊 AŞAMA 1: VERİ ÖN İŞLEME")
        print("\n" + "="*50)
        print("📊 AŞAMA 1: VERİ ÖN İŞLEME")
        print("="*50)
        
        # İşlenmiş veri var mı kontrol et
        processed_path = self.config['data']['processed_data_path']
        train_file = os.path.join(processed_path, 'train_processed.pkl')
        
        if os.path.exists(train_file) and not force:
            self.logger.log("✅ İşlenmiş veri mevcut, ön işleme atlaniyor")
            print("✅ İşlenmiş veri mevcut, ön işleme atlanıyor")
            return True
        
        # Veri ön işlemeyi çalıştır
        success = self.preprocessor.run_full_preprocessing()
        
        if success:
            self.logger.log("✅ Veri ön işleme tamamlandı")
            return True
        else:
            self.logger.log("❌ Veri ön işleme başarısız", "ERROR")
            return False
    
    def _run_feature_engineering_stage(self, force: bool = False) -> bool:
        """
        Feature engineering aşaması
        
        Args:
            force: Zorla çalıştır
            
        Returns:
            bool: Başarılı mı
        """
        self.logger.log("🔧 AŞAMA 2: FEATURE ENGİNEERİNG")
        print("\n" + "="*50)
        print("🔧 AŞAMA 2: FEATURE ENGİNEERİNG")
        print("="*50)
        
        # Feature dosyası var mı kontrol et
        processed_path = self.config['data']['processed_data_path']
        features_file = os.path.join(processed_path, 'features_ready.pkl')
        
        if os.path.exists(features_file) and not force:
            self.logger.log("✅ Feature'lar mevcut, feature engineering atlanıyor")
            print("✅ Feature'lar mevcut, feature engineering atlanıyor")
            
            # Mevcut feature'ları yükle
            import pickle
            with open(features_file, 'rb') as f:
                feature_data = pickle.load(f)
            self.features = feature_data['features']
            self.feature_names = feature_data['feature_names']
            return True
        
        try:
            # İşlenmiş veriyi yükle
            processed_path = self.config['data']['processed_data_path']
            
            self.logger.log("📥 İşlenmiş veriler yükleniyor...")
            train_df = pd.read_pickle(os.path.join(processed_path, 'train_processed.pkl'))
            valid_df = pd.read_pickle(os.path.join(processed_path, 'valid_processed.pkl'))
            test_df = pd.read_pickle(os.path.join(processed_path, 'test_processed.pkl'))
            
            print(f"📊 Veri boyutları:")
            print(f"   Train: {train_df.shape}")
            print(f"   Valid: {valid_df.shape}")
            print(f"   Test: {test_df.shape}")
            
            # Feature engineering çalıştır
            self.logger.log("🔧 Feature engineering başlatılıyor...")
            self.features = self.feature_engineer.create_all_features(train_df, valid_df, test_df)
            
            # Feature isimlerini al
            self.feature_names = self.feature_engineer.get_feature_names()
            
            # Feature'ları kaydet
            feature_data = {
                'features': self.features,
                'feature_names': self.feature_names
            }
            
            FileManager.save_pickle(feature_data, features_file)
            
            self.logger.log("✅ Feature engineering tamamlandı")
            return True
            
        except Exception as e:
            self.logger.log(f"❌ Feature engineering hatası: {str(e)}", "ERROR")
            return False
    
    def _run_training_stage(self) -> bool:
        """
        Model eğitimi aşaması
        
        Returns:
            bool: Başarılı mı
        """
        self.logger.log("🤖 AŞAMA 3: MODEL EĞİTİMİ")
        print("\n" + "="*50)
        print("🤖 AŞAMA 3: MODEL EĞİTİMİ")
        print("="*50)
        
        try:
            # Training data hazırla
            X_train = self.features['X_train']
            y_train = self.features['y_train']
            
            self.logger.log(f"📊 Training veri boyutu: {X_train.shape}")
            
            # Tüm modelleri eğit
            self.training_results = self.model_trainer.train_all_models(X_train, y_train)
            
            if not self.training_results:
                self.logger.log("❌ Hiçbir model eğitilemedi", "ERROR")
                return False
            
            # En iyi modeli al
            self.best_model, self.best_model_name = self.model_trainer.get_best_model()
            
            # Modelleri kaydet
            self.model_trainer.save_models(self.model_save_path)
            
            self.logger.log("✅ Model eğitimi tamamlandı")
            self.logger.log(f"🏆 En iyi model: {self.best_model_name}")
            
            return True
            
        except Exception as e:
            self.logger.log(f"❌ Model eğitimi hatası: {str(e)}", "ERROR")
            return False
    
    def _run_evaluation_stage(self) -> bool:
        """
        Model değerlendirme aşaması
        
        Returns:
            bool: Başarılı mı
        """
        self.logger.log("📊 AŞAMA 4: MODEL DEĞERLENDİRME")
        print("\n" + "="*50)
        print("📊 AŞAMA 4: MODEL DEĞERLENDİRME")
        print("="*50)
        
        try:
            # Test data hazırla
            X_test = self.features['X_test']
            y_test = self.features['y_test']
            
            self.logger.log(f"📊 Test veri boyutu: {X_test.shape}")
            
            # Eğitilmiş modelleri al
            trained_models = self.model_trainer.best_models
            
            # Tüm modelleri değerlendir
            self.evaluation_results = self.model_evaluator.evaluate_multiple_models(
                trained_models, X_test, y_test, self.feature_names
            )
            
            # Değerlendirme raporunu kaydet
            self.model_evaluator.save_evaluation_report(
                self.evaluation_results, "final_evaluation_report"
            )
            
            self.logger.log("✅ Model değerlendirme tamamlandı")
            
            return True
            
        except Exception as e:
            self.logger.log(f"❌ Model değerlendirme hatası: {str(e)}", "ERROR")
            return False
    
    def _create_final_report(self):
        """
        Final rapor oluştur
        """
        self.logger.log("📄 FINAL RAPOR OLUŞTURULUYOR")
        print("\n" + "="*50)
        print("📄 FINAL RAPOR")
        print("="*50)
        
        # En iyi model bilgileri
        print(f"🏆 EN İYİ MODEL: {self.best_model_name}")
        
        if self.best_model_name in self.evaluation_results:
            best_metrics = self.evaluation_results[self.best_model_name]['metrics']
            
            print(f"\n📊 EN İYİ MODEL PERFORMANSI:")
            print(f"   🎯 Accuracy:  {best_metrics['accuracy']:.4f}")
            print(f"   🔍 Precision: {best_metrics['precision']:.4f}")
            print(f"   📈 Recall:    {best_metrics['recall']:.4f}")
            print(f"   ⚖️ F1-Score:  {best_metrics['f1_score']:.4f}")
            
            if 'roc_auc' in best_metrics:
                print(f"   📊 ROC-AUC:   {best_metrics['roc_auc']:.4f}")
        
        # Dosya konumları
        print(f"\n📁 OLUŞTURULAN DOSYALAR:")
        print(f"   🤖 Modeller: {self.model_save_path}")
        print(f"   📊 Grafikler: {self.config.get('output', {}).get('figures_path', 'results/figures/')}")
        print(f"   📄 Raporlar: {self.config.get('output', {}).get('reports_path', 'results/reports/')}")
        
        # Özet istatistikler
        total_models = len(self.training_results) if hasattr(self, 'training_results') else 0
        print(f"\n📈 ÖZET İSTATİSTİKLER:")
        print(f"   🤖 Eğitilen model sayısı: {total_models}")
        print(f"   📊 Toplam veri örneği: {self.features['X_train'].shape[0] + self.features['X_test'].shape[0]}")
        print(f"   🔢 Özellik sayısı: {self.features['X_train'].shape[1]}")
        
        self.logger.log("✅ Final rapor oluşturuldu")
        
        print("\n🎉 FAKE NEWS DETECTION PIPELINE TAMAMLANDI!")
        print("="*80)
    
    def predict(self, text: str) -> Dict:
        """
        Tek bir metin için tahmin yap
        
        Args:
            text: Tahmin yapılacak metin
            
        Returns:
            Dict: Tahmin sonucu
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("Pipeline henüz çalıştırılmadı!")
        
        # Metni işle
        processed_text = self.preprocessor.clean_text(text)
        tokens = self.preprocessor.tokenize_and_filter(processed_text)
        
        # Dummy DataFrame oluştur (feature engineering için)
        dummy_df = pd.DataFrame({
            'cleaned_statement': [processed_text],
            'tokens': [tokens],
            'subject': ['unknown'],
            'party_affiliation': ['unknown'],
            'state_info': ['unknown'],
            'barely_true_counts': [0],
            'false_counts': [0],
            'half_true_counts': [0],
            'mostly_true_counts': [0],
            'pants_on_fire_counts': [0]
        })
        
        # Feature'ları çıkar (sadece istatistiksel ve metadata)
        stat_features = self.feature_engineer.create_statistical_features(dummy_df)
        meta_features = self.feature_engineer.create_metadata_features(dummy_df)
        
        # TF-IDF features
        text_for_tfidf = ' '.join(tokens)
        tfidf_features = self.feature_engineer.tfidf_vectorizer.transform([text_for_tfidf])
        
        # Tüm feature'ları birleştir
        combined_features = self.feature_engineer.combine_features(
            tfidf_features.toarray(), stat_features, meta_features
        )
        
        # Ölçeklendir
        if combined_features.shape[1] > self.feature_engineer.max_features:
            tfidf_part = combined_features[:, :self.feature_engineer.max_features]
            other_part = combined_features[:, self.feature_engineer.max_features:]
            other_part_scaled = self.feature_engineer.scaler.transform(other_part)
            combined_features = np.hstack([tfidf_part, other_part_scaled])
        
        # Tahmin yap
        prediction = self.best_model.predict(combined_features)[0]
        
        # Probability (varsa)
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(combined_features)[0]
            fake_prob = probabilities[0]
            real_prob = probabilities[1]
        else:
            fake_prob = 1 - prediction
            real_prob = prediction
        
        result = {
            'prediction': 'Real' if prediction == 1 else 'Fake',
            'confidence': max(fake_prob, real_prob),
            'probabilities': {
                'fake': fake_prob,
                'real': real_prob
            },
            'processed_text': processed_text,
            'token_count': len(tokens)
        }
        
        return result

def main():
    """Ana fonksiyon"""
    print("🚀 FAKE NEWS DETECTION SYSTEM")
    print("=" * 80)
    
    # Sistem bilgilerini göster
    print_system_info()
    print()
    
    # Pipeline'ı başlat
    pipeline = FakeNewsDetectionPipeline("config.yaml")
    
    # Tam pipeline'ı çalıştır
    success = pipeline.run_full_pipeline()
    
    if success:
        print("\n✅ PIPELINE BAŞARIYLA TAMAMLANDI!")
        
        # Demo tahmin
        print("\n🧪 DEMO TAHMİN TESTİ")
        print("-" * 30)
        
        sample_texts = [
            "The president announced new economic policies yesterday.",
            "Scientists have discovered aliens living in underground cities!",
            "The stock market experienced significant volatility this week."
        ]
        
        for i, text in enumerate(sample_texts, 1):
            print(f"\n{i}. Metin: {text}")
            try:
                result = pipeline.predict(text)
                print(f"   📊 Tahmin: {result['prediction']}")
                print(f"   🎯 Güven: {result['confidence']:.3f}")
                print(f"   📈 Fake olasılık: {result['probabilities']['fake']:.3f}")
                print(f"   📈 Real olasılık: {result['probabilities']['real']:.3f}")
            except Exception as e:
                print(f"   ❌ Tahmin hatası: {str(e)}")
        
    else:
        print("\n❌ PIPELINE BAŞARISIZ!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)