"""
feature_engineering.py
Gelişmiş özellik çıkarma modülü
TF-IDF, N-gram, ve ek özellikler
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    Profesyonel özellik çıkarma sınıfı
    TF-IDF, N-gram ve metin tabanlı özellikler
    """
    
    def __init__(self, config: Dict):
        """
        Feature engineering başlatıcı
        
        Args:
            config (Dict): Konfigürasyon sözlüğü
        """
        self.config = config
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
        # TF-IDF parametreleri
        feature_config = config.get('data', {}).get('feature_extraction', {})
        self.max_features = feature_config.get('max_features', 10000)
        self.ngram_range = tuple(feature_config.get('ngram_range', [1, 3]))
        self.min_df = feature_config.get('min_df', 2)
        self.max_df = feature_config.get('max_df', 0.95)
        
        print("🔧 FeatureEngineering başlatıldı!")
        print(f"   📊 Max Features: {self.max_features}")
        print(f"   📝 N-gram Range: {self.ngram_range}")
        print(f"   📈 Min DF: {self.min_df}, Max DF: {self.max_df}")
    
    def create_tfidf_features(self, train_texts: List[str], 
                            valid_texts: Optional[List[str]] = None,
                            test_texts: Optional[List[str]] = None) -> Tuple[np.ndarray, ...]:
        """
        TF-IDF özelliklerini oluştur
        
        Args:
            train_texts: Train metin listesi
            valid_texts: Validation metin listesi  
            test_texts: Test metin listesi
            
        Returns:
            Tuple: TF-IDF feature matrisleri
        """
        print("📊 TF-IDF özellikleri oluşturuluyor...")
        
        # TF-IDF vectorizer'ı oluştur
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True,
            sublinear_tf=True,  # Logaritmik TF scaling
            smooth_idf=True     # IDF smoothing
        )
        
        # Train verisi üzerinde fit et
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_texts)
        print(f"   ✅ Train TF-IDF: {X_train_tfidf.shape}")
        
        results = [X_train_tfidf.toarray()]
        
        # Validation transform
        if valid_texts is not None:
            X_valid_tfidf = self.tfidf_vectorizer.transform(valid_texts)
            results.append(X_valid_tfidf.toarray())
            print(f"   ✅ Valid TF-IDF: {X_valid_tfidf.shape}")
        
        # Test transform  
        if test_texts is not None:
            X_test_tfidf = self.tfidf_vectorizer.transform(test_texts)
            results.append(X_test_tfidf.toarray())
            print(f"   ✅ Test TF-IDF: {X_test_tfidf.shape}")
        
        print(f"📈 En önemli özellikler:")
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"   📝 Toplam özellik sayısı: {len(feature_names)}")
        print(f"   🔤 İlk 10 özellik: {feature_names[:10]}")
        
        return tuple(results)
    
    def create_statistical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        İstatistiksel metin özellikleri oluştur
        
        Args:
            df: DataFrame (cleaned_statement ve tokens sütunları içermeli)
            
        Returns:
            np.ndarray: İstatistiksel özellikler
        """
        print("📊 İstatistiksel özellikler oluşturuluyor...")
        
        features = []
        
        for _, row in df.iterrows():
            text = row['cleaned_statement']
            tokens = row['tokens'] if isinstance(row['tokens'], list) else []
            
            # Temel metin özellikleri
            char_count = len(text)
            word_count = len(tokens)
            sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
            
            # Ortalama değerler
            avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Özel karakter özellikleri
            exclamation_count = text.count('!')
            question_count = text.count('?')
            capital_count = sum(1 for c in text if c.isupper())
            capital_ratio = capital_count / len(text) if text else 0
            
            # Rakam özellikleri
            digit_count = sum(1 for c in text if c.isdigit())
            digit_ratio = digit_count / len(text) if text else 0
            
            # Benzersiz kelime oranı
            unique_word_ratio = len(set(tokens)) / len(tokens) if tokens else 0
            
            # Stopwords oranı (yaklaşık)
            common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            stopword_count = sum(1 for word in tokens if word.lower() in common_words)
            stopword_ratio = stopword_count / len(tokens) if tokens else 0
            
            feature_vector = [
                char_count,
                word_count,
                sentence_count,
                avg_word_length,
                avg_sentence_length,
                exclamation_count,
                question_count,
                capital_ratio,
                digit_ratio,
                unique_word_ratio,
                stopword_ratio
            ]
            
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        print(f"   ✅ İstatistiksel özellikler: {features_array.shape}")
        print(f"   📊 Özellik isimleri: char_count, word_count, sentence_count, avg_word_length,")
        print(f"       avg_sentence_length, exclamation_count, question_count, capital_ratio,")
        print(f"       digit_ratio, unique_word_ratio, stopword_ratio")
        
        return features_array
    
    def create_metadata_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Metadata tabanlı özellikler
        
        Args:
            df: DataFrame (LIAR dataset sütunları ile)
            
        Returns:
            np.ndarray: Metadata özellikleri
        """
        print("📊 Metadata özellikleri oluşturuluyor...")
        
        features = []
        
        # Label encoder'lar
        subject_encoder = LabelEncoder()
        party_encoder = LabelEncoder()
        state_encoder = LabelEncoder()
        
        # Eksik değerleri doldur
        df_copy = df.copy()
        df_copy['subject'] = df_copy['subject'].fillna('unknown')
        df_copy['party_affiliation'] = df_copy['party_affiliation'].fillna('unknown')
        df_copy['state_info'] = df_copy['state_info'].fillna('unknown')
        
        # Kategorik değişkenleri encode et
        subject_encoded = subject_encoder.fit_transform(df_copy['subject'])
        party_encoded = party_encoder.fit_transform(df_copy['party_affiliation'])
        state_encoded = state_encoder.fit_transform(df_copy['state_info'])
        
        # Sayısal özellikler
        numeric_columns = [
            'barely_true_counts', 'false_counts', 'half_true_counts',
            'mostly_true_counts', 'pants_on_fire_counts'
        ]
        
        for col in numeric_columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        
        # Özellik matrisini oluştur
        for i in range(len(df_copy)):
            row = df_copy.iloc[i]
            
            # Kategorik özellikler
            feature_vector = [
                subject_encoded[i],
                party_encoded[i], 
                state_encoded[i]
            ]
            
            # Sayısal özellikler
            for col in numeric_columns:
                feature_vector.append(float(row[col]))
            
            # Toplam geçmiş claims
            total_claims = sum([float(row[col]) for col in numeric_columns])
            feature_vector.append(total_claims)
            
            # Yalan oranı
            false_claims = float(row['false_counts']) + float(row['pants_on_fire_counts'])
            false_ratio = false_claims / total_claims if total_claims > 0 else 0
            feature_vector.append(false_ratio)
            
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        print(f"   ✅ Metadata özellikleri: {features_array.shape}")
        print(f"   📊 Özellikler: subject, party, state, history_counts, total_claims, false_ratio")
        
        return features_array
    
    def combine_features(self, tfidf_features: np.ndarray,
                        statistical_features: np.ndarray,
                        metadata_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Tüm özellikleri birleştir
        
        Args:
            tfidf_features: TF-IDF özellikleri
            statistical_features: İstatistiksel özellikler
            metadata_features: Metadata özellikleri (opsiyonel)
            
        Returns:
            np.ndarray: Birleştirilmiş özellik matrisi
        """
        print("🔗 Özellikler birleştiriliyor...")
        
        features_list = [tfidf_features, statistical_features]
        
        if metadata_features is not None:
            features_list.append(metadata_features)
        
        combined_features = np.hstack(features_list)
        
        print(f"   ✅ Birleştirilmiş özellikler: {combined_features.shape}")
        
        return combined_features
    
    def scale_features(self, X_train: np.ndarray, 
                      X_valid: Optional[np.ndarray] = None,
                      X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """
        Özellikleri ölçeklendir (sadece non-TF-IDF özellikler için)
        
        Args:
            X_train: Train özellikleri
            X_valid: Validation özellikleri
            X_test: Test özellikleri
            
        Returns:
            Tuple: Ölçeklendirilmiş özellik matrisleri
        """
        print("⚖️ Özellikler ölçeklendiriliyor...")
        
        # Sadece TF-IDF olmayan sütunları ölçeklendir
        # TF-IDF zaten 0-1 arasında normalize edilmiş
        
        # Son sütunlar (statistical + metadata) ölçeklendirilecek
        tfidf_size = self.max_features if self.tfidf_vectorizer else 0
        
        if X_train.shape[1] > tfidf_size:
            # TF-IDF + diğer özellikler var
            X_train_tfidf = X_train[:, :tfidf_size]
            X_train_others = X_train[:, tfidf_size:]
            
            # Diğer özellikleri ölçeklendir
            X_train_others_scaled = self.scaler.fit_transform(X_train_others)
            X_train_scaled = np.hstack([X_train_tfidf, X_train_others_scaled])
            
            results = [X_train_scaled]
            
            if X_valid is not None:
                X_valid_tfidf = X_valid[:, :tfidf_size]
                X_valid_others = X_valid[:, tfidf_size:]
                X_valid_others_scaled = self.scaler.transform(X_valid_others)
                X_valid_scaled = np.hstack([X_valid_tfidf, X_valid_others_scaled])
                results.append(X_valid_scaled)
            
            if X_test is not None:
                X_test_tfidf = X_test[:, :tfidf_size]
                X_test_others = X_test[:, tfidf_size:]
                X_test_others_scaled = self.scaler.transform(X_test_others)
                X_test_scaled = np.hstack([X_test_tfidf, X_test_others_scaled])
                results.append(X_test_scaled)
                
        else:
            # Sadece TF-IDF var, ölçeklendirme gerekmiyor
            results = [X_train]
            if X_valid is not None:
                results.append(X_valid)
            if X_test is not None:
                results.append(X_test)
        
        print("   ✅ Ölçeklendirme tamamlandı!")
        
        return tuple(results)
    
    def get_feature_names(self) -> List[str]:
        """
        Tüm özellik isimlerini döndür
        
        Returns:
            List[str]: Özellik isimleri
        """
        feature_names = []
        
        # TF-IDF özellik isimleri
        if self.tfidf_vectorizer:
            tfidf_names = self.tfidf_vectorizer.get_feature_names_out().tolist()
            feature_names.extend(tfidf_names)
        
        # İstatistiksel özellik isimleri
        stat_names = [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'exclamation_count', 'question_count',
            'capital_ratio', 'digit_ratio', 'unique_word_ratio', 'stopword_ratio'
        ]
        feature_names.extend(stat_names)
        
        # Metadata özellik isimleri
        meta_names = [
            'subject_encoded', 'party_encoded', 'state_encoded',
            'barely_true_counts', 'false_counts', 'half_true_counts',
            'mostly_true_counts', 'pants_on_fire_counts', 'total_claims', 'false_ratio'
        ]
        feature_names.extend(meta_names)
        
        return feature_names
    
    def create_all_features(self, train_df: pd.DataFrame,
                          valid_df: Optional[pd.DataFrame] = None,
                          test_df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Tüm özellikleri oluştur
        
        Args:
            train_df: Train DataFrame
            valid_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Dict: Tüm özellik matrisleri
        """
        print("🚀 Tüm özellikler oluşturuluyor...")
        
        # Text listelerini hazırla
        train_texts = [' '.join(tokens) if isinstance(tokens, list) else str(tokens) 
                      for tokens in train_df['tokens']]
        
        valid_texts = None
        test_texts = None
        
        if valid_df is not None:
            valid_texts = [' '.join(tokens) if isinstance(tokens, list) else str(tokens) 
                          for tokens in valid_df['tokens']]
        
        if test_df is not None:
            test_texts = [' '.join(tokens) if isinstance(tokens, list) else str(tokens) 
                         for tokens in test_df['tokens']]
        
        # 1. TF-IDF özellikleri
        tfidf_results = self.create_tfidf_features(train_texts, valid_texts, test_texts)
        
        # 2. İstatistiksel özellikler
        train_stat_features = self.create_statistical_features(train_df)
        
        valid_stat_features = None
        test_stat_features = None
        
        if valid_df is not None:
            valid_stat_features = self.create_statistical_features(valid_df)
        
        if test_df is not None:
            test_stat_features = self.create_statistical_features(test_df)
        
        # 3. Metadata özellikleri
        train_meta_features = self.create_metadata_features(train_df)
        
        valid_meta_features = None
        test_meta_features = None
        
        if valid_df is not None:
            valid_meta_features = self.create_metadata_features(valid_df)
        
        if test_df is not None:
            test_meta_features = self.create_metadata_features(test_df)
        
        # 4. Özellikleri birleştir
        X_train = self.combine_features(tfidf_results[0], train_stat_features, train_meta_features)
        
        X_valid = None
        X_test = None
        
        if valid_df is not None and len(tfidf_results) > 1:
            X_valid = self.combine_features(tfidf_results[1], valid_stat_features, valid_meta_features)
        
        if test_df is not None and len(tfidf_results) > 2:
            X_test = self.combine_features(tfidf_results[2], test_stat_features, test_meta_features)
        
        # 5. Ölçeklendirme
        scaled_results = self.scale_features(X_train, X_valid, X_test)
        
        # Sonuçları hazırla
        results = {
            'X_train': scaled_results[0],
            'y_train': train_df['binary_label'].values
        }
        
        if valid_df is not None and len(scaled_results) > 1:
            results['X_valid'] = scaled_results[1]
            results['y_valid'] = valid_df['binary_label'].values
        
        if test_df is not None and len(scaled_results) > 2:
            results['X_test'] = scaled_results[2]
            results['y_test'] = test_df['binary_label'].values
        
        print("🎉 Tüm özellikler oluşturuldu!")
        print(f"   📊 Final özellik boyutu: {results['X_train'].shape}")
        print(f"   🎯 Train örnekleri: {len(results['y_train'])}")
        
        if 'X_valid' in results:
            print(f"   🎯 Valid örnekleri: {len(results['y_valid'])}")
        
        if 'X_test' in results:
            print(f"   🎯 Test örnekleri: {len(results['y_test'])}")
        
        return results

# Test fonksiyonu
def main():
    """Test fonksiyonu"""
    import sys
    sys.path.append('.')
    from utils import ConfigManager
    
    # Config yükle
    config = ConfigManager.load_config()
    
    # Feature engineering test
    fe = FeatureEngineering(config)
    
    print("✅ FeatureEngineering modülü test edildi!")

if __name__ == "__main__":
    main()