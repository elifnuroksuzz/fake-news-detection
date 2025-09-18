"""
data_preprocessing.py
Profesyonel veri indirme, temizleme ve ön işleme modülü
LIAR Dataset için optimize edilmiş
"""

import os
import re
import pandas as pd
import numpy as np
import requests
from typing import Tuple, List, Dict, Optional
import yaml
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Profesyonel veri ön işleme sınıfı
    LIAR Dataset için özelleştirilmiş
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Veri ön işleyiciyi başlat
        
        Args:
            config_path (str): Konfigürasyon dosyası yolu
        """
        self.config = self._load_config(config_path)
        self.stemmer = PorterStemmer()
        self._setup_nltk()
        
        # LIAR Dataset sütun isimleri
        self.liar_columns = [
            'label', 'statement', 'subject', 'speaker', 'job_title',
            'state_info', 'party_affiliation', 'barely_true_counts',
            'false_counts', 'half_true_counts', 'mostly_true_counts',
            'pants_on_fire_counts', 'context'
        ]
        
        print("🔧 DataPreprocessor başlatıldı!")
        
    def _load_config(self, config_path: str) -> Dict:
        """Konfigürasyon dosyasını yükle"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"⚠️ Konfigürasyon dosyası bulunamadı: {config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Varsayılan konfigürasyon"""
        return {
            'data': {
                'raw_data_urls': {
                    'train': "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
                    'valid': "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv",
                    'test': "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv"
                },
                'raw_data_path': "data/raw/",
                'processed_data_path': "data/processed/",
                'text_cleaning': {
                    'remove_urls': True,
                    'remove_mentions': True,
                    'remove_hashtags': False,
                    'remove_punctuation': True,
                    'convert_lowercase': True,
                    'remove_stopwords': True,
                    'min_word_length': 2
                }
            }
        }
    
    def _setup_nltk(self):
        """NLTK gerekli paketlerini indir"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("📥 NLTK paketleri indiriliyor...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("✅ NLTK paketleri hazır!")
    
    def download_liar_dataset(self) -> bool:
        """
        LIAR Dataset'i indir
        
        Returns:
            bool: İndirme başarılı mı
        """
        print("📥 LIAR Dataset indiriliyor...")
        
        # Raw data klasörünü oluştur
        raw_path = self.config['data']['raw_data_path']
        os.makedirs(raw_path, exist_ok=True)
        
        urls = self.config['data']['raw_data_urls']
        success_count = 0
        
        for dataset_name, url in urls.items():
            try:
                print(f"📥 {dataset_name}.tsv indiriliyor...")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                file_path = os.path.join(raw_path, f"{dataset_name}.tsv")
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                
                print(f"✅ {dataset_name}.tsv indirildi: {file_path}")
                success_count += 1
                
            except Exception as e:
                print(f"❌ {dataset_name}.tsv indirilemedi: {str(e)}")
        
        if success_count == len(urls):
            print("🎉 Tüm veri dosyaları başarıyla indirildi!")
            return True
        else:
            print(f"⚠️ {success_count}/{len(urls)} dosya indirildi")
            return False
    
    def load_liar_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        LIAR Dataset'i yükle
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, valid, test dataframes
        """
        print("📊 LIAR Dataset yükleniyor...")
        
        raw_path = self.config['data']['raw_data_path']
        
        try:
            # TSV dosyalarını yükle
            train_df = pd.read_csv(
                os.path.join(raw_path, "train.tsv"), 
                sep='\t', 
                names=self.liar_columns,
                encoding='utf-8'
            )
            
            valid_df = pd.read_csv(
                os.path.join(raw_path, "valid.tsv"), 
                sep='\t', 
                names=self.liar_columns,
                encoding='utf-8'
            )
            
            test_df = pd.read_csv(
                os.path.join(raw_path, "test.tsv"), 
                sep='\t', 
                names=self.liar_columns,
                encoding='utf-8'
            )
            
            print(f"✅ Veriler yüklendi:")
            print(f"   📈 Train: {len(train_df):,} örnек")
            print(f"   📊 Valid: {len(valid_df):,} örnек")
            print(f"   🧪 Test: {len(test_df):,} örnək")
            
            return train_df, valid_df, test_df
            
        except Exception as e:
            print(f"❌ Veri yüklenirken hata: {str(e)}")
            return None, None, None
    
    def clean_text(self, text: str) -> str:
        """
        Metin temizleme fonksiyonu
        
        Args:
            text (str): Temizlenecek metin
            
        Returns:
            str: Temizlenmiş metin
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        config = self.config['data']['text_cleaning']
        
        # URL'leri kaldır
        if config.get('remove_urls', True):
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Mention'ları kaldır (@username)
        if config.get('remove_mentions', True):
            text = re.sub(r'@\w+', '', text)
        
        # Hashtag'leri kaldır (#hashtag)
        if config.get('remove_hashtags', False):
            text = re.sub(r'#\w+', '', text)
        
        # Küçük harfe çevir
        if config.get('convert_lowercase', True):
            text = text.lower()
        
        # Noktalama işaretlerini kaldır
        if config.get('remove_punctuation', True):
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Metni tokenize et ve filtrele
        
        Args:
            text (str): İşlenecek metin
            
        Returns:
            List[str]: Filtrelenmiş token listesi
        """
        if not text:
            return []
        
        config = self.config['data']['text_cleaning']
        
        # Tokenize et
        tokens = word_tokenize(text)
        
        # Stopwords'leri kaldır
        if config.get('remove_stopwords', True):
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]
        
        # Minimum kelime uzunluğu filtresi
        min_length = config.get('min_word_length', 2)
        tokens = [token for token in tokens if len(token) >= min_length]
        
        # Sadece alfabetik karakterler
        tokens = [token for token in tokens if token.isalpha()]
        
        return tokens
    
    def preprocess_dataframe(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        DataFrame'i ön işleme
        
        Args:
            df (pd.DataFrame): İşlenecek dataframe
            dataset_name (str): Dataset adı (train/valid/test)
            
        Returns:
            pd.DataFrame: İşlenmiş dataframe
        """
        print(f"🔄 {dataset_name} verisi ön işleme alınıyor...")
        
        df_processed = df.copy()
        
        # Eksik değerleri doldur
        df_processed['statement'] = df_processed['statement'].fillna('')
        df_processed['subject'] = df_processed['subject'].fillna('unknown')
        df_processed['speaker'] = df_processed['speaker'].fillna('unknown')
        
        # Ana metin temizleme
        print("   🧹 Metinler temizleniyor...")
        df_processed['cleaned_statement'] = df_processed['statement'].progress_apply(self.clean_text)
        
        # Tokenize etme
        print("   🔤 Tokenize ediliyor...")
        df_processed['tokens'] = df_processed['cleaned_statement'].progress_apply(self.tokenize_and_filter)
        
        # Token sayısı
        df_processed['token_count'] = df_processed['tokens'].apply(len)
        
        # Boş metinleri filtrele
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed['token_count'] > 0]
        final_count = len(df_processed)
        
        if initial_count != final_count:
            print(f"   🗑️ {initial_count - final_count} boş örnek kaldırıldı")
        
        # Label encoding (6-class -> binary için)
        label_mapping = {
            'pants-fire': 0,    # Kesinlikle yanlış
            'false': 0,         # Yanlış  
            'barely-true': 0,   # Neredeyse doğru (yanlış olarak kabul)
            'half-true': 1,     # Yarı doğru
            'mostly-true': 1,   # Çoğunlukla doğru
            'true': 1          # Doğru
        }
        
        df_processed['binary_label'] = df_processed['label'].map(label_mapping)
        
        print(f"✅ {dataset_name} ön işleme tamamlandı!")
        print(f"   📊 Final örnek sayısı: {len(df_processed):,}")
        print(f"   📊 Label dağılımı:")
        print(f"      🔴 Fake (0): {(df_processed['binary_label'] == 0).sum():,}")
        print(f"      🟢 Real (1): {(df_processed['binary_label'] == 1).sum():,}")
        
        return df_processed
    
    def save_processed_data(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        İşlenmiş veriyi kaydet
        
        Args:
            train_df: İşlenmiş train verisi
            valid_df: İşlenmiş validation verisi  
            test_df: İşlenmiş test verisi
        """
        print("💾 İşlenmiş veriler kaydediliyor...")
        
        processed_path = self.config['data']['processed_data_path']
        os.makedirs(processed_path, exist_ok=True)
        
        # CSV olarak kaydet
        train_df.to_csv(os.path.join(processed_path, "train_processed.csv"), index=False)
        valid_df.to_csv(os.path.join(processed_path, "valid_processed.csv"), index=False)
        test_df.to_csv(os.path.join(processed_path, "test_processed.csv"), index=False)
        
        # Pickle olarak da kaydet (daha hızlı yükleme için)
        train_df.to_pickle(os.path.join(processed_path, "train_processed.pkl"))
        valid_df.to_pickle(os.path.join(processed_path, "valid_processed.pkl"))
        test_df.to_pickle(os.path.join(processed_path, "test_processed.pkl"))
        
        print("✅ İşlenmiş veriler kaydedildi!")
        print(f"   📁 Konum: {processed_path}")
    
    def get_data_statistics(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Veri istatistikleri
        
        Returns:
            Dict: İstatistik bilgileri
        """
        stats = {
            'datasets': {
                'train': {
                    'count': len(train_df),
                    'fake_count': (train_df['binary_label'] == 0).sum(),
                    'real_count': (train_df['binary_label'] == 1).sum(),
                    'avg_token_count': train_df['token_count'].mean()
                },
                'valid': {
                    'count': len(valid_df),
                    'fake_count': (valid_df['binary_label'] == 0).sum(),
                    'real_count': (valid_df['binary_label'] == 1).sum(),
                    'avg_token_count': valid_df['token_count'].mean()
                },
                'test': {
                    'count': len(test_df),
                    'fake_count': (test_df['binary_label'] == 0).sum(),
                    'real_count': (test_df['binary_label'] == 1).sum(),
                    'avg_token_count': test_df['token_count'].mean()
                }
            }
        }
        
        return stats
    
    def run_full_preprocessing(self) -> bool:
        """
        Tam ön işleme pipeline'ını çalıştır
        
        Returns:
            bool: İşlem başarılı mı
        """
        print("🚀 Tam veri ön işleme başlatılıyor...\n")
        
        # 1. Veriyi indir
        if not self.download_liar_dataset():
            return False
        
        # 2. Veriyi yükle
        train_df, valid_df, test_df = self.load_liar_data()
        if train_df is None:
            return False
        
        print()
        
        # 3. Tqdm'i etkinleştir
        tqdm.pandas()
        
        # 4. Her dataset'i ön işleme
        train_processed = self.preprocess_dataframe(train_df, "Train")
        print()
        valid_processed = self.preprocess_dataframe(valid_df, "Valid")
        print()
        test_processed = self.preprocess_dataframe(test_df, "Test")
        print()
        
        # 5. İşlenmiş veriyi kaydet
        self.save_processed_data(train_processed, valid_processed, test_processed)
        print()
        
        # 6. İstatistikleri göster
        stats = self.get_data_statistics(train_processed, valid_processed, test_processed)
        print("📊 Final İstatistikler:")
        for dataset_name, dataset_stats in stats['datasets'].items():
            print(f"   {dataset_name.upper()}:")
            print(f"      📈 Toplam: {dataset_stats['count']:,}")
            print(f"      🔴 Fake: {dataset_stats['fake_count']:,} ({dataset_stats['fake_count']/dataset_stats['count']*100:.1f}%)")
            print(f"      🟢 Real: {dataset_stats['real_count']:,} ({dataset_stats['real_count']/dataset_stats['count']*100:.1f}%)")
            print(f"      📝 Avg Tokens: {dataset_stats['avg_token_count']:.1f}")
            print()
        
        print("🎉 Veri ön işleme tamamlandı!")
        return True

# Test fonksiyonu
def main():
    """Test fonksiyonu"""
    preprocessor = DataPreprocessor()
    success = preprocessor.run_full_preprocessing()
    
    if success:
        print("✅ Veri ön işleme başarıyla tamamlandı!")
    else:
        print("❌ Veri ön işleme sırasında hata oluştu!")

if __name__ == "__main__":
    main()