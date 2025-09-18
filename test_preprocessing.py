"""
test_preprocessing.py
Veri indirme ve ön işleme test dosyası
Bu dosyayla veri indirme ve ön işleme işlemlerini test edebilirsiniz
"""

import sys
import os

# src modülünü import edebilmek için path'e ekle
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from utils import setup_reproducibility, print_system_info, create_project_structure

def main():
    """Ana test fonksiyonu"""
    print("🚀 FAKE NEWS DETECTION - VERİ ÖN İŞLEME TEST\n")
    
    # Sistem bilgilerini göster
    print_system_info()
    print()
    
    # Reproducibility ayarla
    setup_reproducibility(42)
    print()
    
    # Proje yapısını oluştur
    print("📁 Proje klasör yapısını kontrol ediliyor...")
    create_project_structure()
    print()
    
    # Veri ön işleyiciyi başlat
    print("🔧 DataPreprocessor başlatılıyor...")
    preprocessor = DataPreprocessor("config.yaml")
    print()
    
    # Tam ön işleme pipeline'ını çalıştır
    print("🚀 Tam veri ön işleme başlatılıyor...")
    print("=" * 60)
    
    success = preprocessor.run_full_preprocessing()
    
    print("=" * 60)
    
    if success:
        print("✅ VERİ ÖN İŞLEME BAŞARIYLA TAMAMLANDI!")
        print("\n📁 Oluşturulan dosyalar:")
        print("   📊 data/processed/train_processed.csv")
        print("   📊 data/processed/valid_processed.csv") 
        print("   📊 data/processed/test_processed.csv")
        print("   🥒 data/processed/train_processed.pkl")
        print("   🥒 data/processed/valid_processed.pkl")
        print("   🥒 data/processed/test_processed.pkl")
        
        print("\n🎯 Bir sonraki adım:")
        print("   Feature engineering ve model eğitime hazırız!")
        
    else:
        print("❌ VERİ ÖN İŞLEME BAŞARISIZ!")
        print("   Lütfen internet bağlantınızı kontrol edin")
        print("   veya manuel olarak veri dosyalarını indirin")

if __name__ == "__main__":
    main()