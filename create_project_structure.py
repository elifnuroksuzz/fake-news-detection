import os

def create_project_structure():
    # Ana proje dizini
    project_root = "fake_news_detector"
    os.makedirs(project_root, exist_ok=True)

    # Klasör yapısını tanımlama
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "src",
        "models/trained_models",
        "models/model_configs",
        "notebooks",
        "results/figures",
        "results/reports",
        "results/metrics"
    ]

    # Klasörleri oluşturma
    for directory in directories:
        os.makedirs(os.path.join(project_root, directory), exist_ok=True)

    # Dosyaları oluşturma
    files = {
        "src/__init__.py": "",
        "src/data_preprocessing.py": "# Veri ön işleme işlemleri\n\ndef preprocess_data():\n    pass\n",
        "src/feature_engineering.py": "# Özellik çıkarma işlemleri\n\ndef extract_features():\n    pass\n",
        "src/model_training.py": "# Model eğitimi işlemleri\n\ndef train_model():\n    pass\n",
        "src/model_evaluation.py": "# Model değerlendirme işlemleri\n\ndef evaluate_model():\n    pass\n",
        "src/utils.py": "# Yardımcı fonksiyonlar\n\ndef helper_function():\n    pass\n",
        "notebooks/01_data_exploration.ipynb": '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
        "notebooks/02_preprocessing.ipynb": '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
        "notebooks/03_model_experiments.ipynb": '{\n "cells": [],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 5\n}',
        "requirements.txt": "# Proje bağımlılıkları\npandas\nnumpy\nscikit-learn\ntensorflow\n",
        "setup.py": "# Paket kurulum dosyası\n"
                   "from setuptools import setup, find_packages\n\n"
                   "setup(\n"
                   "    name='fake_news_detector',\n"
                   "    version='0.1.0',\n"
                   "    packages=find_packages(),\n"
                   "    install_requires=[\n"
                   "        'pandas',\n"
                   "        'numpy',\n"
                   "        'scikit-learn',\n"
                   "        'tensorflow'\n"
                   "    ],\n"
                   "    author='Your Name',\n"
                   "    author_email='your.email@example.com',\n"
                   "    description='Fake News Detection Project',\n"
                   "    url='https://github.com/yourusername/fake_news_detector'\n"
                   ")\n",
        "config.yaml": "# Proje konfigürasyon dosyası\nmodel:\n  name: default_model\n  parameters:\n    learning_rate: 0.01\n",
        "main.py": "# Ana çalışma dosyası\n\ndef main():\n    print('Fake News Detector Project')\n\nif __name__ == '__main__':\n    main()\n",
        "README.md": "# Fake News Detector\nBu proje, sahte haber tespitine yönelik bir makine öğrenimi projesidir.\n"
                   "## Kurulum\n1. Depoyu klonlayın\n2. `pip install -r requirements.txt` komutunu çalıştırın\n"
                   "## Kullanım\n`python main.py` komutu ile projeyi çalıştırabilirsiniz.\n"
    }

    # Dosyaları oluşturma
    for file_path, content in files.items():
        with open(os.path.join(project_root, file_path), "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Proje yapısı '{project_root}' dizininde oluşturuldu.")

if __name__ == "__main__":
    create_project_structure()