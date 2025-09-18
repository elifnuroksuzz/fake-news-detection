# Paket kurulum dosyası
from setuptools import setup, find_packages

setup(
    name='fake_news_detector',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='Fake News Detection Project',
    url='https://github.com/yourusername/fake_news_detector'
)
