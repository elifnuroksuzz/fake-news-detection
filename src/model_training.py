"""
model_training.py
Profesyonel model eğitimi ve hiperparametre optimizasyonu
10-fold cross validation, multiple algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost bulunamadı, Random Forest kullanılacak")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM bulunamadı, Random Forest kullanılacak")

try:
    import optuna
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna loglarını azalt
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna bulunamadı, Grid Search kullanılacak")

class ModelTrainer:
    """
    Profesyonel model eğitimi sınıfı
    Multiple algorithms, hyperparameter optimization, cross-validation
    """
    
    def __init__(self, config: Dict):
        """
        Model trainer başlatıcısı
        
        Args:
            config (Dict): Konfigürasyon sözlüğü
        """
        self.config = config
        self.models = {}
        self.best_models = {}
        self.cv_results = {}
        self.random_state = config.get('reproducibility', {}).get('random_seed', 42)
        
        # Cross-validation ayarları
        cv_config = config.get('models', {}).get('cross_validation', {})
        self.n_folds = cv_config.get('n_folds', 10)
        self.cv_shuffle = cv_config.get('shuffle', True)
        self.cv_stratify = cv_config.get('stratify', True)
        
        # Hyperparameter tuning ayarları
        tuning_config = config.get('models', {}).get('hyperparameter_tuning', {})
        self.tuning_method = tuning_config.get('method', 'grid_search')
        self.n_trials = tuning_config.get('n_trials', 50)
        self.timeout = tuning_config.get('timeout', 1800)  # 30 dakika
        
        print("🤖 ModelTrainer başlatıldı!")
        print(f"   🔄 Cross-validation: {self.n_folds}-fold")
        print(f"   🎯 Tuning method: {self.tuning_method}")
        print(f"   🎲 Random state: {self.random_state}")
    
    def get_model_definitions(self) -> Dict[str, Any]:
        """
        Model tanımlarını döndür
        
        Returns:
            Dict: Model tanımları
        """
        models = {
            'RandomForest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # XGBoost ekle (varsa)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        
        # LightGBM ekle (varsa)
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
        
        # Logistic Regression ekle
        models['LogisticRegression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        
        # SVM ekle (küçük veri setleri için)
        models['SVM'] = SVC(
            random_state=self.random_state,
            probability=True  # ROC-AUC için gerekli
        )
        
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """
        Hyperparameter grid'leri döndür
        
        Returns:
            Dict: Her model için hyperparameter grid'i
        """
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            
            'LogisticRegression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            
            'SVM': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
            }
        }
        
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        if LIGHTGBM_AVAILABLE:
            param_grids['LightGBM'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'num_leaves': [31, 63, 127, 255],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        return param_grids
    
    def perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray, 
                               model_name: str) -> Dict[str, float]:
        """
        10-fold cross validation gerçekleştir
        
        Args:
            model: ML modeli
            X: Özellik matrisi
            y: Etiketler
            model_name: Model adı
            
        Returns:
            Dict: CV metrikleri
        """
        print(f"🔄 {model_name} - 10-fold cross validation...")
        
        # StratifiedKFold oluştur
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=self.cv_shuffle,
            random_state=self.random_state
        )
        
        # Scoring metrikleri
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_scores = {}
        
        for metric in scoring_metrics:
            scores = cross_val_score(
                model, X, y, 
                cv=skf, 
                scoring=metric,
                n_jobs=-1
            )
            
            cv_scores[f'{metric}_mean'] = scores.mean()
            cv_scores[f'{metric}_std'] = scores.std()
            
            print(f"   📊 {metric.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return cv_scores
    
    def optimize_hyperparameters_optuna(self, model_class: Any, param_grid: Dict,
                                       X: np.ndarray, y: np.ndarray, 
                                       model_name: str) -> Tuple[Any, Dict]:
        """
        Optuna ile hyperparameter optimization
        
        Args:
            model_class: Model sınıfı
            param_grid: Parameter grid'i
            X: Özellik matrisi
            y: Etiketler
            model_name: Model adı
            
        Returns:
            Tuple: (best_model, best_params)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna yüklü değil!")
        
        print(f"🎯 {model_name} - Optuna hyperparameter optimization...")
        
        def objective(trial):
            # Parametre önerilerini al
            params = {}
            
            for param_name, param_values in param_grid.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(
                        param_name, min(param_values), max(param_values)
                    )
                elif isinstance(param_values[0], float):
                    params[param_name] = trial.suggest_float(
                        param_name, min(param_values), max(param_values)
                    )
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Model oluştur
            if model_name == 'LogisticRegression':
                # Logistic Regression için solver uyumluluğu
                if params.get('penalty') == 'elasticnet':
                    params['solver'] = 'saga'
                elif params.get('penalty') == 'l1':
                    params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'saga'])
                
                # l1_ratio sadece elasticnet için
                if params.get('penalty') != 'elasticnet' and 'l1_ratio' in params:
                    del params['l1_ratio']
            
            model = model_class(**params, random_state=self.random_state)
            
            # Cross-validation skoru
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=skf, scoring='f1', n_jobs=-1)
            
            return scores.mean()
        
        # Optuna study oluştur
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.random_state))
        
        # Optimization çalıştır
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
        
        # En iyi parametreleri al
        best_params = study.best_params
        
        # LogisticRegression özel durumu
        if model_name == 'LogisticRegression':
            if best_params.get('penalty') != 'elasticnet' and 'l1_ratio' in best_params:
                del best_params['l1_ratio']
        
        # En iyi modeli oluştur
        best_model = model_class(**best_params, random_state=self.random_state)
        
        print(f"   🏆 En iyi F1-Score: {study.best_value:.4f}")
        print(f"   ⚙️ En iyi parametreler: {best_params}")
        
        return best_model, best_params
    
    def optimize_hyperparameters_grid(self, model: Any, param_grid: Dict,
                                     X: np.ndarray, y: np.ndarray,
                                     model_name: str) -> Tuple[Any, Dict]:
        """
        Grid Search ile hyperparameter optimization
        
        Args:
            model: ML modeli
            param_grid: Parameter grid'i
            X: Özellik matrisi
            y: Etiketler
            model_name: Model adı
            
        Returns:
            Tuple: (best_model, best_params)
        """
        print(f"🎯 {model_name} - Grid Search hyperparameter optimization...")
        
        # LogisticRegression özel durumu
        if model_name == 'LogisticRegression':
            # Uyumlu parametre kombinasyonları oluştur
            valid_combinations = []
            
            for penalty in param_grid['penalty']:
                for solver in param_grid['solver']:
                    for C in param_grid['C']:
                        combination = {'penalty': penalty, 'solver': solver, 'C': C}
                        
                        # Uyumluluk kontrolü
                        if penalty == 'elasticnet' and solver == 'saga':
                            for l1_ratio in param_grid['l1_ratio']:
                                combination['l1_ratio'] = l1_ratio
                                valid_combinations.append(combination.copy())
                        elif penalty == 'l1' and solver in ['liblinear', 'saga']:
                            valid_combinations.append(combination)
                        elif penalty == 'l2':
                            valid_combinations.append(combination)
            
            param_grid = valid_combinations
        
        # StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Grid Search
        if isinstance(param_grid, list):
            # Logistic Regression özel durumu
            grid_search = GridSearchCV(
                model, param_grid, cv=skf, scoring='f1',
                n_jobs=-1, verbose=0
            )
        else:
            grid_search = GridSearchCV(
                model, param_grid, cv=skf, scoring='f1',
                n_jobs=-1, verbose=0
            )
        
        # Fit et
        grid_search.fit(X, y)
        
        print(f"   🏆 En iyi F1-Score: {grid_search.best_score_:.4f}")
        print(f"   ⚙️ En iyi parametreler: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def train_single_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Tek bir modeli eğit
        
        Args:
            model_name: Model adı
            X: Özellik matrisi
            y: Etiketler
            
        Returns:
            Dict: Model sonuçları
        """
        print(f"\n🤖 {model_name} modeli eğitiliyor...")
        print("=" * 50)
        
        # Model ve parameter grid'ini al
        models = self.get_model_definitions()
        param_grids = self.get_hyperparameter_grids()
        
        if model_name not in models:
            print(f"❌ {model_name} modeli bulunamadı!")
            return {}
        
        base_model = models[model_name]
        param_grid = param_grids.get(model_name, {})
        
        # 1. Baseline cross-validation
        baseline_cv_scores = self.perform_cross_validation(base_model, X, y, f"{model_name} (Baseline)")
        
        # 2. Hyperparameter optimization
        best_model = base_model
        best_params = {}
        
        if param_grid and len(param_grid) > 0:
            try:
                if self.tuning_method == 'optuna' and OPTUNA_AVAILABLE:
                    best_model, best_params = self.optimize_hyperparameters_optuna(
                        type(base_model), param_grid, X, y, model_name
                    )
                else:
                    best_model, best_params = self.optimize_hyperparameters_grid(
                        base_model, param_grid, X, y, model_name
                    )
            except Exception as e:
                print(f"⚠️ Hyperparameter optimization hatası: {str(e)}")
                print("   Baseline model kullanılacak...")
                best_model = base_model
        
        # 3. En iyi modelle final cross-validation
        print(f"\n🔄 {model_name} - Final cross validation (optimized)...")
        optimized_cv_scores = self.perform_cross_validation(best_model, X, y, f"{model_name} (Optimized)")
        
        # 4. Modeli tam veri üzerinde eğit
        print(f"🎯 {model_name} - Full dataset training...")
        best_model.fit(X, y)
        
        # Sonuçları hazırla
        results = {
            'model': best_model,
            'model_name': model_name,
            'best_params': best_params,
            'baseline_cv_scores': baseline_cv_scores,
            'optimized_cv_scores': optimized_cv_scores,
            'improvement': {
                'f1_improvement': optimized_cv_scores['f1_mean'] - baseline_cv_scores['f1_mean'],
                'accuracy_improvement': optimized_cv_scores['accuracy_mean'] - baseline_cv_scores['accuracy_mean']
            }
        }
        
        print(f"✅ {model_name} eğitimi tamamlandı!")
        print(f"   📈 F1-Score Improvement: {results['improvement']['f1_improvement']:+.4f}")
        print(f"   📈 Accuracy Improvement: {results['improvement']['accuracy_improvement']:+.4f}")
        
        return results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        Tüm modelleri eğit
        
        Args:
            X: Özellik matrisi
            y: Etiketler
            
        Returns:
            Dict: Tüm model sonuçları
        """
        print("🚀 TÜM MODELLER EĞİTİLİYOR!")
        print("=" * 60)
        
        # Kullanılabilir modeller
        available_models = ['RandomForest', 'LogisticRegression', 'SVM']
        
        if XGBOOST_AVAILABLE:
            available_models.append('XGBoost')
        
        if LIGHTGBM_AVAILABLE:
            available_models.append('LightGBM')
        
        print(f"📋 Eğitilecek modeller: {available_models}")
        print(f"📊 Veri boyutu: {X.shape}")
        print(f"🎯 Hedef dağılımı: Fake={np.sum(y==0)}, Real={np.sum(y==1)}")
        print()
        
        all_results = {}
        
        for model_name in available_models:
            try:
                result = self.train_single_model(model_name, X, y)
                if result:
                    all_results[model_name] = result
                    self.best_models[model_name] = result['model']
                    self.cv_results[model_name] = result['optimized_cv_scores']
            
            except Exception as e:
                print(f"❌ {model_name} eğitimi başarısız: {str(e)}")
                continue
        
        # En iyi modeli belirle
        if all_results:
            best_model_name = max(all_results.keys(), 
                                key=lambda x: all_results[x]['optimized_cv_scores']['f1_mean'])
            
            print("\n" + "=" * 60)
            print("🏆 MODEL KARŞILAŞTIRMA ÖZETİ")
            print("=" * 60)
            
            for model_name, result in all_results.items():
                cv_scores = result['optimized_cv_scores']
                improvement = result['improvement']
                
                status = "🥇" if model_name == best_model_name else "🥈"
                
                print(f"{status} {model_name}:")
                print(f"   📊 F1-Score: {cv_scores['f1_mean']:.4f} ± {cv_scores['f1_std']:.4f} (+{improvement['f1_improvement']:+.4f})")
                print(f"   🎯 Accuracy: {cv_scores['accuracy_mean']:.4f} ± {cv_scores['accuracy_std']:.4f} (+{improvement['accuracy_improvement']:+.4f})")
                print(f"   🔍 Precision: {cv_scores['precision_mean']:.4f} ± {cv_scores['precision_std']:.4f}")
                print(f"   📈 Recall: {cv_scores['recall_mean']:.4f} ± {cv_scores['recall_std']:.4f}")
                print(f"   📊 ROC-AUC: {cv_scores['roc_auc_mean']:.4f} ± {cv_scores['roc_auc_std']:.4f}")
                print()
            
            print(f"🏆 EN İYİ MODEL: {best_model_name}")
            print(f"🎯 En İyi F1-Score: {all_results[best_model_name]['optimized_cv_scores']['f1_mean']:.4f}")
            
            # En iyi modeli kaydet
            self.best_model_name = best_model_name
            self.best_model = all_results[best_model_name]['model']
        
        return all_results
    
    def get_best_model(self) -> Tuple[Any, str]:
        """
        En iyi modeli döndür
        
        Returns:
            Tuple: (best_model, model_name)
        """
        if hasattr(self, 'best_model'):
            return self.best_model, self.best_model_name
        else:
            return None, None
    
    def save_models(self, save_path: str):
        """
        Modelleri kaydet
        
        Args:
            save_path: Kayıt klasörü
        """
        import os
        import joblib
        
        os.makedirs(save_path, exist_ok=True)
        
        # Tüm modelleri kaydet
        for model_name, model in self.best_models.items():
            model_file = os.path.join(save_path, f"{model_name.lower()}_model.pkl")
            joblib.dump(model, model_file)
            print(f"💾 {model_name} kaydedildi: {model_file}")
        
        # CV sonuçlarını kaydet - dict olarak kaydet
        cv_results_file = os.path.join(save_path, "cv_results.json")
        import json
        with open(cv_results_file, 'w') as f:
            json.dump(self.cv_results, f, indent=2)
        print(f"💾 CV results kaydedildi: {cv_results_file}")
        
        # En iyi model bilgisini kaydet
        if hasattr(self, 'best_model_name'):
            best_model_file = os.path.join(save_path, f"best_model_{self.best_model_name.lower()}.pkl")
            joblib.dump(self.best_model, best_model_file)
            print(f"🏆 En iyi model kaydedildi: {best_model_file}")

# Test fonksiyonu
def main():
    """Test fonksiyonu"""
    print("🧪 ModelTrainer test ediliyor...")
    
    # Dummy data
    np.random.seed(42)
    X = np.random.rand(100, 20)
    y = np.random.randint(0, 2, 100)
    
    # Config
    config = {
        'models': {
            'cross_validation': {'n_folds': 5},
            'hyperparameter_tuning': {'method': 'grid_search', 'n_trials': 10}
        },
        'reproducibility': {'random_seed': 42}
    }
    
    # Model trainer
    trainer = ModelTrainer(config)
    
    # Tek model test
    result = trainer.train_single_model('RandomForest', X, y)
    print(f"Test sonucu: {result['optimized_cv_scores']['f1_mean']:.4f}")
    
    print("✅ ModelTrainer test tamamlandı!")

if __name__ == "__main__":
    main()