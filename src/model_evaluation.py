"""
model_evaluation.py
Kapsamlı model değerlendirme ve görselleştirme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Kapsamlı model değerlendirme sınıfı
    """
    
    def __init__(self, config: Dict):
        """
        Model evaluator başlatıcısı
        
        Args:
            config (Dict): Konfigürasyon sözlüğü
        """
        self.config = config
        self.results = {}
        
        # Görselleştirme ayarları
        plot_config = config.get('evaluation', {}).get('plot_settings', {})
        self.figure_size = plot_config.get('figure_size', [12, 8])
        self.save_plots = plot_config.get('save_plots', True)
        self.plot_format = plot_config.get('plot_format', 'png')
        self.dpi = plot_config.get('dpi', 300)
        
        # Sonuç kayıt yolları
        output_config = config.get('output', {})
        self.figures_path = output_config.get('figures_path', 'results/figures/')
        self.reports_path = output_config.get('reports_path', 'results/reports/')
        
        # Stil ayarları
        plt.style.use('default')
        sns.set_style('whitegrid')
        
        print("📊 ModelEvaluator başlatıldı!")
        print(f"   📐 Figure boyutu: {self.figure_size}")
        print(f"   💾 Grafik kayıt: {self.save_plots}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None,
                         model_name: str = "Model") -> Dict:
        """
        Kapsamlı metrikler hesapla
        
        Args:
            y_true: Gerçek etiketler
            y_pred: Tahmin edilen etiketler  
            y_pred_proba: Tahmin olasılıkları
            model_name: Model adı
            
        Returns:
            Dict: Hesaplanan metrikler
        """
        print(f"📊 {model_name} - Metrikler hesaplanıyor...")
        
        # Temel metrikler
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # ROC-AUC (probability gerekli)
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification Report
        metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Sınıf bazında metrikler
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn) 
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Ek metrikler
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        
        return metrics
    
    def print_metrics(self, metrics: Dict, model_name: str = "Model"):
        """
        Metrikleri konsola yazdır
        
        Args:
            metrics: Metrik sözlüğü
            model_name: Model adı
        """
        print(f"\n📊 {model_name} - PERFORMANS METRİKLERİ")
        print("=" * 50)
        
        print(f"🎯 Accuracy:           {metrics['accuracy']:.4f}")
        print(f"🔍 Precision:          {metrics['precision']:.4f}")
        print(f"📈 Recall (Sensitivity): {metrics['recall']:.4f}")
        print(f"⚖️ F1-Score:           {metrics['f1_score']:.4f}")
        print(f"🔒 Specificity:        {metrics['specificity']:.4f}")
        print(f"📉 FPR:                {metrics['fpr']:.4f}")
        print(f"📉 FNR:                {metrics['fnr']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"📊 ROC-AUC:            {metrics['roc_auc']:.4f}")
        
        if 'average_precision' in metrics:
            print(f"📈 Average Precision:  {metrics['average_precision']:.4f}")
        
        print(f"\n🔢 CONFUSION MATRIX:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"   TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
        print(f"   FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
    
    def plot_confusion_matrix(self, metrics: Dict, model_name: str = "Model") -> plt.Figure:
        """
        Confusion matrix çiz
        
        Args:
            metrics: Metrik sözlüğü
            model_name: Model adı
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        cm = np.array(metrics['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Heatmap çiz
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'],
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Accuracy bilgisini ekle
        accuracy = metrics['accuracy']
        ax.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.4f}', 
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if self.save_plots:
            import os
            os.makedirs(self.figures_path, exist_ok=True)
            save_path = os.path.join(self.figures_path, f'{model_name.lower()}_confusion_matrix.{self.plot_format}')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"💾 Confusion matrix kaydedildi: {save_path}")
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str = "Model") -> plt.Figure:
        """
        ROC Curve çiz
        
        Args:
            y_true: Gerçek etiketler
            y_pred_proba: Tahmin olasılıkları
            model_name: Model adı
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # ROC curve hesapla
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # ROC curve çiz
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC Curve (AUC = {auc_score:.4f})')
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier (AUC = 0.5)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{model_name} - ROC Curve', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            import os
            os.makedirs(self.figures_path, exist_ok=True)
            save_path = os.path.join(self.figures_path, f'{model_name.lower()}_roc_curve.{self.plot_format}')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📊 ROC curve kaydedildi: {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model") -> plt.Figure:
        """
        Precision-Recall Curve çiz
        
        Args:
            y_true: Gerçek etiketler
            y_pred_proba: Tahmin olasılıkları
            model_name: Model adı
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        ax.plot(recall, precision, color='darkred', lw=2,
               label=f'Precision-Recall (AP = {avg_precision:.4f})')
        
        # Baseline (random)
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                  label=f'Random Classifier (AP = {baseline:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{model_name} - Precision-Recall Curve', fontsize=16, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            import os
            os.makedirs(self.figures_path, exist_ok=True)
            save_path = os.path.join(self.figures_path, f'{model_name.lower()}_pr_curve.{self.plot_format}')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📈 PR curve kaydedildi: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str],
                              model_name: str = "Model", top_n: int = 20) -> Optional[plt.Figure]:
        """
        Feature importance çiz
        
        Args:
            model: Trained model
            feature_names: Özellik isimleri
            model_name: Model adı
            top_n: Gösterilecek özellik sayısı
            
        Returns:
            plt.Figure: Matplotlib figure (varsa)
        """
        # Feature importance'ı al
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            print(f"⚠️ {model_name} feature importance desteklemiyor")
            return None
        
        if importance is None:
            return None
        
        # En önemli özellikleri seç
        indices = np.argsort(importance)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(max(self.figure_size), 10))
        
        # Horizontal bar plot
        y_pos = np.arange(len(indices))
        ax.barh(y_pos, importance[indices], color='skyblue', alpha=0.8)
        
        # Özellik isimlerini ayarla
        feature_labels = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                         for i in indices]
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'{model_name} - Feature Importance (Top {top_n})', 
                    fontsize=16, fontweight='bold')
        
        # Değerleri bar'ların üzerine yaz
        for i, (idx, score) in enumerate(zip(indices, importance[indices])):
            ax.text(score + score*0.01, i, f'{score:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if self.save_plots:
            import os
            os.makedirs(self.figures_path, exist_ok=True)
            save_path = os.path.join(self.figures_path, f'{model_name.lower()}_feature_importance.{self.plot_format}')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📊 Feature importance kaydedildi: {save_path}")
        
        return fig
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict]) -> plt.Figure:
        """
        Model karşılaştırma grafiği
        
        Args:
            results_dict: Model sonuçları sözlüğü
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Metrikleri topla
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        if 'roc_auc' in results_dict[models[0]]:
            metrics.append('roc_auc')
        
        # DataFrame oluştur
        data = []
        for model_name in models:
            for metric in metrics:
                if metric in results_dict[model_name]:
                    data.append({
                        'Model': model_name,
                        'Metric': metric.upper().replace('_', '-'),
                        'Score': results_dict[model_name][metric]
                    })
        
        df = pd.DataFrame(data)
        
        # Grouped bar plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        sns.barplot(data=df, x='Metric', y='Score', hue='Model', ax=ax)
        
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        # Değerleri bar'ların üzerine yaz
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9, rotation=90)
        
        plt.tight_layout()
        
        if self.save_plots:
            import os
            os.makedirs(self.figures_path, exist_ok=True)
            save_path = os.path.join(self.figures_path, f'model_comparison.{self.plot_format}')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📊 Model comparison kaydedildi: {save_path}")
        
        return fig
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: Optional[List[str]] = None,
                      model_name: str = "Model", 
                      create_all_plots: bool = True) -> Dict:
        """
        Kapsamlı model değerlendirme
        
        Args:
            model: Eğitilmiş model
            X_test: Test özellikleri
            y_test: Test etiketleri
            feature_names: Özellik isimleri
            model_name: Model adı
            create_all_plots: Tüm grafikleri oluştur
            
        Returns:
            Dict: Değerlendirme sonuçları
        """
        print(f"\n🔍 {model_name} - KAPSAMLI DEĞERLENDİRME")
        print("=" * 60)
        
        # Tahminler
        y_pred = model.predict(X_test)
        
        # Probability predictions (varsa)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            # SVM için decision function'ı probability'e çevir
            decision_scores = model.decision_function(X_test)
            y_pred_proba = 1 / (1 + np.exp(-decision_scores))  # Sigmoid
        
        # Metrikleri hesapla
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba, model_name)
        
        # Metrikleri yazdır
        self.print_metrics(metrics, model_name)
        
        # Grafikleri oluştur
        figures = {}
        
        if create_all_plots:
            print(f"\n📊 {model_name} - Görselleştirmeler oluşturuluyor...")
            
            # Confusion Matrix
            figures['confusion_matrix'] = self.plot_confusion_matrix(metrics, model_name)
            
            # ROC Curve (probability varsa)
            if y_pred_proba is not None:
                figures['roc_curve'] = self.plot_roc_curve(y_test, y_pred_proba, model_name)
                figures['pr_curve'] = self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
            
            # Feature Importance (varsa)
            if feature_names is not None:
                feature_fig = self.plot_feature_importance(model, feature_names, model_name)
                if feature_fig is not None:
                    figures['feature_importance'] = feature_fig
        
        # Sonuçları kaydet
        evaluation_results = {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': {
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
            },
            'figures': figures
        }
        
        self.results[model_name] = evaluation_results
        
        return evaluation_results
    
    def evaluate_multiple_models(self, models_dict: Dict[str, Any], 
                                X_test: np.ndarray, y_test: np.ndarray,
                                feature_names: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Birden fazla modeli değerlendir
        
        Args:
            models_dict: Model sözlüğü {name: model}
            X_test: Test özellikleri
            y_test: Test etiketleri
            feature_names: Özellik isimleri
            
        Returns:
            Dict: Tüm model değerlendirme sonuçları
        """
        print("\n🔍 ÇOKLu MODEL DEĞERLENDİRMESİ")
        print("=" * 70)
        
        all_results = {}
        all_metrics = {}
        
        # Her modeli değerlendir
        for model_name, model in models_dict.items():
            result = self.evaluate_model(
                model, X_test, y_test, feature_names, model_name, create_all_plots=True
            )
            all_results[model_name] = result
            all_metrics[model_name] = result['metrics']
        
        # Model karşılaştırma grafiği
        if len(models_dict) > 1:
            print("\n📊 Model karşılaştırma grafiği oluşturuluyor...")
            comparison_fig = self.plot_model_comparison(all_metrics)
            
            # Karşılaştırma tablosu
            self.create_comparison_table(all_metrics)
        
        return all_results
    
    def create_comparison_table(self, results_dict: Dict[str, Dict]):
        """
        Model karşılaştırma tablosu oluştur
        
        Args:
            results_dict: Model sonuçları sözlüğü
        """
        print("\n📋 MODEL KARŞILAŞTIRMA TABLOSU")
        print("=" * 80)
        
        # DataFrame oluştur
        data = []
        for model_name, metrics in results_dict.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}"
            }
            
            if 'roc_auc' in metrics:
                row['ROC-AUC'] = f"{metrics['roc_auc']:.4f}"
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        print(df.to_string(index=False))
        
        # En iyi modeli bul
        numeric_df = df.copy()
        for col in df.columns[1:]:  # Model sütunu hariç
            numeric_df[col] = pd.to_numeric(numeric_df[col])
        
        best_model_idx = numeric_df['F1-Score'].idxmax()
        best_model = df.iloc[best_model_idx]['Model']
        
        print(f"\n🏆 EN İYİ MODEL: {best_model}")
        print(f"🎯 F1-Score: {numeric_df.iloc[best_model_idx]['F1-Score']:.4f}")
        
        # Tabloyu CSV olarak kaydet
        if self.save_plots:
            import os
            os.makedirs(self.reports_path, exist_ok=True)
            csv_path = os.path.join(self.reports_path, 'model_comparison.csv')
            df.to_csv(csv_path, index=False)
            print(f"💾 Karşılaştırma tablosu kaydedildi: {csv_path}")
    
    def save_evaluation_report(self, results_dict: Dict[str, Dict], 
                             report_name: str = "evaluation_report"):
        """
        Değerlendirme raporunu kaydet
        
        Args:
            results_dict: Değerlendirme sonuçları
            report_name: Rapor dosya adı
        """
        import os
        import json
        from datetime import datetime
        
        os.makedirs(self.reports_path, exist_ok=True)
        
        # JSON raporu
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': results_dict
        }
        
        # JSON dosyası kaydet
        json_path = os.path.join(self.reports_path, f'{report_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Değerlendirme raporu kaydedildi: {json_path}")
        
        # Markdown raporu
        self.create_markdown_report(results_dict, report_name)
    
    def create_markdown_report(self, results_dict: Dict[str, Dict], 
                             report_name: str = "evaluation_report"):
        """
        Markdown formatında rapor oluştur
        
        Args:
            results_dict: Değerlendirme sonuçları
            report_name: Rapor dosya adı
        """
        import os
        from datetime import datetime
        
        md_content = f"""# Fake News Detection - Model Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report contains the evaluation results of multiple machine learning models trained for fake news detection using the LIAR dataset.

## Model Performance Summary

"""
        
        # Her model için sonuçları ekle
        for model_name, result in results_dict.items():
            metrics = result['metrics']
            
            md_content += f"""### {model_name}

| Metric | Score |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-Score | {metrics['f1_score']:.4f} |
| Specificity | {metrics['specificity']:.4f} |
"""
            
            if 'roc_auc' in metrics:
                md_content += f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n"
            
            # Confusion Matrix
            cm = np.array(metrics['confusion_matrix'])
            md_content += f"""
**Confusion Matrix:**
```
     Pred
True  Fake  Real
Fake   {cm[0,0]:4d}  {cm[0,1]:4d}
Real   {cm[1,0]:4d}  {cm[1,1]:4d}
```

"""
        
        # En iyi model
        best_f1 = 0
        best_model = ""
        
        for model_name, result in results_dict.items():
            f1_score = result['metrics']['f1_score']
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model = model_name
        
        md_content += f"""## Best Model

🏆 **{best_model}** achieved the highest F1-Score of **{best_f1:.4f}**

## Conclusions

Based on the evaluation results, the {best_model} model performs best for fake news detection on this dataset.

---
*Report generated by Fake News Detection System*
"""
        
        # Markdown dosyasını kaydet
        md_path = os.path.join(self.reports_path, f'{report_name}.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"📄 Markdown raporu kaydedildi: {md_path}")

# Test fonksiyonu
def main():
    """Test fonksiyonu"""
    print("🧪 ModelEvaluator test ediliyor...")
    
    # Dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    
    # Config
    config = {
        'evaluation': {
            'plot_settings': {'figure_size': [10, 6], 'save_plots': False}
        },
        'output': {
            'figures_path': 'test_figures/',
            'reports_path': 'test_reports/'
        }
    }
    
    # Evaluator
    evaluator = ModelEvaluator(config)
    
    # Metrik testi
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba, "TestModel")
    evaluator.print_metrics(metrics, "TestModel")
    
    print("✅ ModelEvaluator test tamamlandı!")

if __name__ == "__main__":
    main()