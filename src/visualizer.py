# -*- coding: utf-8 -*-
"""
Visualizer Module

Module này chứa class Visualizer để tạo các visualizations cho analysis và presentation.
Bao gồm: distribution plots, confusion matrix, ROC curves, feature importance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


class Visualizer:
    """
    Lớp tạo các visualizations cho analysis và presentation.
    
    Attributes:
        output_dir (str): Thư mục lưu plots
        figsize (tuple): Kích thước mặc định của figure
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Khởi tạo Visualizer.
        
        Args:
            output_dir: Thư mục lưu plots (tạo nếu chưa tồn tại)
        """
        self.output_dir = output_dir
        self.figsize = (10, 6)
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Cấu hình style cho plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_original_severity_distribution(self, severity: pd.Series, 
                                              title: str = 'Phân bố Severity Gốc (1-4)',
                                              save: bool = True):
        """
        Vẽ biểu đồ phân bố Severity gốc (4 classes: 1, 2, 3, 4).
        
        Args:
            severity: Series chứa giá trị Severity gốc (1-4)
            title: Tiêu đề biểu đồ
            save: True để lưu file
        """
        # Tính counts và percentages
        counts = severity.value_counts().sort_index()
        percentages = (counts / len(severity) * 100).round(2)
        
        # In text output
        print("\n" + "="*50)
        print(f"=== {title} ===")
        print("="*50)
        print(f"\nTổng số mẫu: {len(severity):,}")
        print("\nChi tiết phân bố:")
        print("-"*40)
        for sev in counts.index:
            print(f"  Severity {sev}: {counts[sev]:>10,} mẫu ({percentages[sev]:>6.2f}%)")
        print("-"*40)
        
        # Nhận xét về imbalanced
        max_pct = percentages.max()
        min_pct = percentages.min()
        print(f"\nNhận xét:")
        print(f"  - Class chiếm nhiều nhất: Severity {percentages.idxmax()} ({max_pct:.2f}%)")
        print(f"  - Class chiếm ít nhất: Severity {percentages.idxmin()} ({min_pct:.2f}%)")
        print(f"  - Tỷ lệ imbalance: {max_pct/min_pct:.1f}x")
        
        # Vẽ biểu đồ
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Màu sắc cho 4 levels (từ nhẹ đến nặng)
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        labels = [f'Severity {i}' for i in counts.index]
        
        # Bar chart
        bars = axes[0].bar(labels, counts.values, color=colors[:len(counts)])
        axes[0].set_xlabel('Mức độ Severity', fontsize=12)
        axes[0].set_ylabel('Số lượng', fontsize=12)
        axes[0].set_title('Số lượng theo Severity', fontsize=13, fontweight='bold')
        
        for bar, count, pct in zip(bars, counts.values, percentages.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                        f'{count:,}\n({pct}%)', ha='center', fontweight='bold', fontsize=10)
        
        axes[0].set_ylim(0, max(counts.values) * 1.15)
        
        # Pie chart
        explode = [0.02] * len(counts)
        wedges, texts, autotexts = axes[1].pie(
            counts.values, labels=labels, autopct='%1.2f%%',
            colors=colors[:len(counts)], startangle=90, explode=explode, shadow=True
        )
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        axes[1].set_title('Tỷ lệ phần trăm', fontsize=13, fontweight='bold')
        
        legend_labels = [f'{labels[i]}: {counts.values[i]:,} ({percentages.values[i]}%)' 
                        for i in range(len(counts))]
        axes[1].legend(wedges, legend_labels, title="Chi tiết", loc="center left", 
                      bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'original_severity_distribution.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()

    def plot_target_distribution(self, y: pd.Series, title: str = 'Phân bố Severity (Binary)',
                                  save: bool = True):
        """
        Vẽ biểu đồ phân bố target variable (binary: Mild/Severe).
        
        Args:
            y: Target Series (0 = Mild, 1 = Severe)
            title: Tiêu đề biểu đồ
            save: True để lưu file
        """
        # Tính counts và percentages
        counts = y.value_counts().sort_index()
        percentages = (counts / len(y) * 100).round(2)
        labels = ['Mild (0)', 'Severe (1)'] if len(counts) == 2 else counts.index.tolist()
        
        # In text output
        print("\n" + "="*50)
        print(f"=== {title} ===")
        print("="*50)
        print(f"\nTổng số mẫu: {len(y):,}")
        print("\nChi tiết phân bố:")
        print("-"*40)
        for i, (idx, count) in enumerate(counts.items()):
            label = labels[i] if i < len(labels) else f"Class {idx}"
            print(f"  {label}: {count:>10,} mẫu ({percentages[idx]:>6.2f}%)")
        print("-"*40)
        
        # Tính class ratio
        if len(counts) == 2:
            ratio = counts.iloc[0] / counts.iloc[1]
            print(f"\nClass ratio (Mild/Severe): {ratio:.2f}")
        
        # Vẽ biểu đồ
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ['#2ecc71', '#e74c3c']
        
        bars = axes[0].bar(labels, counts.values, color=colors[:len(counts)])
        axes[0].set_xlabel('Severity')
        axes[0].set_ylabel('Số lượng')
        axes[0].set_title('Số lượng theo Severity')
        
        for bar, count, pct in zip(bars, counts.values, percentages.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f'{count:,}\n({pct}%)', ha='center', fontweight='bold')
        
        axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%', 
                    colors=colors[:len(counts)], startangle=90, shadow=True)
        axes[1].set_title('Tỷ lệ phần trăm')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'target_distribution.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()
    
    def plot_missing_values(self, df: pd.DataFrame, top_n: int = 15, save: bool = True):
        """
        Vẽ biểu đồ missing values.
        
        Args:
            df: DataFrame cần visualize
            top_n: Số cột hiển thị (top missing)
            save: True để lưu file
        """
        # Tính missing values
        missing = df.isnull().sum()
        missing_percent = (missing / len(df) * 100).round(2)
        
        # Lọc cột có missing và sort
        missing_df = pd.DataFrame({
            'column': missing.index,
            'missing_count': missing.values,
            'missing_percent': missing_percent.values
        })
        missing_df = missing_df[missing_df['missing_count'] > 0]
        missing_df = missing_df.sort_values('missing_percent', ascending=False)
        
        # In text output
        print("\n" + "="*50)
        print("=== Phân tích Missing Values ===")
        print("="*50)
        print(f"\nTổng số dòng: {len(df):,}")
        print(f"Tổng số cột: {len(df.columns)}")
        print(f"Số cột có missing: {len(missing_df)}")
        
        if len(missing_df) > 0:
            print(f"\nTop {min(top_n, len(missing_df))} cột có missing values cao nhất:")
            print("-"*60)
            print(f"{'Cột':<25} {'Số lượng':>12} {'Tỷ lệ (%)':>12}")
            print("-"*60)
            for _, row in missing_df.head(top_n).iterrows():
                print(f"{row['column']:<25} {row['missing_count']:>12,} {row['missing_percent']:>11.2f}%")
            print("-"*60)
        else:
            print("\nKhông có missing values!")
            return
        
        # Vẽ biểu đồ
        plot_df = missing_df.head(top_n).sort_values('missing_percent', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(plot_df['column'], plot_df['missing_percent'], color='coral')
        
        ax.set_xlabel('Tỷ lệ Missing (%)')
        ax.set_title(f'Top {top_n} Missing Values theo Cột', fontsize=14, fontweight='bold')
        
        for bar, pct in zip(bars, plot_df['missing_percent']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'missing_values.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str, save: bool = True):
        """
        Vẽ confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Tên model
            save: True để lưu file
        """
        # Tính confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Tính các metrics từ confusion matrix
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        
        # In text output
        print("\n" + "="*50)
        print(f"=== Confusion Matrix: {model_name} ===")
        print("="*50)
        print(f"\nTổng số mẫu test: {total:,}")
        print("\nChi tiết Confusion Matrix:")
        print("-"*40)
        print(f"  True Negative (TN):  {tn:>8,} ({tn/total*100:.2f}%)")
        print(f"  False Positive (FP): {fp:>8,} ({fp/total*100:.2f}%)")
        print(f"  False Negative (FN): {fn:>8,} ({fn/total*100:.2f}%)")
        print(f"  True Positive (TP):  {tp:>8,} ({tp/total*100:.2f}%)")
        print("-"*40)
        
        # Tính thêm metrics
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMetrics từ Confusion Matrix:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Mild (0)', 'Severe (1)'],
                    yticklabels=['Mild (0)', 'Severe (1)'])
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()
    
    def plot_roc_curves(self, results: dict, y_test, save: bool = True):
        """
        Vẽ ROC curves cho tất cả models.
        
        Args:
            results: Dict chứa results với y_pred_proba
            y_test: True labels
            save: True để lưu file
        """
        # In text output
        print("\n" + "="*50)
        print("=== ROC Curves - So sánh Models ===")
        print("="*50)
        print(f"\nSố mẫu test: {len(y_test):,}")
        print("\nROC-AUC Score của các models:")
        print("-"*40)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
        
        auc_scores = {}
        for i, (name, metrics) in enumerate(results.items()):
            y_pred_proba = metrics['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            auc_scores[name] = roc_auc
            
            print(f"  {name:<25} AUC = {roc_auc:.4f}")
            
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{name} (AUC = {roc_auc:.3f})')
        
        print("-"*40)
        
        # Tìm model tốt nhất
        best_model = max(auc_scores, key=auc_scores.get)
        print(f"\n★ Model có AUC cao nhất: {best_model} ({auc_scores[best_model]:.4f})")
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - So sánh Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'roc_curves.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()

    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                                 model_name: str, top_n: int = 10, save: bool = True):
        """
        Vẽ biểu đồ feature importance.
        
        Args:
            importance_df: DataFrame với columns ['feature', 'importance']
            model_name: Tên model
            top_n: Số features hiển thị
            save: True để lưu file
        """
        # In text output
        print("\n" + "="*50)
        print(f"=== Feature Importance ({model_name}) ===")
        print("="*50)
        print(f"\nTop {top_n} features quan trọng nhất:")
        print("-"*50)
        print(f"{'Rank':<6} {'Feature':<25} {'Importance':>12}")
        print("-"*50)
        
        for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
            print(f"{i+1:<6} {row['feature']:<25} {row['importance']:>12.4f}")
        
        print("-"*50)
        
        # Tính tổng importance của top features
        total_importance = importance_df.head(top_n)['importance'].sum()
        print(f"\nTổng importance của top {top_n}: {total_importance:.4f} ({total_importance*100:.1f}%)")
        
        # Vẽ biểu đồ
        top_features = importance_df.head(top_n).sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(top_features['feature'], top_features['importance'], color='steelblue')
        
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}', 
                     fontsize=14, fontweight='bold')
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save: bool = True):
        """
        Vẽ biểu đồ so sánh metrics giữa các models.
        
        Args:
            comparison_df: DataFrame với metrics của các models
            save: True để lưu file
        """
        # In text output
        print("\n" + "="*60)
        print("=== So sánh Performance giữa các Models ===")
        print("="*60)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        models = comparison_df['Model'].tolist()
        
        print(f"\n{'Model':<20}", end='')
        for metric in metrics:
            print(f"{metric:>12}", end='')
        print()
        print("-"*80)
        
        for _, row in comparison_df.iterrows():
            print(f"{row['Model']:<20}", end='')
            for metric in metrics:
                print(f"{row[metric]:>12.4f}", end='')
            print()
        
        print("-"*80)
        
        # Tìm model tốt nhất theo F1-Score
        best_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.loc[best_idx, 'Model']
        best_f1 = comparison_df.loc[best_idx, 'F1-Score']
        print(f"\n★ Model tốt nhất (theo F1-Score): {best_model} ({best_f1:.4f})")
        
        # Vẽ biểu đồ
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for i, model in enumerate(models):
            values = comparison_df[comparison_df['Model'] == model][metrics].values[0]
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, color=colors[i % len(colors)])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('Score')
        ax.set_title('So sánh Performance giữa các Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()

    def plot_correlation_matrix(self, df: pd.DataFrame, features: list = None, 
                                 save: bool = True):
        """
        Vẽ correlation matrix của các features.
        
        Args:
            df: DataFrame
            features: List features cần visualize (None = tất cả numerical)
            save: True để lưu file
        """
        # Chọn features
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Tính correlation matrix
        corr = df[features].corr()
        
        # In text output
        print("\n" + "="*50)
        print("=== Correlation Matrix ===")
        print("="*50)
        print(f"\nSố features: {len(features)}")
        
        # Tìm các cặp có correlation cao
        print("\nTop 10 cặp features có correlation cao nhất (|r| > 0.3):")
        print("-"*60)
        
        corr_pairs = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                r = corr.iloc[i, j]
                if abs(r) > 0.3:
                    corr_pairs.append((features[i], features[j], r))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        if corr_pairs:
            print(f"{'Feature 1':<20} {'Feature 2':<20} {'Correlation':>12}")
            print("-"*60)
            for f1, f2, r in corr_pairs[:10]:
                print(f"{f1:<20} {f2:<20} {r:>12.4f}")
        else:
            print("Không có cặp features nào có |correlation| > 0.3")
        
        print("-"*60)
        
        # Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, square=True, linewidths=0.5)
        
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'correlation_matrix.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()
    
    def plot_numerical_distributions(self, df: pd.DataFrame, columns: list, 
                                      save: bool = True):
        """
        Vẽ distribution của các numerical features.
        
        Args:
            df: DataFrame
            columns: List tên cột cần visualize
            save: True để lưu file
        """
        # In text output
        print("\n" + "="*50)
        print("=== Phân bố các Numerical Features ===")
        print("="*50)
        print(f"\nSố features: {len(columns)}")
        print("\nThống kê mô tả:")
        print("-"*70)
        print(f"{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-"*70)
        
        for col in columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"{col:<20} {mean:>12.2f} {std:>12.2f} {min_val:>12.2f} {max_val:>12.2f}")
        
        print("-"*70)
        
        # Vẽ biểu đồ
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes
        
        for i, col in enumerate(columns):
            if col in df.columns:
                sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
                axes[i].set_title(col)
                axes[i].set_xlabel('')
        
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Phân bố các Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'numerical_distributions.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"\nĐã lưu: {filepath}")
        
        plt.show()


# Test module nếu chạy trực tiếp
if __name__ == "__main__":
    # Test với sample data
    viz = Visualizer()
    
    # Test target distribution
    y = pd.Series([0]*6000 + [1]*4000)
    viz.plot_target_distribution(y, save=False)
    
    print("\nVisualizer module test completed!")
