# -*- coding: utf-8 -*-
"""
Model Trainer Module

Module này chứa class ModelTrainer để huấn luyện và đánh giá các ML models.
Hỗ trợ: Logistic Regression, Random Forest, XGBoost.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class ModelTrainer:
    """
    Lớp huấn luyện và đánh giá các ML models.
    
    Attributes:
        models (dict): Dict chứa các model instances
        trained_models (dict): Dict chứa các model đã train
        results (dict): Dict chứa kết quả đánh giá
        best_model_name (str): Tên model tốt nhất
        best_model (object): Model tốt nhất
    """
    
    def __init__(self):
        """
        Khởi tạo ModelTrainer.
        """
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model_name = None
        self.best_model = None
    
    def get_models(self, class_ratio: float = 1.0) -> dict:
        """
        Khởi tạo các models với hyperparameters.
        
        Args:
            class_ratio: Tỷ lệ class 0 / class 1 để set scale_pos_weight cho XGBoost
            
        Returns:
            Dict chứa các model instances
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',  # Xử lý imbalanced data
                max_iter=1000,            # Tăng số iterations
                random_state=42,
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,         # Số cây
                class_weight='balanced',  # Xử lý imbalanced data
                max_depth=10,             # Giới hạn độ sâu để tránh overfit
                min_samples_split=5,
                random_state=42,
                n_jobs=-1                 # Sử dụng tất cả CPU cores
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=class_ratio,  # Xử lý imbalanced data
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
        }
        
        print(f"Đã khởi tạo {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self.models
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2) -> tuple:
        """
        Chia dữ liệu thành train và test sets (stratified).
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Tỷ lệ test set (default 0.2 = 20%)
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,           # Giữ tỷ lệ class trong train/test
            random_state=42
        )
        
        print(f"\nChia dữ liệu (stratified split):")
        print(f"  - Training set: {len(X_train):,} samples ({(1-test_size)*100:.0f}%)")
        print(f"  - Test set: {len(X_test):,} samples ({test_size*100:.0f}%)")
        
        # Kiểm tra phân bố class
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        print(f"\nPhân bố class trong Training set:")
        print(f"  - Class 0 (Mild): {train_dist.get(0, 0)*100:.1f}%")
        print(f"  - Class 1 (Severe): {train_dist.get(1, 0)*100:.1f}%")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, X_train, y_train, model_name: str = "Model"):
        """
        Huấn luyện một model.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            model_name: Tên model để hiển thị
            
        Returns:
            Trained model
        """
        print(f"\nĐang train {model_name}...")
        model.fit(X_train, y_train)
        print(f"  ✓ Hoàn thành training {model_name}")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name: str = "Model") -> dict:
        """
        Đánh giá model trên test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Tên model
            
        Returns:
            Dict chứa các metrics
        """
        # Dự đoán
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Tính các metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\n=== Kết quả {model_name} ===")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def train_and_evaluate_all(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Huấn luyện và đánh giá tất cả models.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            
        Returns:
            Dict chứa results của tất cả models
        """
        print("=" * 50)
        print("BẮT ĐẦU TRAINING VÀ EVALUATION")
        print("=" * 50)
        
        # Tính class ratio cho XGBoost
        class_counts = y_train.value_counts()
        class_ratio = class_counts[0] / class_counts[1] if 1 in class_counts.index else 1.0
        
        # Khởi tạo models
        self.get_models(class_ratio)
        
        # Train và evaluate từng model
        for name, model in self.models.items():
            # Train
            trained_model = self.train_model(model, X_train, y_train, name)
            self.trained_models[name] = trained_model
            
            # Evaluate
            metrics = self.evaluate_model(trained_model, X_test, y_test, name)
            self.results[name] = metrics
        
        # Tìm model tốt nhất
        self._find_best_model()
        
        print("\n" + "=" * 50)
        print("HOÀN THÀNH TRAINING VÀ EVALUATION")
        print("=" * 50)
        
        return self.results
    
    def _find_best_model(self):
        """
        Tìm model tốt nhất dựa trên F1-score.
        """
        best_f1 = 0
        for name, metrics in self.results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                self.best_model_name = name
                self.best_model = self.trained_models[name]
        
        print(f"\n★ Model tốt nhất: {self.best_model_name} (F1-Score: {best_f1:.4f})")
    
    def get_best_model(self) -> tuple:
        """
        Lấy model tốt nhất.
        
        Returns:
            Tuple (model_name, model_instance, metrics)
        """
        if self.best_model_name is None:
            raise ValueError("Chưa train models. Hãy gọi train_and_evaluate_all() trước.")
        
        return (
            self.best_model_name,
            self.best_model,
            self.results[self.best_model_name]
        )
    
    def get_feature_importance(self, model_name: str, feature_names: list) -> pd.DataFrame:
        """
        Lấy feature importance từ model.
        
        Args:
            model_name: Tên model (Random Forest hoặc XGBoost)
            feature_names: List tên features
            
        Returns:
            DataFrame với feature importance, sorted descending
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' chưa được train")
        
        model = self.trained_models[model_name]
        
        # Lấy feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Logistic Regression: dùng absolute value của coefficients
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model '{model_name}' không hỗ trợ feature importance")
        
        # Tạo DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Tạo bảng so sánh các models.
        
        Returns:
            DataFrame với metrics của tất cả models
        """
        comparison = []
        for name, metrics in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def save_model(self, model_name: str, filepath: str):
        """
        Lưu model ra file.
        
        Args:
            model_name: Tên model cần lưu
            filepath: Đường dẫn file (nên dùng .joblib)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' chưa được train")
        
        joblib.dump(self.trained_models[model_name], filepath)
        print(f"Đã lưu model '{model_name}' vào: {filepath}")
    
    def load_model(self, filepath: str) -> object:
        """
        Load model từ file.
        
        Args:
            filepath: Đường dẫn file
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"Đã load model từ: {filepath}")
        return model
    
    def print_classification_report(self, model_name: str, y_test):
        """
        In classification report chi tiết.
        
        Args:
            model_name: Tên model
            y_test: True labels
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' chưa được evaluate")
        
        y_pred = self.results[model_name]['y_pred']
        
        print(f"\n=== Classification Report: {model_name} ===")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Mild (0)', 'Severe (1)']))


# Test module nếu chạy trực tiếp
if __name__ == "__main__":
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from feature_engineer import FeatureEngineer
    
    # Load và preprocess
    loader = DataLoader("dataset/US_Accidents_March23.csv")
    df = loader.load_data(nrows=50000)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    X, y = engineer.transform(df_processed)
    
    # Train và evaluate
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    results = trainer.train_and_evaluate_all(X_train, X_test, y_train, y_test)
    
    # So sánh models
    print("\n=== Bảng so sánh Models ===")
    print(trainer.get_comparison_table())
    
    # Feature importance
    print("\n=== Feature Importance (Best Model) ===")
    best_name, best_model, best_metrics = trainer.get_best_model()
    importance = trainer.get_feature_importance(best_name, engineer.get_feature_names())
    print(importance.head(10))
