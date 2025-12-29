# -*- coding: utf-8 -*-
"""
Accident Severity Prediction - Source Package

Package chứa các modules cho pipeline ML dự đoán mức độ nghiêm trọng tai nạn.

Modules:
    - data_loader: Load và khám phá dữ liệu
    - data_preprocessor: Tiền xử lý và làm sạch dữ liệu
    - feature_engineer: Tạo và biến đổi features
    - model_trainer: Huấn luyện và đánh giá models
    - visualizer: Tạo visualizations
"""

from .data_loader import DataLoader
from .data_preprocessor import DataPreprocessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .visualizer import Visualizer

__all__ = [
    'DataLoader',
    'DataPreprocessor', 
    'FeatureEngineer',
    'ModelTrainer',
    'Visualizer'
]

__version__ = '1.0.0'
