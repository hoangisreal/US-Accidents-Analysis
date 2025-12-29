# -*- coding: utf-8 -*-
"""
Data Loader Module

Module này chứa class DataLoader để load và khám phá dữ liệu từ CSV file.
Sử dụng cho bài toán dự đoán mức độ nghiêm trọng tai nạn giao thông.
"""

import pandas as pd
import numpy as np


class DataLoader:
    """
    Lớp load và khám phá dữ liệu từ CSV file.
    
    Attributes:
        file_path (str): Đường dẫn đến file CSV
        df (pd.DataFrame): DataFrame chứa dữ liệu đã load
    """
    
    def __init__(self, file_path: str):
        """
        Khởi tạo DataLoader với đường dẫn file CSV.
        
        Args:
            file_path: Đường dẫn đến file CSV chứa dữ liệu tai nạn
        """
        self.file_path = file_path
        self.df = None
    
    def load_data(self, nrows: int = None) -> pd.DataFrame:
        """
        Load dữ liệu từ CSV file.
        
        Args:
            nrows: Số dòng cần load. None = load tất cả dữ liệu.
                   Khuyến nghị dùng 500000 để tiết kiệm bộ nhớ.
        
        Returns:
            DataFrame chứa dữ liệu đã load
            
        Raises:
            FileNotFoundError: Nếu file không tồn tại
        """
        print(f"Đang load dữ liệu từ: {self.file_path}")
        
        # Load CSV với số dòng chỉ định
        self.df = pd.read_csv(self.file_path, nrows=nrows)
        
        print(f"Đã load thành công {len(self.df):,} dòng dữ liệu")
        return self.df
    
    def get_basic_info(self) -> dict:
        """
        Lấy thông tin cơ bản về dataset.
        
        Returns:
            Dict chứa các thông tin:
            - shape: Kích thước (rows, columns)
            - columns: Danh sách tên cột
            - dtypes: Kiểu dữ liệu của từng cột
            - missing_values: Số lượng và tỷ lệ missing values
            - memory_usage: Dung lượng bộ nhớ sử dụng
        """
        if self.df is None:
            raise ValueError("Chưa load dữ liệu. Hãy gọi load_data() trước.")
        
        # Tính missing values
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df) * 100).round(2)
        
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': pd.DataFrame({
                'count': missing_count,
                'percent': missing_percent
            }).sort_values('percent', ascending=False),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        return info
    
    def get_target_distribution(self) -> pd.Series:
        """
        Lấy phân bố của biến target (Severity).
        
        Returns:
            Series chứa value counts của Severity với tỷ lệ phần trăm
        """
        if self.df is None:
            raise ValueError("Chưa load dữ liệu. Hãy gọi load_data() trước.")
        
        if 'Severity' not in self.df.columns:
            raise KeyError("Không tìm thấy cột 'Severity' trong dataset")
        
        # Đếm số lượng mỗi class
        counts = self.df['Severity'].value_counts().sort_index()
        
        # Tính tỷ lệ phần trăm
        percentages = (counts / len(self.df) * 100).round(2)
        
        # Tạo DataFrame kết quả
        distribution = pd.DataFrame({
            'Count': counts,
            'Percentage': percentages
        })
        
        print("\n=== Phân bố Severity ===")
        print(distribution)
        print(f"\nTổng số mẫu: {len(self.df):,}")
        
        return distribution
    
    def display_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Hiển thị một số dòng mẫu từ dataset.
        
        Args:
            n: Số dòng cần hiển thị
            
        Returns:
            DataFrame chứa n dòng đầu tiên
        """
        if self.df is None:
            raise ValueError("Chưa load dữ liệu. Hãy gọi load_data() trước.")
        
        return self.df.head(n)
    
    def get_numerical_stats(self) -> pd.DataFrame:
        """
        Lấy thống kê mô tả cho các cột số.
        
        Returns:
            DataFrame chứa thống kê (count, mean, std, min, max, etc.)
        """
        if self.df is None:
            raise ValueError("Chưa load dữ liệu. Hãy gọi load_data() trước.")
        
        return self.df.describe()
    
    def get_categorical_stats(self) -> dict:
        """
        Lấy thống kê cho các cột categorical.
        
        Returns:
            Dict với key là tên cột, value là value_counts
        """
        if self.df is None:
            raise ValueError("Chưa load dữ liệu. Hãy gọi load_data() trước.")
        
        # Xác định các cột categorical (object type)
        cat_columns = self.df.select_dtypes(include=['object']).columns
        
        stats = {}
        for col in cat_columns:
            stats[col] = {
                'unique_count': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().head(5)
            }
        
        return stats


# Test module nếu chạy trực tiếp
if __name__ == "__main__":
    # Test với dataset
    loader = DataLoader("dataset/US_Accidents_March23.csv")
    df = loader.load_data(nrows=10000)
    
    print("\n=== Thông tin cơ bản ===")
    info = loader.get_basic_info()
    print(f"Shape: {info['shape']}")
    print(f"Memory: {info['memory_usage']}")
    
    print("\n=== Missing Values (Top 10) ===")
    print(info['missing_values'].head(10))
    
    loader.get_target_distribution()
