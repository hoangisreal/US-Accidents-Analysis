# -*- coding: utf-8 -*-
"""
Data Preprocessor Module

Module này chứa class DataPreprocessor để tiền xử lý và làm sạch dữ liệu.
Bao gồm: xử lý missing values, loại bỏ duplicates, chuyển đổi target variable.
"""

import pandas as pd
import numpy as np


class DataPreprocessor:
    """
    Lớp tiền xử lý dữ liệu: cleaning, imputation, transformation.
    
    Attributes:
        columns_to_drop (list): Danh sách cột cần xóa (missing >50% hoặc không cần thiết)
        numerical_columns (list): Danh sách cột số cần impute
        categorical_columns (list): Danh sách cột categorical cần impute
    """
    
    def __init__(self):
        """
        Khởi tạo DataPreprocessor với cấu hình mặc định.
        """
        # Các cột cần xóa: missing >50% hoặc không cần thiết cho model
        self.columns_to_drop = [
            'End_Lat',           # 100% missing
            'End_Lng',           # 100% missing
            'Wind_Chill(F)',     # ~96% missing
            'Precipitation(in)', # ~93% missing
            'ID',                # Identifier, không cần cho model
            'Description',       # Text, phức tạp để xử lý
            'Source',            # Không liên quan đến severity
            'Country',           # Chỉ có 1 giá trị (US)
            'Zipcode',           # Quá nhiều unique values
            'Airport_Code',      # Không trực tiếp liên quan
            'Weather_Timestamp', # Timestamp riêng, không cần
            'End_Time',          # Có thể tính từ Start_Time + Duration
            'Street',            # Quá nhiều unique values
            'City',              # Quá nhiều unique values  
            'County',            # Quá nhiều unique values
            'Timezone',          # Có thể suy từ location
            'Number',            # Số nhà, không cần
            'Wind_Direction',    # Có thể bỏ qua
            'Turning_Loop',      # Gần như toàn bộ False
            'Roundabout',        # Rất ít True
            'Civil_Twilight',    # Tương tự Sunrise_Sunset
            'Nautical_Twilight', # Tương tự Sunrise_Sunset
            'Astronomical_Twilight', # Tương tự Sunrise_Sunset
            'Weather_Condition', # Có thể bỏ để đơn giản hóa
            'Pressure(in)',      # Ít ảnh hưởng
        ]
        
        # Các cột số cần impute với median
        self.numerical_columns = [
            'Temperature(F)',
            'Humidity(%)',
            'Visibility(mi)',
            'Wind_Speed(mph)',
            'Distance(mi)',
            'Start_Lat',
            'Start_Lng'
        ]
        
        # Các cột categorical cần impute với mode
        self.categorical_columns = [
            'Sunrise_Sunset',
            'State'
        ]
        
        # Lưu giá trị impute để dùng cho test data
        self.impute_values = {}
    
    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xóa các cột không cần thiết hoặc có quá nhiều missing values.
        
        Args:
            df: DataFrame gốc
            
        Returns:
            DataFrame đã xóa các cột không cần thiết
        """
        # Chỉ xóa các cột tồn tại trong DataFrame
        cols_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        
        print(f"Xóa {len(cols_to_drop)} cột không cần thiết...")
        df_cleaned = df.drop(columns=cols_to_drop, errors='ignore')
        
        print(f"Còn lại {len(df_cleaned.columns)} cột")
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xóa các dòng trùng lặp.
        
        Args:
            df: DataFrame gốc
            
        Returns:
            DataFrame không có duplicates
        """
        original_len = len(df)
        df_cleaned = df.drop_duplicates()
        removed = original_len - len(df_cleaned)
        
        print(f"Đã xóa {removed:,} dòng trùng lặp")
        return df_cleaned
    
    def impute_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Điền giá trị missing: numerical với median, categorical với mode.
        
        Args:
            df: DataFrame có missing values
            fit: True nếu cần tính toán giá trị impute (training data)
                 False nếu dùng giá trị đã tính (test data)
            
        Returns:
            DataFrame đã impute missing values
        """
        df_imputed = df.copy()
        
        # Impute numerical columns với median
        for col in self.numerical_columns:
            if col in df_imputed.columns:
                if fit:
                    # Tính median từ training data
                    self.impute_values[col] = df_imputed[col].median()
                
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    df_imputed[col].fillna(self.impute_values.get(col, df_imputed[col].median()), inplace=True)
                    print(f"  - {col}: điền {missing_count:,} missing với median = {self.impute_values.get(col, 0):.2f}")
        
        # Impute categorical columns với mode
        for col in self.categorical_columns:
            if col in df_imputed.columns:
                if fit:
                    # Lấy mode từ training data
                    mode_val = df_imputed[col].mode()
                    self.impute_values[col] = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    df_imputed[col].fillna(self.impute_values.get(col, 'Unknown'), inplace=True)
                    print(f"  - {col}: điền {missing_count:,} missing với mode = '{self.impute_values.get(col)}'")
        
        return df_imputed
    
    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển Severity từ 4 class sang binary classification.
        
        Mapping:
            - Severity 1, 2 -> 0 (Mild - Nhẹ)
            - Severity 3, 4 -> 1 (Severe - Nghiêm trọng)
        
        Args:
            df: DataFrame với Severity gốc (1-4)
            
        Returns:
            DataFrame với Severity binary (0/1)
        """
        if 'Severity' not in df.columns:
            raise KeyError("Không tìm thấy cột 'Severity' trong DataFrame")
        
        df_transformed = df.copy()
        
        # Lưu phân bố gốc
        original_dist = df_transformed['Severity'].value_counts().sort_index()
        print("\nPhân bố Severity gốc:")
        print(original_dist)
        
        # Chuyển đổi: 1,2 -> 0 (Mild), 3,4 -> 1 (Severe)
        df_transformed['Severity'] = df_transformed['Severity'].apply(
            lambda x: 0 if x in [1, 2] else 1
        )
        
        # Hiển thị phân bố mới
        new_dist = df_transformed['Severity'].value_counts().sort_index()
        print("\nPhân bố Severity sau chuyển đổi (Binary):")
        print(f"  0 (Mild):   {new_dist.get(0, 0):,} ({new_dist.get(0, 0)/len(df_transformed)*100:.1f}%)")
        print(f"  1 (Severe): {new_dist.get(1, 0):,} ({new_dist.get(1, 0)/len(df_transformed)*100:.1f}%)")
        
        return df_transformed
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Thực hiện toàn bộ preprocessing pipeline.
        
        Pipeline:
            1. Xóa cột không cần thiết
            2. Xóa duplicates
            3. Impute missing values
            4. Chuyển đổi target variable
        
        Args:
            df: DataFrame gốc
            fit: True cho training data, False cho test data
            
        Returns:
            DataFrame đã preprocess hoàn chỉnh
        """
        print("=" * 50)
        print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
        print("=" * 50)
        
        print(f"\nDữ liệu ban đầu: {df.shape[0]:,} dòng, {df.shape[1]} cột")
        
        # Bước 1: Xóa cột không cần thiết
        print("\n[1/4] Xóa cột không cần thiết...")
        df_processed = self.drop_unnecessary_columns(df)
        
        # Bước 2: Xóa duplicates
        print("\n[2/4] Xóa dòng trùng lặp...")
        df_processed = self.remove_duplicates(df_processed)
        
        # Bước 3: Impute missing values
        print("\n[3/4] Điền missing values...")
        df_processed = self.impute_missing_values(df_processed, fit=fit)
        
        # Bước 4: Chuyển đổi target
        print("\n[4/4] Chuyển đổi target variable...")
        df_processed = self.transform_target(df_processed)
        
        print("\n" + "=" * 50)
        print(f"HOÀN THÀNH TIỀN XỬ LÝ")
        print(f"Dữ liệu sau xử lý: {df_processed.shape[0]:,} dòng, {df_processed.shape[1]} cột")
        print("=" * 50)
        
        return df_processed
    
    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo bảng tóm tắt missing values.
        
        Args:
            df: DataFrame cần kiểm tra
            
        Returns:
            DataFrame với thông tin missing values
        """
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df) * 100).round(2)
        
        summary = pd.DataFrame({
            'Missing Count': missing_count,
            'Missing %': missing_percent
        })
        
        # Chỉ hiển thị cột có missing
        summary = summary[summary['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        return summary


# Test module nếu chạy trực tiếp
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load dữ liệu
    loader = DataLoader("dataset/US_Accidents_March23.csv")
    df = loader.load_data(nrows=50000)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    
    # Kiểm tra missing values sau xử lý
    print("\n=== Missing values sau xử lý ===")
    missing = preprocessor.get_missing_summary(df_processed)
    if len(missing) == 0:
        print("Không còn missing values!")
    else:
        print(missing)
