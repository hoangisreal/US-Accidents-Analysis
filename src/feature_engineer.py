# -*- coding: utf-8 -*-
"""
Feature Engineer Module

Module này chứa class FeatureEngineer để tạo và biến đổi features cho model.
Bao gồm: trích xuất time features, encoding, scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineer:
    """
    Lớp tạo và biến đổi features cho model.
    
    Attributes:
        numerical_features (list): Danh sách features số cần scale
        boolean_features (list): Danh sách features boolean
        categorical_features (list): Danh sách features categorical cần encode
        scaler (StandardScaler): Scaler cho numerical features
        label_encoders (dict): Dict chứa LabelEncoder cho từng categorical feature
    """
    
    def __init__(self):
        """
        Khởi tạo FeatureEngineer với cấu hình features.
        """
        # Features số cần scale
        self.numerical_features = [
            'Distance(mi)',
            'Temperature(F)',
            'Humidity(%)',
            'Visibility(mi)',
            'Start_Lat',
            'Start_Lng'
        ]
        
        # Features boolean (sẽ convert sang 0/1)
        self.boolean_features = [
            'Traffic_Signal',
            'Junction',
            'Crossing',
            'Stop',
            'Amenity',
            'Bump',
            'Give_Way',
            'No_Exit',
            'Railway',
            'Station'
        ]
        
        # Features categorical cần encode
        self.categorical_features = [
            'Sunrise_Sunset',
            'State'
        ]
        
        # Time features sẽ được tạo
        self.time_features = ['hour', 'day_of_week']
        
        # Scaler và encoders
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Danh sách features cuối cùng
        self.final_features = []
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất features từ cột Start_Time.
        
        Tạo các features:
            - hour: Giờ trong ngày (0-23)
            - day_of_week: Ngày trong tuần (0=Monday, 6=Sunday)
        
        Args:
            df: DataFrame với cột Start_Time
            
        Returns:
            DataFrame với features thời gian mới
        """
        df_time = df.copy()
        
        if 'Start_Time' not in df_time.columns:
            print("Cảnh báo: Không tìm thấy cột 'Start_Time'")
            return df_time
        
        # Convert sang datetime
        df_time['Start_Time'] = pd.to_datetime(df_time['Start_Time'], errors='coerce')
        
        # Trích xuất hour và day_of_week
        df_time['hour'] = df_time['Start_Time'].dt.hour
        df_time['day_of_week'] = df_time['Start_Time'].dt.dayofweek
        
        # Xóa cột Start_Time gốc
        df_time = df_time.drop(columns=['Start_Time'])
        
        print(f"Đã tạo time features: hour (0-23), day_of_week (0-6)")
        
        return df_time
    
    def convert_boolean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuyển đổi boolean features sang integer (0/1).
        
        Args:
            df: DataFrame với boolean features
            
        Returns:
            DataFrame với boolean features đã convert
        """
        df_bool = df.copy()
        
        for col in self.boolean_features:
            if col in df_bool.columns:
                # Convert True/False sang 1/0
                df_bool[col] = df_bool[col].astype(int)
        
        print(f"Đã convert {len([c for c in self.boolean_features if c in df_bool.columns])} boolean features sang 0/1")
        
        return df_bool
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features bằng LabelEncoder.
        
        Args:
            df: DataFrame với categorical features
            fit: True nếu fit encoder (training), False nếu chỉ transform (test)
            
        Returns:
            DataFrame với features đã encode
        """
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            if col in df_encoded.columns:
                if fit:
                    # Fit và transform cho training data
                    self.label_encoders[col] = LabelEncoder()
                    # Xử lý missing values trước khi encode
                    df_encoded[col] = df_encoded[col].fillna('Unknown')
                    self.label_encoders[col].fit(df_encoded[col].astype(str))
                
                # Transform
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                print(f"  - {col}: encoded {len(self.label_encoders[col].classes_)} classes")
        
        return df_encoded
    
    def scale_numerical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features bằng StandardScaler.
        
        Sau khi scale: mean ≈ 0, std ≈ 1
        
        Args:
            df: DataFrame với numerical features
            fit: True nếu fit scaler (training), False nếu chỉ transform (test)
            
        Returns:
            DataFrame với features đã scale
        """
        df_scaled = df.copy()
        
        # Lọc các cột numerical có trong DataFrame
        cols_to_scale = [col for col in self.numerical_features if col in df_scaled.columns]
        
        if len(cols_to_scale) == 0:
            print("Cảnh báo: Không có numerical features để scale")
            return df_scaled
        
        if fit:
            # Fit và transform cho training data
            df_scaled[cols_to_scale] = self.scaler.fit_transform(df_scaled[cols_to_scale])
            print(f"Đã fit và scale {len(cols_to_scale)} numerical features")
        else:
            # Chỉ transform cho test data
            df_scaled[cols_to_scale] = self.scaler.transform(df_scaled[cols_to_scale])
            print(f"Đã scale {len(cols_to_scale)} numerical features (dùng scaler đã fit)")
        
        return df_scaled
    
    def select_features(self, df: pd.DataFrame) -> tuple:
        """
        Chọn các features cần thiết cho model và tách X, y.
        
        Args:
            df: DataFrame đã engineer
            
        Returns:
            Tuple (X, y) với X là features DataFrame, y là target Series
        """
        # Xác định tất cả features cần dùng
        all_features = []
        
        # Numerical features
        for col in self.numerical_features:
            if col in df.columns:
                all_features.append(col)
        
        # Boolean features
        for col in self.boolean_features:
            if col in df.columns:
                all_features.append(col)
        
        # Categorical features (đã encode)
        for col in self.categorical_features:
            if col in df.columns:
                all_features.append(col)
        
        # Time features
        for col in self.time_features:
            if col in df.columns:
                all_features.append(col)
        
        # Lưu danh sách features cuối cùng
        self.final_features = all_features
        
        # Tách X và y
        X = df[all_features].copy()
        y = df['Severity'].copy() if 'Severity' in df.columns else None
        
        print(f"\nĐã chọn {len(all_features)} features cho model:")
        print(f"  - Numerical: {len([c for c in self.numerical_features if c in df.columns])}")
        print(f"  - Boolean: {len([c for c in self.boolean_features if c in df.columns])}")
        print(f"  - Categorical: {len([c for c in self.categorical_features if c in df.columns])}")
        print(f"  - Time: {len([c for c in self.time_features if c in df.columns])}")
        
        return X, y
    
    def get_feature_names(self) -> list:
        """
        Lấy danh sách tên features cuối cùng.
        
        Returns:
            List tên features
        """
        return self.final_features
    
    def transform(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        """
        Thực hiện toàn bộ feature engineering pipeline.
        
        Pipeline:
            1. Trích xuất time features
            2. Convert boolean features
            3. Encode categorical features
            4. Scale numerical features
            5. Chọn features cuối cùng
        
        Args:
            df: DataFrame đã preprocess
            fit: True cho training data, False cho test data
            
        Returns:
            Tuple (X, y) với X là features, y là target
        """
        print("=" * 50)
        print("BẮT ĐẦU FEATURE ENGINEERING")
        print("=" * 50)
        
        # Bước 1: Trích xuất time features
        print("\n[1/5] Trích xuất time features...")
        df_engineered = self.extract_time_features(df)
        
        # Bước 2: Convert boolean features
        print("\n[2/5] Convert boolean features...")
        df_engineered = self.convert_boolean_features(df_engineered)
        
        # Bước 3: Encode categorical features
        print("\n[3/5] Encode categorical features...")
        df_engineered = self.encode_categorical(df_engineered, fit=fit)
        
        # Bước 4: Scale numerical features
        print("\n[4/5] Scale numerical features...")
        df_engineered = self.scale_numerical(df_engineered, fit=fit)
        
        # Bước 5: Chọn features
        print("\n[5/5] Chọn features cho model...")
        X, y = self.select_features(df_engineered)
        
        print("\n" + "=" * 50)
        print(f"HOÀN THÀNH FEATURE ENGINEERING")
        print(f"X shape: {X.shape}, y shape: {y.shape if y is not None else 'None'}")
        print("=" * 50)
        
        return X, y


# Test module nếu chạy trực tiếp
if __name__ == "__main__":
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    
    # Load và preprocess dữ liệu
    loader = DataLoader("dataset/US_Accidents_March23.csv")
    df = loader.load_data(nrows=10000)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    
    # Feature engineering
    engineer = FeatureEngineer()
    X, y = engineer.transform(df_processed)
    
    print("\n=== Features cuối cùng ===")
    print(engineer.get_feature_names())
    
    print("\n=== Sample X ===")
    print(X.head())
