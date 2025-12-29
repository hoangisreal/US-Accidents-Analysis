================================================================================
        DỰ ĐOÁN MỨC ĐỘ NGHIÊM TRỌNG TAI NẠN GIAO THÔNG
              (Accident Severity Prediction)
================================================================================

Bài tập lớn môn: Tiền xử lý dữ liệu

================================================================================
1. MÔ TẢ PROJECT
================================================================================

Mục tiêu:
---------
Xây dựng mô hình Machine Learning để dự đoán mức độ nghiêm trọng của tai nạn 
giao thông (Mild - Nhẹ hoặc Severe - Nghiêm trọng) dựa trên các yếu tố như 
thời tiết, vị trí địa lý, điều kiện đường xá.

Dataset:
--------
- Nguồn: US Accidents (Kaggle)
- Link: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- Mô tả: Dataset chứa thông tin về ~7.7 triệu tai nạn giao thông tại Mỹ 
         từ năm 2016 đến 2023

Phương pháp:
------------
- Bài toán: Binary Classification (Mild vs Severe)
- Models: Logistic Regression, Random Forest, XGBoost
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

================================================================================
2. CÀI ĐẶT MÔI TRƯỜNG
================================================================================

Yêu cầu:
--------
- Python 3.8 trở lên
- pip (Python package manager)

Cài đặt các thư viện:
---------------------
Mở Command Prompt/Terminal và chạy lệnh:

    pip install -r requirements.txt

Các thư viện chính:
-------------------
- pandas: Xử lý dữ liệu
- numpy: Tính toán số học
- scikit-learn: Machine Learning
- xgboost: Gradient Boosting
- matplotlib, seaborn: Visualization
- jupyter: Notebook

================================================================================
3. CẤU TRÚC PROJECT
================================================================================

project/
│
├── dataset/
│   └── US_Accidents_March23.csv    # Dataset (cần download từ Kaggle)
│
├── src/                            # Source code Python
│   ├── __init__.py                 # Package init
│   ├── data_loader.py              # Load và khám phá dữ liệu
│   ├── data_preprocessor.py        # Tiền xử lý dữ liệu
│   ├── feature_engineer.py         # Feature engineering
│   ├── model_trainer.py            # Huấn luyện và đánh giá models
│   └── visualizer.py               # Tạo visualizations
│
├── notebooks/
│   └── main_analysis.ipynb         # Jupyter notebook chính
│
├── models/                         # Lưu models đã train
│
├── outputs/                        # Lưu plots và kết quả
│
├── requirements.txt                # Danh sách thư viện
└── README.txt                      # File này

================================================================================
4. HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH
================================================================================

Bước 1: Download Dataset
------------------------
1. Truy cập: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
2. Download file CSV
3. Đặt file vào thư mục: dataset/US_Accidents_March23.csv

Bước 2: Cài đặt thư viện
------------------------
    pip install -r requirements.txt

Bước 3: Chạy Jupyter Notebook
-----------------------------
    cd notebooks
    jupyter notebook main_analysis.ipynb

Hoặc mở Jupyter Lab:
    jupyter lab

Bước 4: Chạy từng cell trong notebook
-------------------------------------
- Chạy tuần tự từ trên xuống dưới
- Mỗi section có giải thích chi tiết

================================================================================
5. MÔ TẢ CÁC MODULES
================================================================================

data_loader.py
--------------
- Class DataLoader: Load dữ liệu từ CSV
- Methods: load_data(), get_basic_info(), get_target_distribution()

data_preprocessor.py
--------------------
- Class DataPreprocessor: Tiền xử lý dữ liệu
- Methods: drop_unnecessary_columns(), impute_missing_values(), 
           transform_target(), preprocess()

feature_engineer.py
-------------------
- Class FeatureEngineer: Tạo và biến đổi features
- Methods: extract_time_features(), encode_categorical(), 
           scale_numerical(), transform()

model_trainer.py
----------------
- Class ModelTrainer: Huấn luyện và đánh giá models
- Methods: get_models(), split_data(), train_and_evaluate_all(),
           get_feature_importance(), save_model()

visualizer.py
-------------
- Class Visualizer: Tạo visualizations
- Methods: plot_target_distribution(), plot_confusion_matrix(),
           plot_roc_curves(), plot_feature_importance()


