"""
Handong Global University MCE 
Title: Spark Segmentation 
Author: EunChan Kim 
Date: 23-11-24
Description: Spark segmentation
"""
import cv2
import numpy as np
import glob
import os
from scipy import stats
from tqdm import tqdm
import numpy.fft as fft
from scipy.stats import entropy
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import seaborn as sns


# 클래스별로 폴더를 분류
classes = ['C10', 'C20', 'C25', 'C35', 'C40', 'C53', 'C55']

# 결과를 저장할 딕셔너리
train_features = {}
test_features = {}

# 테스트 데이터 위치 
train_data_path = 'C:/Users/eunchan/Capstone/Code/Train/expand'
test_data_path = 'C:/Users/eunchan/Capstone/Code/Test/expand'

# 프레임 평균 기준(몇개의 데이터를 하나의 데이터셋으로 만들건지)
frame = 50
# 이미지 크기를 통일할 너비와 높이 설정
target_width = 224
target_height = 224

for class_name in tqdm(classes):
    # 각 클래스별로 이미지 파일 경로를 모두 가져옴
    image_files = glob.glob(os.path.join(train_data_path, f'{class_name}_*_*_roi_*.jpg'))

    # 파일 이름을 기준으로 정렬
    image_files.sort()

    # n개 프레임씩 특징을 계산
    for i in tqdm(range(0, len(image_files), frame)):
        batch_files = image_files[i:i+frame]
        batch_images = []

        for file in batch_files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (target_width, target_height))
            batch_images.append(resized_img)

        batch_images = np.array(batch_images)

        # FFT 변환
        fft_images = fft.fftn(batch_images, axes=(1, 2))
        fft_images_abs = np.abs(fft_images)
        
        # 특징 계산
        energy = np.sum(batch_images**2, axis=(1, 2))
        mean = np.mean(batch_images, axis=(1, 2))
        std_dev = np.std(batch_images, axis=(1, 2))
        skewness = stats.skew(batch_images.reshape(batch_images.shape[0], -1), axis=1)
        kurtosis = stats.kurtosis(batch_images.reshape(batch_images.shape[0], -1), axis=1)
        rms = np.sqrt(np.mean(batch_images**2, axis=(1, 2)))
        sra = np.power(np.mean(np.sqrt(np.abs(batch_images)), axis=(1, 2)), 2)
        aav = np.mean(np.abs(batch_images), axis=(1, 2))
        peak = np.max(batch_images, axis=(1, 2))
        ppv = np.ptp(batch_images, axis=(1, 2))
        impact_factor = peak / aav
        shape_factor = rms / aav
        crest_factor = peak / rms
        margin_factor = peak / sra
        
        # FFT 특징
        peak_freq = np.unravel_index(np.argmax(fft_images_abs, axis=None), fft_images_abs.shape)
        energy_dist = np.sum(fft_images_abs**2, axis=(1, 2))

        # 엔트로피
        entropy_list = []
        for img in batch_images:
            hist = cv2.calcHist([img], [0], None, [256], [0,256])
            hist /= hist.sum()  # normalize
            entropy_list.append(entropy(hist))


        # 결과를 딕셔너리에 저장
        video_number, frame_number = os.path.splitext(batch_files[0])[0].split('_')[1:3]  # 영상 번호와 프레임 번호 추출
        feature_key = f"{class_name}_{video_number}_{frame_number}"
        train_features[feature_key] = {
            'energy_mean': np.mean(energy),
            'mean_mean': np.mean(mean),
            'std_mean': np.mean(std_dev),
            'sk_mean': np.mean(skewness),
            'kt_mean': np.mean(kurtosis),
            'rms_mean': np.mean(rms),
            'sra_mean': np.mean(sra),
            'aav_mean': np.mean(aav),
            'peak_mean': np.mean(peak),
            'ppv_mean': np.mean(ppv),
            'if_mean': np.mean(impact_factor),
            'sf_mean': np.mean(shape_factor),
            'cf_mean': np.mean(crest_factor),
            'mf_mean': np.mean(margin_factor),
            'peakfreq_mean' : np.mean(peak_freq),
            'energydist_mean' : np.mean(energy_dist),
            'energy_rms': np.sqrt(np.mean(energy)),
            'mean_rms': np.sqrt(np.mean(mean)),
            'std_rms': np.sqrt(np.mean(std_dev)),
            'sk_rms': np.sqrt(np.mean(skewness)),
            'kt_rms': np.sqrt(np.mean(kurtosis)),
            'rms_rms': np.sqrt(np.mean(rms)),
            'sra_rms': np.sqrt(np.mean(sra)),
            'aav_rms': np.sqrt(np.mean(aav)),
            'peak_rms': np.sqrt(np.mean(peak)),
            'ppv_rms': np.sqrt(np.mean(ppv)),
            'if_rms': np.sqrt(np.mean(impact_factor)),
            'sf_rms': np.sqrt(np.mean(shape_factor)),
            'cf_rms': np.sqrt(np.mean(crest_factor)),
            'mf_rms': np.sqrt(np.mean(margin_factor)),
            'peakfreq_rms' : np.sqrt(np.mean(peak_freq)),
            'energydist_rms' : np.sqrt(np.mean(energy_dist)),
            'energy_max': np.max(energy),
            'mean_max': np.max(mean),
            'std_max': np.max(std_dev),
            'sk_max': np.max(skewness),
            'kt_max': np.max(kurtosis),
            'rms_max': np.max(rms),
            'sra_max': np.max(sra),
            'aav_max': np.max(aav),
            'peak_max': np.max(peak),
            'ppv_max': np.max(ppv),
            'if_max': np.max(impact_factor),
            'sf_max': np.max(shape_factor),
            'cf_max': np.max(crest_factor),
            'mf_max': np.max(margin_factor),
            'peakfreq_max' : np.max(peak_freq),
            'energydist_max' : np.max(energy_dist),
            'entropy_mean': np.mean(entropy_list),
            'entropy_max': np.max(entropy_list),
            'entropy_rms': np.sqrt(np.mean(np.square(entropy_list))) 

        }

for class_name in tqdm(classes):
    # 각 클래스별로 이미지 파일 경로를 모두 가져옴
    image_files = glob.glob(os.path.join(test_data_path, f'{class_name}_*_*_roi_*.jpg'))

    # 파일 이름을 기준으로 정렬 
    image_files.sort()

    # n개 프레임씩 특징을 계산
    for i in tqdm(range(0, len(image_files), frame)):
        batch_files = image_files[i:i+frame]
        batch_images = []

        for file in batch_files:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (target_width, target_height))
            batch_images.append(resized_img)

        batch_images = np.array(batch_images)

        # FFT 변환
        fft_images = fft.fftn(batch_images, axes=(1, 2))
        fft_images_abs = np.abs(fft_images)
        
        # 특징 계산
        energy = np.sum(batch_images**2, axis=(1, 2))
        mean = np.mean(batch_images, axis=(1, 2))
        std_dev = np.std(batch_images, axis=(1, 2))
        skewness = stats.skew(batch_images.reshape(batch_images.shape[0], -1), axis=1)
        kurtosis = stats.kurtosis(batch_images.reshape(batch_images.shape[0], -1), axis=1)
        rms = np.sqrt(np.mean(batch_images**2, axis=(1, 2)))
        sra = np.power(np.mean(np.sqrt(np.abs(batch_images)), axis=(1, 2)), 2)
        aav = np.mean(np.abs(batch_images), axis=(1, 2))
        peak = np.max(batch_images, axis=(1, 2))
        ppv = np.ptp(batch_images, axis=(1, 2))
        impact_factor = peak / aav
        shape_factor = rms / aav
        crest_factor = peak / rms
        margin_factor = peak / sra
        
        # FFT 특징
        peak_freq = np.unravel_index(np.argmax(fft_images_abs, axis=None), fft_images_abs.shape)
        energy_dist = np.sum(fft_images_abs**2, axis=(1, 2))

        # 엔트로피
        entropy_list = []
        for img in batch_images:
            hist = cv2.calcHist([img], [0], None, [256], [0,256])
            hist /= hist.sum()  # normalize
            entropy_list.append(entropy(hist))


        # 결과를 딕셔너리에 저장
        video_number, frame_number = os.path.splitext(batch_files[0])[0].split('_')[1:3]  # 영상 번호와 프레임 번호 추출
        feature_key = f"{class_name}_{video_number}_{frame_number}"
        test_features[feature_key] = {
            'energy_mean': np.mean(energy),
            'mean_mean': np.mean(mean),
            'std_mean': np.mean(std_dev),
            'sk_mean': np.mean(skewness),
            'kt_mean': np.mean(kurtosis),
            'rms_mean': np.mean(rms),
            'sra_mean': np.mean(sra),
            'aav_mean': np.mean(aav),
            'peak_mean': np.mean(peak),
            'ppv_mean': np.mean(ppv),
            'if_mean': np.mean(impact_factor),
            'sf_mean': np.mean(shape_factor),
            'cf_mean': np.mean(crest_factor),
            'mf_mean': np.mean(margin_factor),
            'peakfreq_mean' : np.mean(peak_freq),
            'energydist_mean' : np.mean(energy_dist),
            'energy_rms': np.sqrt(np.mean(energy)),
            'mean_rms': np.sqrt(np.mean(mean)),
            'std_rms': np.sqrt(np.mean(std_dev)),
            'sk_rms': np.sqrt(np.mean(skewness)),
            'kt_rms': np.sqrt(np.mean(kurtosis)),
            'rms_rms': np.sqrt(np.mean(rms)),
            'sra_rms': np.sqrt(np.mean(sra)),
            'aav_rms': np.sqrt(np.mean(aav)),
            'peak_rms': np.sqrt(np.mean(peak)),
            'ppv_rms': np.sqrt(np.mean(ppv)),
            'if_rms': np.sqrt(np.mean(impact_factor)),
            'sf_rms': np.sqrt(np.mean(shape_factor)),
            'cf_rms': np.sqrt(np.mean(crest_factor)),
            'mf_rms': np.sqrt(np.mean(margin_factor)),
            'peakfreq_rms' : np.sqrt(np.mean(peak_freq)),
            'energydist_rms' : np.sqrt(np.mean(energy_dist)),
            'energy_max': np.max(energy),
            'mean_max': np.max(mean),
            'std_max': np.max(std_dev),
            'sk_max': np.max(skewness),
            'kt_max': np.max(kurtosis),
            'rms_max': np.max(rms),
            'sra_max': np.max(sra),
            'aav_max': np.max(aav),
            'peak_max': np.max(peak),
            'ppv_max': np.max(ppv),
            'if_max': np.max(impact_factor),
            'sf_max': np.max(shape_factor),
            'cf_max': np.max(crest_factor),
            'mf_max': np.max(margin_factor),
            'peakfreq_max' : np.max(peak_freq),
            'energydist_max' : np.max(energy_dist),
            'entropy_mean': np.mean(entropy_list),
            'entropy_max': np.max(entropy_list),
            'entropy_rms': np.sqrt(np.mean(np.square(entropy_list))) 

        }

# 딕셔너리를 데이터프레임으로 변환
train_df = pd.DataFrame(train_features).T
test_df = pd.DataFrame(test_features).T

# 클래스 정보를 포함하는 새로운 열 추가
train_df['class'] = train_df.index.str.split('_').str[0]
test_df['class'] = test_df.index.str.split('_').str[0]

# 클래스별로 데이터 분리 및 Forward Selection 적용
for class_name in classes:
    train_class_data = train_df[train_df['class'] == class_name]
    test_class_data = test_df[test_df['class'] == class_name]
    
    # 학습 및 테스트 데이터 분리
    X_train = train_df.drop(['class'], axis=1)
    y_train = train_df['class']
    X_test = test_df.drop(['class'], axis=1)
    y_test = test_df['class']
    
    # 라벨 인코딩
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # 스케일러 초기화
    scaler = StandardScaler()

    # 데이터 스케일링
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # NaN 값을 대체하는 Imputer 생성
    imputer = SimpleImputer()

    # NaN 값 대체
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # 특징 선택
    sfs = SFS(LinearRegression(),
                k_features=30,  # 상위 30개의 특징 선택
                forward=True,
                floating=False,
                scoring='neg_mean_squared_error',
                cv=0)

    sfs.fit(X_train, y_train)

    # Selected feature names and corresponding indices
    selected_features = sfs.k_feature_names_
    selected_indices = sfs.k_feature_idx_
    # LinearRegression 모델 학습
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # 특성 선택 결과로 얻은 상위 3개 특성의 인덱스 (문자열을 정수로 변환)
    top_features_indices = list(map(int, selected_features[:3]))
    # 원본 데이터프레임에서 특성 이름 가져오기
    top_features = [train_df.columns[i] for i in top_features_indices]

    # 클래스별로 데이터 분리 및 시각화
    for feature in top_features:
        plt.figure(figsize=(10, 6))

        for i, class_name in enumerate(classes):
            class_data = train_df[train_df['class'] == class_name]
            sns.scatterplot(x=[class_name]*len(class_data), y=class_data[feature], label=class_name)

        plt.title(f"Feature {feature}")
        plt.xlabel('Class')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.show()

    # SVM model
    svm_model = SVC()
    svm_model.fit(X_train[:, list(selected_indices)], y_train)
    svm_predictions = svm_model.predict(X_test[:, list(selected_indices)])

    # KNN model
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train[:, list(selected_indices)], y_train)
    knn_predictions = knn_model.predict(X_test[:, list(selected_indices)])

    # Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train[:, list(selected_indices)], y_train)
    rf_predictions = rf_model.predict(X_test[:, list(selected_indices)])

    # Accuracy, Precision, and Recall
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_predictions)}")
    print(f"SVM Precision: {precision_score(y_test, svm_predictions, average='weighted', zero_division=1)}")
    print(f"SVM Recall: {recall_score(y_test, svm_predictions, average='weighted')}")

    print(f"KNN Accuracy: {accuracy_score(y_test, knn_predictions)}")
    print(f"KNN Precision: {precision_score(y_test, knn_predictions, average='weighted', zero_division=1)}")
    print(f"KNN Recall: {recall_score(y_test, knn_predictions, average='weighted')}")

    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions)}")
    print(f"Random Forest Precision: {precision_score(y_test, rf_predictions, average='weighted', zero_division=1)}")
    print(f"Random Forest Recall: {recall_score(y_test, rf_predictions, average='weighted')}")

    # Confusion Matrix
    svm_cm = confusion_matrix(y_test, svm_predictions)
    knn_cm = confusion_matrix(y_test, knn_predictions)
    rf_cm = confusion_matrix(y_test, rf_predictions)

    # 클래스 이름 가져오기
    class_names = le.classes_

    plt.figure(figsize=(12, 6))

    plt.subplot(131)
    sns.heatmap(svm_cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"SVM Confusion Matrix")

    plt.subplot(132)
    sns.heatmap(knn_cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"KNN Confusion Matrix")

    plt.subplot(133)
    sns.heatmap(rf_cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Random Forest Confusion Matrix")

    plt.show()

    # Plotting the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(list(sfs.get_metric_dict().keys()), [-v['avg_score'] for v in sfs.get_metric_dict().values()], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss as a function of number of features')
    plt.show()

    # 각 모델 별로 정밀도, 재현율, 정확도, F1 스코어를 클래스별로 계산하고 출력합니다.
    def print_evaluation_scores(y_true, y_pred, class_names):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        print(f"Accuracy: {accuracy:.4f}")
        for idx, cls in enumerate(class_names):
            print(f"Class '{cls}' - Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}, F1 Score: {f1[idx]:.4f}")

    # 여기서 'class_names' 변수는 클래스의 이름을 담고 있는 리스트입니다.
    # 예를 들어 class_names = ['class1', 'class2', 'class3']

    # SVM 평가 지표 출력
    print("SVM Model Evaluation")
    print_evaluation_scores(y_test, svm_predictions, class_names)

    # KNN 평가 지표 출력
    print("\nKNN Model Evaluation")
    print_evaluation_scores(y_test, knn_predictions, class_names)

    # Random Forest 평가 지표 출력
    print("\nRandom Forest Model Evaluation")
    print_evaluation_scores(y_test, rf_predictions, class_names)
