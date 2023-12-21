# MIP2023-SpartTestClassification

## Introduction

 본 repository는 2023년 2학기에 진행된 기전융합종합설계, **Basic Research on Steel Type Distinction Technology Based on Spark Video** 에 대한 실행 파일과 튜토리얼으로 구성되어 있습니다.

 연구의 목적은 공정에서 서로 다른 강재가 섞이는 것을 예방하기 위한 강종 혼재 감별 기술 개발입니다. 본 연구에서는 **탄소 함유량에 따른 강철 스파크 특성**을 **영상 처리 기법**과 접목하여 강종 분류를 진행하였습니다.   



## Requirements

 본 연구에서는 미리 촬영된 스파크 영상을 사용했으므로 SW 외의 다른 요구 사항은 없습니다. 

- Python 3.9.18
- Anaconda 23.5.2
- OpenCV  4.8.1
- Numpy `pip install numpy`
- PIL `pip install pillow`
- sklearn `pip install scikit-learn`
- matplot `pip install matplotlib`
- scipy `pip install scipy`
- librosa `pip install librosa`
- PIL `pip install pillow`



## Tutorial

 파이썬 코드는 **데이터 전처리** + **특징 추출 및 머신러닝 학습**  두가지로 구성되어 있습니다. `FFT_remove.py` 를 동작하게 되면 데이터 전처리와 학습 및 시험 데이터 셋을 저장할 수 있습니다. `ML(SVM,KNN,RF).py`를 구동하게 되면 특징 추출 및 머신러닝 학습을 진행하고 결과를 플롯하게 됩니다.

![플로우](https://github.com/GracenPraise/Embedded-Controller/assets/91367451/d311ebf8-48b1-484b-ae67-b72b13fd6981)

<center>Figure1. Steel Classification Flow Chart</center>



코드 구동 속도 향상을 위해, 두 코드 모두 아나콘다 환경에서 실행 하는 것을 추천드립니다. 실행 순서는 아래와 같습니다.

- `cd (파이썬 파일 위치한 경로)`
- `conda create -n (원하는 이름) python=3.9`  <- 가상 환경 미 생성 시

- `conda activate (가상 환경)`
- `python FFT_remove.py`
- `python ML(SVM,KNN,RF).py`



### 코드 설명

코드 내의 설명을 보고 파라미터를 수정하여 사용하실 수 있습니다.

#### FFT_remove.py

##### 라이브러리

```c
import cv2
import numpy as np
import time
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy
import librosa
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
```



##### 클래스 및 함수 정의

그레이 스케일 변환 및 이진화 실행 함수와 컨투어 면적을 계산하는 함수를 제작했습니다

```c
def nothing(x):
    pass

class ImgProcessor:
    def __init__(self, threshold=150):
        self.threshold = threshold

    def PreProcess(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        ret, binary = cv2.threshold(gray, self.threshold ,255,cv2.THRESH_BINARY)
    
        return binary    

    def contour_area(self, image):

        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        total_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            total_area += area

        return total_area
```



##### 데이터 전처리

ROI를 설정하고 해당 영역을 이진화 합니다. 또한 필요한 변수를 정의 했습니다.

```c
        height, width, _ = frame.shape
        x=0; y=700; w=1700;      
        roi = frame[y:, x:x+w]       
        img_processor = ImgProcessor()  
        processed_roi = img_processor.PreProcess(roi)  
        processed_frame = img_processor.PreProcess(frame)
        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        offset = 8
        area = img_processor.contour_area(processed_roi)
        
        grid_size=15

        num_grids_x = width // grid_size
        num_grids_y = height // grid_size

        save_angle = np.zeros((num_grids_y, num_grids_x))

        cv2.rectangle(frame, (x,y), (w,y+height), (219,225,162), 5)
```



##### FFT

이미지를 FFT 변환하고 magnitude 임계 값을 설정하여 저주파 영역을 제거하였습니다. 또한 폭발하는 스파크 성분을 FFT 에서 유지하기 위해 타워 형태의 마스크를 만들어 값을 유지했습니다.

```c
# FFT 수행
            fft_result = np.fft.fft2(processed_roi)

            # 주파수 이동 (중심을 0,0으로)
            fft_result_shifted = np.fft.fftshift(fft_result)
            fft_result_shifted_copy = np.fft.fftshift(fft_result)

            # 복소수 값에서 절댓값을 취하여 실수 행렬로 변환
            magnitude_spectrum = np.abs(fft_result_shifted)
            magnitude_spectrum_copy = np.abs(fft_result_shifted_copy)
            threshold_fft = 10000
            fft_result_shifted[(magnitude_spectrum > threshold_fft) ] = 0
            magnitude_spectrum[(magnitude_spectrum > threshold_fft) ] = 0
            


            #------------------------ 원 영역 마스크 추가 -----------------
            # FFT 결과에 대한 마스크 생성
            height, width = fft_result_shifted.shape
            center = (width//2, height//2)
            mask = np.zeros((height, width), np.uint8)

            # 250 반지름의 타원 생성
            cv2.ellipse(mask, center, (270, 70), -45, 0, 360, 1, -1)  # 타원형 마스크 생성. 타원의 중심, 장축과 단축의 길이, 회전 각도, 시작 각도, 끝 각도, 색상, 두께(-1은 내부를 채움)를 지정


            # 원을 중심으로 4등분 하여 오른쪽 상단과 왼쪽 하단 영역 마스킹
            mask[:center[1], :center[0]] = 0  # 왼쪽 상단
            mask[center[1]:, center[0]:] = 0  # 오른쪽 하단

            fft_result_shifted_copy *= mask

            # fft_result_shifted_copy에서 0이 아닌 값들의 인덱스를 찾음
            non_zero_indices = np.where(fft_result_shifted_copy != 0)

            # 해당 인덱스의 fft_result_shifted 값들을 fft_result_shifted_copy의 값으로 덮어씌움
            fft_result_shifted[non_zero_indices] = fft_result_shifted_copy[non_zero_indices]

            remove_magnitude_spectrum = np.abs(fft_result_shifted)
```



##### FFT 시각화

FFT 이미지에서 필터링한 결과를 시각화 하는 부분입니다. 

```c
            # 로그 스케일로 변환
            magnitude_spectrum_log = np.log1p(magnitude_spectrum)
            mask_log = np.log1p(magnitude_spectrum_copy)
            remove_magnitude_spectrum_log = np.log1p(remove_magnitude_spectrum)

            # 실수 행렬을 [0, 255] 범위의 8비트 정수로 변환
            magnitude_spectrum_uint8 = cv2.normalize(magnitude_spectrum_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            mask_uint8 = cv2.normalize(mask_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            remove_uint8 = cv2.normalize(remove_magnitude_spectrum_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            cv2.imshow('prep', remove_uint8)
```



##### IFFT

필터링 된 FFT 이미지를 IFFT로 역변환 하여 실제 이미지 도메인으로 복원합니다.

```c
            #------------------------ IFFT -------------------------------
            # 역 FFT 변환
            ifft_result = np.fft.ifft2(np.fft.ifftshift(fft_result_shifted))

            # 복소수 값에서 절댓값을 취하여 실수 행렬로 변환
            ifft_result_abs = np.abs(ifft_result)

            # 실수 행렬을 [0, 255] 범위의 8비트 정수로 변환
            ifft_result_uint8 = (ifft_result_abs / ifft_result_abs.max() * 255).astype(np.uint8)
```



##### 데이터 출력

줄기 스파크를 제거한 이미지에서 스파크 개별 영역을 추출하기 위해 컨투어를 측정하고 특정 면적 이상인 컨투어를 하나의 스파크 개체로 판단했습니다. 개별 스파크를 저장하여 데이터 셋으로 사용합니다.

```c
            #------------------------ 후처리 -------------------------------
            ret, thresh_tozero = cv2.threshold(ifft_result_uint8, 25, 255, cv2.THRESH_TOZERO)

            dst = cv2.medianBlur(thresh_tozero, 3)

            # 컨투어 감지
            contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 특정 면적 이상인 컨투어 필터링
            min_area_threshold = 800  # 특정 면적 값 이상을 설정해주세요.
            filtered_contours = [cnt for cnt in contours if 12000 > cv2.contourArea(cnt) >= min_area_threshold]

            # roi 위치 반영하여 외곽 라인 그리기 및 사각형 표시
            for i, cnt in enumerate(filtered_contours):
                # 각 컨투어의 좌표에 roi의 시작점을 더해줌
                contour_shifted = cnt + (x, y)
                cv2.drawContours(frame, [contour_shifted], -1, (0, 255, 0), 2)
                
                # 컨투어를 감싸는 사각형의 좌표 얻기
                x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)

                # 사각형에 해당하는 영역 추출하여 저장
                roi = dst[y_rect:y_rect+h_rect, x_rect:x_rect+w_rect]
                cv2.imwrite(f"{video_name}_{frame_count}_roi_{i}.jpg", roi) #개별 스파크 저장

            cv2.imshow('result', dst)
```



#### ML(SVM,KNN,RF).py

##### 라이브러리

```c
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
```



##### 전처리

데이터 경로와 필요한 변수를 선언하고 개별 스파크 이미지를 모두 동일한 이미지 사이즈로 통일하였습니다. 

```c
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
```



##### 특징 추출

각 스파크 이미지에서 특징을 추출하고 50개의 평균을 하나의 데이터로 확보하였습니다.

```c
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
```



##### 학습 준비 및 SFS 특징 선별

추출한 특징을 데이터프레임으로 변환합니다. 데이터의 영향력을 동일하게 하기 위해 스케일러를 사용해 정규화 해주었습니다. SFS 기법을 사용하여 클래스 특성을 잘 반영하는 상위 특징을 선별했습니다.

```c
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

```



##### 학습 및 시각화

SVM, KNN, 랜덤 포레스트 세가지 모델로 학습을 진행하고 분류 결과를 시각화 합니다.

```c
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
```