"""
Handong Global University MCE 
Title: Spark Segmentation 
Author: EunChan Kim 
Date: 23-11-24
Description: Spark segmentation
"""

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

#---------------------------------------------------
#                   Data Read
#---------------------------------------------------
video_files = glob.glob('C:/Users/eunchan/Capstone/Code/Test/*.avi')  # 경로는 실제 비디오 파일이 위치한 폴더로 변경

for video_file in video_files:
    # 비디오 파일 이름에서 확장자를 제거, 파일명만 추출
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    cap = cv2.VideoCapture(video_file)


    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_time = time.time()  # 이전 프레임의 처리 시간
    frame_count = 0  # 현재까지 처리한 프레임 수
    angles = []
    start = False
    anomalies = np.empty((0,2))
    threshold = 1.3
    box_size = 150

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.namedWindow('prep', cv2.WINDOW_NORMAL)

    # mouse callback function
    def print_coordinates(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'x={x}, y={y}')


        
    cv2.setMouseCallback('video', print_coordinates)

    pTime = 0


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (int(width), int(height)))
    ifft_areas = []  

    # Define initial values for the trackbars
    threshold_value = 50
    min_line_length = 50
    max_line_gap = 140

    # Create a window for the trackbars
    cv2.namedWindow('Parameters')

    # Create trackbars for adjusting parameters
    cv2.createTrackbar('Threshold', 'Parameters', threshold_value, 1000, nothing)
    cv2.createTrackbar('Min Line Length', 'Parameters', min_line_length, 200, nothing)
    cv2.createTrackbar('Max Line Gap', 'Parameters', max_line_gap, 1000, nothing)

    ifft_result_uint8 = None


    while True:
        anomalies = np.empty((0,2))
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q') or key == 27:  # If 'q' or 'ESC' is pressed, break the loop
            break 
        elif key == ord('p'):  # If spacebar is pressed, pause the video
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('video', frame)
                if key2 == ord('p'):  # If spacebar is pressed again, resume the video
                    break

        ret, frame = cap.read()
        if not ret:
            break
        frame_count = frame_count + 1

        # Get parameter values from the trackbars
        threshold_value = cv2.getTrackbarPos('Threshold', 'Parameters')
        min_line_length = cv2.getTrackbarPos('Min Line Length', 'Parameters')
        max_line_gap = cv2.getTrackbarPos('Max Line Gap', 'Parameters')
        

    #---------------------------------------------------
    #                   Processing
    #---------------------------------------------------
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

        

        if area > 4000:
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

            #------------------------ FFT 출력 ---------------------------
            # 로그 스케일로 변환
            magnitude_spectrum_log = np.log1p(magnitude_spectrum)
            mask_log = np.log1p(magnitude_spectrum_copy)
            remove_magnitude_spectrum_log = np.log1p(remove_magnitude_spectrum)

            # 실수 행렬을 [0, 255] 범위의 8비트 정수로 변환
            magnitude_spectrum_uint8 = cv2.normalize(magnitude_spectrum_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            mask_uint8 = cv2.normalize(mask_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            remove_uint8 = cv2.normalize(remove_magnitude_spectrum_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            cv2.imshow('prep', remove_uint8)

            #------------------------ IFFT -------------------------------
            # 역 FFT 변환
            ifft_result = np.fft.ifft2(np.fft.ifftshift(fft_result_shifted))

            # 복소수 값에서 절댓값을 취하여 실수 행렬로 변환
            ifft_result_abs = np.abs(ifft_result)

            # 실수 행렬을 [0, 255] 범위의 8비트 정수로 변환
            ifft_result_uint8 = (ifft_result_abs / ifft_result_abs.max() * 255).astype(np.uint8)


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

        out.write(frame)


    #---------------------------------------------------
    #                       Result
    #---------------------------------------------------

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        fps_str = f"FPS: {fps:.02f}"

        cv2.putText(frame ,fps_str ,(70 ,150) ,cv2.FONT_HERSHEY_SIMPLEX ,3 ,(255 ,255 ,255) ,3)

        # Show video
        cv2.imshow('video', frame)

        key=cv2.waitKey(1)&0xFF 
        if key==ord('q') or key ==27:
            break 

cap.release()
cv2.destroyAllWindows()

out.release()
    
