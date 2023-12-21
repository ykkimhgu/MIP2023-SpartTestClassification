"""
Handong Global University MCE 
Title: Spark Segmentation 
Author: EunChan Kim 
Date: 23-10-10
Description: Spark segmentation
"""

import cv2
import numpy as np
import time
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import glob

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

    @staticmethod
    def skeletonize(img):

        img = img.copy() # don't clobber original
        skel = img.copy()

        skel[:,:] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        while True:
            eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
            temp  = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img[:,:] = eroded[:,:]
            if cv2.countNonZero(img) == 0:
                break

        return skel
    
    def get_grid_orientations(self, img, grid_size):
        height, width = img.shape
        num_grids_x = width // grid_size
        num_grids_y = height // grid_size

        orientations = np.zeros((num_grids_y, num_grids_x))

        for i in range(num_grids_y):
            for j in range(num_grids_x):
                grid = img[i*grid_size:(i+1)*grid_size,
                           j*grid_size:(j+1)*grid_size]

                # Find coordinates of all white pixels in the grid.
                y_coords, x_coords = np.where(grid == 255)
                
                if len(x_coords) < 5:  # If less than two white pixels are found.
                    continue

                coords = np.column_stack((x_coords, y_coords))

                pca = PCA(n_components=2)
                pca.fit(coords)

                # Get the angle of the first principal component.
                angle = np.arctan2(pca.components_[0][1], pca.components_[0][0])
                
                orientations[i,j] = angle
                
        return orientations

    def fractal_dimension(image: np.ndarray) -> np.float64:

        M = image.shape[0]  # image shape
        G_min = image.min()  # lowest gray level (0=white)
        G_max = image.max()  # highest gray level (255=black)
        G = G_max - G_min + 1  # number of gray levels, typically 256
        prev = -1  # used to check for plateaus
        r_Nr = []
        
        box_sizes = [3, 4, 6, 8, 12, 16, 32]
        for L in box_sizes:
            h = max(1, G // (M // L))  # minimum box height is 1
            N_r = 0
            r = L / M
            for i in range(0, M, L):
                boxes = [[]] * ((G + h - 1) // h)  # create enough boxes with height h to fill the fractal space
                for row in image[i:i + L]:  # boxes that exceed bounds are shrunk to fit
                    for pixel in row[i:i + L]:
                        height = (pixel - G_min) // h  # lowest box is at G_min and each is h gray levels tall
                        boxes[height].append(pixel)  # assign the pixel intensity to the correct box
                stddev = np.sqrt(np.var(boxes, axis=1))  # calculate the standard deviation of each box
                stddev = stddev[~np.isnan(stddev)]  # remove boxes with NaN standard deviations (empty)
                nBox_r = 2 * (stddev // h) + 1
                N_r += sum(nBox_r)
            if N_r != prev:  # check for plateauing
                r_Nr.append([r, N_r])
                prev = N_r

        x = np.array([np.log(1 / point[0]) for point in r_Nr])  # log(1/r)
        y = np.array([np.log(point[1]) for point in r_Nr])  # log(Nr)
        D = np.polyfit(x, y, 1)[0]  # D = lim r -> 0 log(Nr)/log(1/r)

        return D
    

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

cap = cv2.VideoCapture('../Data/C35/C35_2.avi') #사용할 영상
video_name = 'C35_2'

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

explosion_num = 0
streamline_num  = 0
explosion_ratio = 0

cv2.namedWindow('video', cv2.WINDOW_NORMAL)

# mouse callback function
def print_coordinates(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'x={x}, y={y}')


    
cv2.setMouseCallback('video', print_coordinates)

pTime = 0

# 파일 이름을 현재 시간을 이용하여 생성
current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f'output_{current_time}.avi'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_filename, fourcc, fps, (int(width), int(height)))

while True:
    anomalies = np.empty((0,2))
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q') or key == 27: # If 'q' or 'ESC' is pressed, break the loop
        break 
    elif key == ord('p'): # If spacebar is pressed, pause the video
        while True:
            key2 = cv2.waitKey(1) or 0xff
            cv2.imshow('video', frame)
            if key2 == ord('p'): # If spacebar is pressed again, resume the video
                break

    ret, frame = cap.read()
    if not ret:
        break
    frame_count = frame_count + 1

#---------------------------------------------------
#                   Processing
#---------------------------------------------------
    
    x=0; y=700; w=1700;      
    roi = frame[y:, x:x+w]       
    #roi = frame[:, :]
    img_processor = ImgProcessor()  
    processed_roi = img_processor.PreProcess(roi)  
    processed_frame = img_processor.PreProcess(frame)
    edge = cv2.Canny(processed_roi, 200, 200)
    offset = 8
    area = img_processor.contour_area(processed_roi)
    
    grid_size=15
    # explosion_num = 0
    # streamline_num  = 0
    # explosion_ratio = 0

    height, width = edge.shape
    num_grids_x = width // grid_size
    num_grids_y = height // grid_size

    save_angle = np.zeros((num_grids_y, num_grids_x))

    #cv2.rectangle(frame, (x,y), (w,y+height), (219,225,162), 5)

    

    if area > 10000:

        orientations = img_processor.get_grid_orientations(edge, grid_size)  

        max_pca_count = 0  
        best_rectangle_index = 0  
        pca_counts = []  

        
        for i in range(orientations.shape[0]):
            for j in range(orientations.shape[1]):

                angle = orientations[i, j]

                if abs(angle) > 0.1:
                    save_angle[i,j] = angle

                    
        mean_angle = np.mean(save_angle[save_angle != 0])
        std_deviation = np.std(save_angle[save_angle != 0])
        z_scores = np.zeros((num_grids_y, num_grids_x))


        for i in range(orientations.shape[0]):
            for j in range(orientations.shape[1]):

                if save_angle[i,j] != 0:
                    z_scores[i,j] =  (save_angle[i,j] - mean_angle) / std_deviation
        
        # for i in range(z_scores.shape[0]):
        #     for j in range(z_scores.shape[1]):
        #         print(f"z_scores[{i},{j}] = {z_scores[i,j]}")

        

        for i in range(orientations.shape[0]):
            for j in range(orientations.shape[1]):
                
                angle = save_angle[i,j]

                if abs(z_scores[i,j]) > threshold:
                    #explosion_num += 1
                    center_x = int(x + j * grid_size)  # 중심 x 좌표
                    center_y = int(y + i * grid_size)  # 중심 y 좌표

                    line_length = 10  
                    line_x = int(center_x + line_length * np.cos(angle))
                    line_y = int(center_y + line_length * np.sin(angle))   
                    #cv2.line(frame, (center_x, center_y), (line_x, line_y), (0, 0, 255), 2)

                    anomalies = np.append(anomalies, [[center_x, center_y]], axis=0)
                #else:
                    #streamline_num += 1

        # explosion_ratio = explosion_num / streamline_num * 100
        # print(explosion_ratio)

        # DBSCAN clustering
        dbscan = DBSCAN(eps=25, min_samples=3)  # 적절한 eps와 min_samples 값을 설정
        anomalies = np.array(anomalies, dtype=np.int32)
        dbscan.fit(anomalies)
        cluster_labels = dbscan.labels_

        # Visualize
        for cluster_label in np.unique(cluster_labels):
            if cluster_label == -1:
                # Noise Cluster
                noise_points = anomalies[cluster_labels == -1]
                for point in noise_points:
                    x, y = point
                    #cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            else:
                # Cluster
                cluster_points = anomalies[cluster_labels == cluster_label]
                # for point in cluster_points:
                #     x, y = point
                #     cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
                if len(cluster_points) > 0: 
                    center_x = int(np.mean(cluster_points[:, 0]))
                    center_y = int(np.mean(cluster_points[:, 1]))

                    top_left_x = max(0, center_x - box_size // 2)
                    top_left_y = max(0, center_y - box_size // 2)

                    bottom_right_x = min(frame.shape[1], center_x + box_size // 2)
                    bottom_right_y = min(frame.shape[0], center_y + box_size // 2)

                    # cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,255,0),3)

                    extracted_area = processed_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                    # # pixel count
                    pixel_count = cv2.countNonZero(extracted_area)
                    
                    if 1500 < pixel_count < 3000:
                        print(pixel_count)
                        # fractal_dimension = ImgProcessor.fractal_dimension(extracted_area)
                        # print(fractal_dimension)

                        cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,255,0),3)
                        # with open(txt_filename, "a") as file:
                        #     file.write(f"{pixel_count}\n")

                        green_box_frame = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                        cv2.imwrite(f"{video_name}_{frame_count}_box.jpg", green_box_frame)

                    else:
                        cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),3)

                    #     with open(txt_filename, "a") as file:
                    #         file.write(f"{pixel_count}\n")
                        #cv2.rectangle(frame,(top_left_x,top_left_y),(bottom_right_x,bottom_right_y),(0,0,255),3)

                    # fractal_dimension = ImgProcessor.fractal_dimension(extracted_area)
                    # with open(txt_filename, "a") as file:
                    #         file.write(f"{fractal_dimension}\n")
                
                video_name = 'C35_2'
                cv2.imwrite(f"{video_name}_{frame_count}_pca.jpg", frame)
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

