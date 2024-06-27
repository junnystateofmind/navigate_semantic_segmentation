import cv2
import numpy as np

# Load video
video_path = '../Ship Navigation.mp4'  # 비디오 파일의 절대 경로 사용
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video pre-processing
def pre_process(frame):
    # Gaussian blur
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # # Histogram equalization
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.equalizeHist(frame)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = clahe.apply(frame)

    # # Adaptive thresholding
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return frame

# semantic segmentation

def semantic_process(frame):
    # already gray scale

    # Canny edge detection
    frame = cv2.Canny(frame, 100, 200)

    # # Dilation
    # kernel = np.ones((5, 5), np.uint8)
    # frame = cv2.dilate(frame, kernel, iterations=1)

    # Watershed segmentation


    return frame


# Video show
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = pre_process(frame)
    frame = semantic_process(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
