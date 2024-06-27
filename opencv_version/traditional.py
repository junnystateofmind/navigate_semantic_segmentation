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
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh


# Contour detection and object recognition
def detect_objects(thresh, original_frame):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []

    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect_ratio = float(w) / h
        valid_size = float(w * h) / (original_frame.shape[0] * original_frame.shape[1] * 0.1) # 전체 프레임의 10% 이상을 차지하는 객체만 인식

        # Ignore contours that are too narrow or too wide
        if 0.2 < aspect_ratio < 5.0 and valid_size > 0.05:
            valid_contours.append(contour)
            # Draw contours
            cv2.drawContours(original_frame, [contour], 0, (0, 255, 0), 2)

    return original_frame, valid_contours


# Video show
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    thresh = pre_process(frame)
    detected_frame, contours = detect_objects(thresh, frame.copy())

    # Display the results
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Thresholded Frame', thresh)
    cv2.imshow('Detected Objects', detected_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
