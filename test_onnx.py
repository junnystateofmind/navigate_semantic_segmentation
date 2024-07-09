import onnxruntime as ort
import numpy as np
import cv2

# 모델 경로 설정
onnx_model_path = 'PIDNet_S_Camvid_Test.onnx'

# ONNX Runtime 세션 생성
ort_session = ort.InferenceSession(onnx_model_path)

# 입력 비디오 데이터로 Ship Navigation.mp4 파일을 사용
video_path = 'blue-cross.mp4'

# OpenCV를 사용하여 비디오 파일 열기
cap = cv2.VideoCapture(video_path)

# 비디오 파일이 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 비디오의 속성 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ONNX 모델의 입력 이름 확인
input_name = ort_session.get_inputs()[0].name

# opencv filtering and processing
def filter(frame):
    # 이미지 크기 조정
    resized_frame = cv2.resize(frame, (2048, 1024))

    # 밝기 조정
    resized_frame = mean_brightness(resized_frame)

    # 이미지를 BGR에서 HSV로 변환
    hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

    # gaussian blur
    hsv_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)

    # 특정 색상 필터링 (예제 색상 범위: 변경 필요)
    color_range = np.array([[81, 124, 0], [128, 182, 255]])
    filtered_frame = select_color(hsv_frame, color_range)

    # 모폴로지 연산
    morph_kernel = np.ones((9, 9), np.uint8)
    morph = cv2.morphologyEx(filtered_frame, cv2.MORPH_CLOSE, morph_kernel)

    # 이미지를 numpy 배열로 변환
    input_image = np.array(resized_frame)

    # 각 채널에 대해 히스토그램 평활화 적용
    for i in range(3):
        input_image[:, :, i] = cv2.equalizeHist(input_image[:, :, i])

    # 입력 데이터 형식 변경
    input_image = input_image.transpose(2, 0, 1).astype(np.float32)  # (H, W, C) -> (C, H, W)
    input_image = np.expand_dims(input_image, axis=0)  # (C, H, W) -> (1, C, H, W)

    # HSV 프레임을 다시 RGB로 변환
    hsv_frame_rgb = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
    hsv_input = hsv_frame_rgb.transpose(2, 0, 1).astype(np.float32)
    hsv_input = np.expand_dims(hsv_input, axis=0)

    # morph 프레임을 3채널로 변환
    morph_rgb = cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB)
    morph_input = morph_rgb.transpose(2, 0, 1).astype(np.float32)
    morph_input = np.expand_dims(morph_input, axis=0)

    return input_image, resized_frame, hsv_frame, filtered_frame, morph, hsv_input, morph_input

# 필요한 전처리 함수
def mean_brightness(img):
    fixed = 100  # 이 값 주변으로 평균 밝기 조절함
    m = cv2.mean(img)  # 평균 밝기
    scalar = (-int(m[0]) + fixed, -int(m[1]) + fixed, -int(m[2]) + fixed, 0)
    dst = cv2.add(img, scalar)
    return dst

def select_color(img, range):
    selceted = cv2.inRange(img, range[0], range[1])
    return selceted

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 프레임 필터링 및 처리
    input_image, resized_frame, hsv_frame, filtered_frame, morph_frame, hsv_input, morph_input = filter(frame)

    # 원본 프레임으로 추론 수행
    outputs_orig = ort_session.run(None, {input_name: input_image})
    output_image_orig = outputs_orig[0][0]  # 배치 차원 제거
    output_image_orig = np.argmax(output_image_orig, axis=0)  # 클래스 차원에서 argmax
    output_image_orig = output_image_orig.astype(np.uint8)
    output_color_orig = cv2.applyColorMap(output_image_orig * 15, cv2.COLORMAP_JET)  # 클래스 값을 임의의 스케일로 변환 후 컬러맵 적용
    output_color_orig = cv2.resize(output_color_orig, (input_width, input_height))
    combined_frame_orig = cv2.addWeighted(frame, 0.5, output_color_orig, 0.5, 0)
    cv2.imshow('Original Frame', cv2.resize(frame, (960, 540)))
    cv2.imshow('Segmented Frame (Original)', cv2.resize(combined_frame_orig, (960, 540)))



    # # HSV 프레임으로 추론 수행
    # outputs_hsv = ort_session.run(None, {input_name: hsv_input})
    # output_image_hsv = outputs_hsv[0][0]  # 배치 차원 제거
    # output_image_hsv = np.argmax(output_image_hsv, axis=0)  # 클래스 차원에서 argmax
    # output_image_hsv = output_image_hsv.astype(np.uint8)
    # output_color_hsv = cv2.applyColorMap(output_image_hsv * 15, cv2.COLORMAP_JET)  # 클래스 값을 임의의 스케일로 변환 후 컬러맵 적용
    # output_color_hsv = cv2.resize(output_color_hsv, (input_width, input_height))
    # combined_frame_hsv = cv2.addWeighted(resized_frame, 0.5, output_color_hsv, 0.5, 0)
    # cv2.imshow('HSV Frame', cv2.resize(hsv_frame, (960, 540)))
    # cv2.imshow('Segmented Frame (HSV)', cv2.resize(combined_frame_hsv, (960, 540)))



    # # morph 프레임으로 추론 수행
    # outputs_morph = ort_session.run(None, {input_name: morph_input})
    # output_image_morph = outputs_morph[0][0]  # 배치 차원 제거
    # output_image_morph = np.argmax(output_image_morph, axis=0)  # 클래스 차원에서 argmax
    # output_image_morph = output_image_morph.astype(np.uint8)
    # output_color_morph = cv2.applyColorMap(output_image_morph * 15, cv2.COLORMAP_JET)  # 클래스 값을 임의의 스케일로 변환 후 컬러맵 적용
    # output_color_morph = cv2.resize(output_color_morph, (input_width, input_height))
    # combined_frame_morph = cv2.addWeighted(resized_frame, 0.5, output_color_morph, 0.5, 0)
    # cv2.imshow('Morph Frame', cv2.resize(morph_frame, (960, 540)))
    # cv2.imshow('Segmented Frame (Morph)', cv2.resize(combined_frame_morph, (960, 540)))

    # 사용자가 'q' 키를 누르면 루프를 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 작업 완료 후 자원 해제
cap.release()
cv2.destroyAllWindows()
