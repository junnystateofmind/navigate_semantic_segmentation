import torch
import torch.onnx
from models.pidnet import PIDNet, get_pred_model

# 모델 인스턴스 생성
# 체크포인트의 num_classes를 11로 설정
model = PIDNet(num_classes=11, augment=True)

# state_dict 로드
checkpoint = torch.load('PIDNet_S_Camvid_Test.pt', map_location=torch.device('cpu'))

# 'model.' 접두사 제거
state_dict = {key.replace('model.', ''): value for key, value in checkpoint.items() if 'seghead_p.' not in key and 'seghead_d.' not in key}

# 필요한 키만 로드
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

print('Missing keys:', missing_keys)
print('Unexpected keys:', unexpected_keys)

model.eval()

# 더미 입력 데이터 생성 (배치 크기 4, 채널 수 3, 이미지 크기 720x960)
dummy_input = torch.randn(4, 3, 720, 960)

# 모델을 ONNX 형식으로 변환하여 저장
torch.onnx.export(model, dummy_input, "PIDNet_S_Camvid_Test.onnx", verbose=True)
