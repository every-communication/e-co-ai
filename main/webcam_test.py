import torch
import cv2
import numpy as np
from model.p3d import P3D199
from model.cnn_lstm import CNN_LSTM
from utils.util import preprocess_video  # 전처리 함수
import argparse

# 인자 파서 설정
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['p3d', 'cnn_lstm'], required=True, help="Choose model: 'p3d' or 'cnn_lstm'")
parser.add_argument('--sequence_length', type=int, default=16, help='Number of frames to predict on')
args = parser.parse_args()

# 모델 로드
if args.model == 'p3d':
    model = P3D199(num_classes=60)
elif args.model == 'cnn_lstm':
    model = CNN_LSTM(num_classes=60)

model.load_state_dict(torch.load(f"{args.model}_model.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 웹캠 캡처
cap = cv2.VideoCapture(0)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)
    
    # 지정된 프레임 수에 도달하면 예측
    if len(frames) == args.sequence_length:
        inputs = preprocess_video(frames)
        inputs = inputs.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            print(f"Predicted sign language class: {predicted.item()}")
        
        frames = []  # 초기화하여 다음 시퀀스를 받음

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
