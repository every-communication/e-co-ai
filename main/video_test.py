import torch
import cv2
import numpy as np
from model.p3d import P3D
from model.cnn_lstm import CNN_LSTM
from utils.util import preprocess_video
import argparse

# 인자 파서 설정
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['p3d', 'cnn_lstm'], required=True, help="Choose model: 'p3d' or 'cnn_lstm'")
parser.add_argument('--video', type=str, required=True, help='Path to the video file')
args = parser.parse_args()

# 모델 로드
if args.model == 'p3d':
    model = P3D(num_classes=60)
elif args.model == 'cnn_lstm':
    model = CNN_LSTM(num_classes=60)

model.load_state_dict(torch.load(f"{args.model}_model.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 비디오 파일 읽기
cap = cv2.VideoCapture(args.video)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# 전처리 및 예측
inputs = preprocess_video(frames)
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    print(f"Predicted sign language class: {predicted.item()}")
