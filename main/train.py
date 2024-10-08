import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.models import save_model
from sklearn.preprocessing import LabelEncoder

# OpenCV로 수어 영상 입력 및 프레임 단위로 나누기
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

# MediaPipe를 통해 손의 키포인트 읽기
def get_hand_keypoints(frames):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    keypoints_list = []
    
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                keypoints = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
                keypoints_list.append(keypoints)
    
    return keypoints_list

# 키포인트 시퀀스로 단어 예측
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 단어를 문장으로 변환
def words_to_sentence(predicted_words):
    return ' '.join(predicted_words)

# 비디오 파일 경로 설정
video_path = 'path_to_your_video.mp4'
frames = extract_frames(video_path)
keypoints_list = get_hand_keypoints(frames)

# LSTM 모델 학습을 위한 데이터 준비 (이 부분은 실제 데이터에 맞게 수정 필요)
# 여기에 키포인트 데이터를 X로, 레이블 데이터를 y로 설정
X = np.array(keypoints_list)  # 입력 데이터
y = np.array([0, 1, 2])  # 예시 레이블 (단어의 인덱스)

# 레이블 인코딩
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 모델 생성 및 훈련
input_shape = (X.shape[1], X.shape[2])  # (프레임 길이, 키포인트 차원)
num_classes = len(label_encoder.classes_)
model = create_model(input_shape, num_classes)

# 모델 훈련
model.fit(X, y_encoded, epochs=10, batch_size=32)  # 실제 데이터에 맞게 조정

# 예측
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)
predicted_words = label_encoder.inverse_transform(predicted_classes)

# 문장 생성
sentence = words_to_sentence(predicted_words)
print("생성된 문장:", sentence)
