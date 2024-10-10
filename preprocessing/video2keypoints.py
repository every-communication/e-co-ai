import os
import json
import cv2
import mediapipe as mp
from tqdm import tqdm

# Mediapipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# 비디오 및 키포인트 디렉토리 설정
video_dir = 'C:\\Users\\admin\\Documents\\GitHub\\e-co-ai\\dataset\\raw_video'
keypoints_dir = 'C:\\Users\\admin\\Documents\\GitHub\\e-co-ai\\dataset\\keypoints'

if not os.path.exists(keypoints_dir):
    os.makedirs(keypoints_dir)

# 비디오 파일에서 키포인트 추출 및 저장
for root, dirs, files in os.walk(video_dir):
    for file in tqdm(files):
        if file.endswith('.mp4') or file.endswith('.mov'):
            video_path = os.path.join(root, file)
            keypoints_data = {}

            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 이미지에서 손 키포인트 감지
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    keypoints = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand_keypoints = []
                        for landmark in hand_landmarks.landmark:
                            hand_keypoints.append((landmark.x, landmark.y))  # x, y 좌표 저장
                        keypoints.append(hand_keypoints)
                    keypoints_data[str(frame_count)] = keypoints  # 프레임 번호를 키로 사용

                frame_count += 1

            cap.release()

            # JSON 파일로 저장
            vocab = os.path.basename(root)  # 디렉토리 이름에서 vocab 추출
            output_file = os.path.join(keypoints_dir, vocab, f"{os.path.splitext(file)[0]}_keypoints.json")
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            with open(output_file, 'w') as json_file:
                json.dump(keypoints_data, json_file, indent=4)

print("Keypoints extraction complete!")
