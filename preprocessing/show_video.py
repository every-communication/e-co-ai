import numpy as np
import os
import cv2

# Mediapipe 설정 (키포인트 시각화를 위해 사용)
video_dir = 'dataset/processed_video'
keypoints_dir = 'dataset/keypoints'

# 저장된 npy 파일을 불러오기
def load_keypoints(npy_path):
    return np.load(npy_path, allow_pickle=True)

# 비디오 파일과 npy 파일을 매칭해 불러오기
for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith('.mp4'):
            video_path = os.path.join(root, file)
            vocab = os.path.basename(root)
            npy_file = os.path.join(keypoints_dir, vocab, f"{os.path.splitext(file)[0]}.npy")

            # npy 파일 로드
            keypoints_data = load_keypoints(npy_file)
            
            # 비디오 캡처
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 키포인트 데이터가 있는 경우에만 처리
                if frame_count < len(keypoints_data):
                    hand_keypoints = keypoints_data[frame_count]

                    # 키포인트 시각화
                    for kp in hand_keypoints:
                        x = int(kp[0] * frame.shape[1])  # x 값 비율로 변환
                        y = int(kp[1] * frame.shape[0])  # y 값 비율로 변환
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 키포인트를 그리기

                # 비디오 프레임 보여주기
                cv2.imshow('Video with Keypoints', frame)

                # ESC 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == 27:  
                    break

                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()
