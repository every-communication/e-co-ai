import numpy as np
import os
import cv2

# 비디오와 키포인트 디렉토리 설정
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
                    keypoint_info = keypoints_data[frame_count]
                    
                    # 왼손 키포인트
                    left_hand_keypoints = keypoint_info['left_hand']
                    for (x, y) in left_hand_keypoints:
                        if x > 0 and y > 0:
                            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
                    
                    # 오른손 키포인트
                    right_hand_keypoints = keypoint_info['right_hand']
                    for (x, y) in right_hand_keypoints:
                        if x > 0 and y > 0:
                            cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (255, 0, 0), -1)

                    # 얼굴 경계 상자
                    face_bbox = keypoint_info['face_bbox']
                    (xmin, ymin, width, height) = face_bbox
                    start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
                    end_point = (int((xmin + width) * frame.shape[1]), int((ymin + height) * frame.shape[0]))
                    cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 2)

                # 비디오 프레임 보여주기
                cv2.imshow('Video with Keypoints', frame)

                # ESC 키를 누르면 종료
                if cv2.waitKey(3) & 0xFF == 27:
                    break

                frame_count += 1
            
            cap.release()
            cv2.destroyAllWindows()
