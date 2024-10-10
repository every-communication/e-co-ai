import os
import cv2
import json
from tqdm import tqdm

# 비디오 및 키포인트 디렉토리 설정
video_dir = 'C:\\Users\\admin\\Documents\\GitHub\\e-co-ai\\dataset\\raw_video'
keypoints_dir = 'C:\\Users\\admin\\Documents\\GitHub\\e-co-ai\\dataset\\keypoints'

# 비디오 재생
for root, dirs, files in os.walk(video_dir):
    for file in tqdm(files):
        if file.endswith('.mp4') or file.endswith('.mov'):
            video_path = os.path.join(root, file)
            keypoints_file = os.path.join(keypoints_dir, os.path.basename(root), f"{os.path.splitext(file)[0]}_keypoints.json")

            # 키포인트 데이터 로드
            with open(keypoints_file, 'r') as json_file:
                keypoints_data = json.load(json_file)

            # 비디오 캡처
            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 현재 프레임에 해당하는 키포인트를 가져오기
                if str(frame_count) in keypoints_data:
                    current_keypoints = keypoints_data[str(frame_count)]

                    # 키포인트를 비디오 프레임에 그리기
                    for hand_keypoints in current_keypoints:
                        for kp in hand_keypoints:
                            x = int(kp[0] * frame.shape[1])  # x 값 비율로 변환
                            y = int(kp[1] * frame.shape[0])  # y 값 비율로 변환
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 키포인트를 그리기

                # 비디오 프레임 보여주기
                cv2.imshow('Video', frame)

                # ESC 키를 누르면 종료
                if cv2.waitKey(1) & 0xFF == 27:  
                    break

                frame_count += 1  # 다음 프레임으로 이동

            cap.release()
            cv2.destroyAllWindows()
