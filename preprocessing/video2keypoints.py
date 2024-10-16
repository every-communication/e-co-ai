import os
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

# Mediapipe 설정
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# 비디오 및 키포인트 디렉토리 설정
video_dir = 'dataset/processed_video'
keypoints_dir = 'dataset/keypoints'

if not os.path.exists(keypoints_dir):
    os.makedirs(keypoints_dir)

def extract_keypoints_from_video(video_path):
    all_frame_keypoints = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        face_results = face_detection.process(frame_rgb)

        # 프레임별 좌우 손과 얼굴 경계 상자 데이터를 위한 초기 리스트
        #left_hand_keypoints = np.zeros((21, 2))  # 21개의 랜드마크
        #right_hand_keypoints = np.zeros((21, 2))
        #face_bbox = np.zeros((2, 2))  # 얼굴의 경계 상자 [xmin, ymin, width, height]
        frame_keypoints = np.zeros((44, 2))

        # 손 키포인트 저장
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = hand_results.multi_handedness[i].classification[0].label
                hand_keypoints = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]

                if handedness == 'Left':
                    #left_hand_keypoints = hand_keypoints
                    frame_keypoints[:21] = hand_keypoints
                elif handedness == 'Right':
                    #right_hand_keypoints = hand_keypoints
                    frame_keypoints[21:42] = hand_keypoints

        # 얼굴 경계 상자 저장
        if face_results.detections:
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            #face_bbox = [bbox.xmin, bbox.ymin, bbox.width, bbox.height]
            face_bbox = np.array([
                [bbox.xmin, bbox.ymin], 
                [bbox.xmin + bbox.width, bbox.ymin + bbox.height]
            ])
            frame_keypoints[42:] = face_bbox
        
        # 프레임의 좌우 손 키포인트와 얼굴 경계 상자 정보를 합쳐서 저장
        all_frame_keypoints.append(frame_keypoints)

    cap.release()
    return frame_keypoints

# 비디오 파일에서 키포인트 추출 및 저장
for root, dirs, files in os.walk(video_dir):
    print(root)
    for file in tqdm(files):
        if file.endswith('.mp4') or file.endswith('.mov'):
            video_path = os.path.join(root, file)
            keypoints_data = extract_keypoints_from_video(video_path)

            # NPY 파일로 저장
            vocab = os.path.basename(root)
            output_file = os.path.join(keypoints_dir, vocab, f"{os.path.splitext(file)[0]}.npy")
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))

            np.save(output_file, keypoints_data)  # 하나의 NPY 파일로 통합 저장

print("Keypoints extraction complete!")
