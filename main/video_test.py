import cv2
import numpy as np
import tensorflow as tf
import os
import mediapipe as mp

# Mediapipe 설정
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# 비디오 경로 및 모델 경로 설정
video_path = "../dataset/raw_video/0001/김소희_가치_1.mp4"
model_path = "results/model_best.tflite"

# TFLite 모델 로드
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# 프레임 전처리 및 비디오 크기 조정
def preprocess_frame(frame):
    frame = cv2.resize(frame, (1080, 1080))  # 비디오 프레임을 1080x1080으로 조정
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

# 키포인트 추출 함수
def extract_keypoints(frame):
    frame_keypoints = np.zeros((44, 2))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 키포인트 추출
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            hand_keypoints = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
            if i == 0:  # 첫 번째 손
                frame_keypoints[:21] = hand_keypoints  # 왼손
            else:  # 두 번째 손
                frame_keypoints[21:42] = hand_keypoints  # 오른손

    # 얼굴 경계 상자 추출
    face_results = face_detection.process(frame_rgb)
    if face_results.detections:
        detection = face_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        face_bbox = np.array([[bbox.xmin, bbox.ymin], 
                               [bbox.xmin + bbox.width, bbox.ymin + bbox.height]])
        frame_keypoints[42:] = face_bbox.flatten()  # 얼굴 경계 상자 정보를 추가

    return frame_keypoints

# 비디오 처리 함수
def process_video(video_path, model_interpreter):
    cap = cv2.VideoCapture(video_path)

    input_details = model_interpreter.get_input_details()
    output_details = model_interpreter.get_output_details()

    print(f"Input details: {input_details}")  # 입력 세부 사항 출력

    keypoints_list = []  # 프레임에서 추출한 키포인트를 저장할 리스트

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 전처리 및 크기 조정
        processed_frame = preprocess_frame(frame)

        # 키포인트 추출
        keypoints = extract_keypoints(processed_frame)

        # 입력 데이터 타입에 맞게 변환
        keypoints = keypoints.astype(np.float32)

        # 추출한 키포인트를 리스트에 추가
        keypoints_list.append(keypoints)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 처리 완료 후, 프레임 리스트 확인
    num_frames = len(keypoints_list)
    print(f"Total frames processed: {num_frames}")

    # 256개보다 프레임이 적을 경우 패딩
    if num_frames < 256:
        padding = np.zeros((256 - num_frames, 44, 2), dtype=np.float32)
        keypoints_list.extend(padding)
    # 256개보다 많을 경우 잘라내기
    elif num_frames > 256:
        keypoints_list = keypoints_list[:256]

    # 리스트를 numpy 배열로 변환
    keypoints_array = np.array(keypoints_list)

    # 차원을 맞추기 위해 (1, 256, 44, 2)로 변환
    keypoints_array = np.expand_dims(keypoints_array, axis=0)

    # 모델에 입력하기 위해 차원 맞추기
    model_interpreter.set_tensor(input_details[0]['index'], keypoints_array)

    # 예측 수행
    model_interpreter.invoke()

    # 결과 가져오기
    output_data = model_interpreter.get_tensor(output_details[0]['index'])
    predictions = output_data[0]

    # 예측 결과를 단어로 변환
    predicted_class = np.argmax(predictions)  # 가장 높은 확률의 클래스 인덱스
    print(f'Predicted class index: {predicted_class}')

    cap.release()
    cv2.destroyAllWindows()


# 메인 함수
if __name__ == "__main__":
    model_interpreter = load_model(model_path)
    process_video(video_path, model_interpreter)
