import sys
import cv2
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image
import time

actions = ['End', 'BackSpace', 'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

seq_length = 10
detector = hm.HolisticDetector(min_detection_confidence=0.3)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\admin\\Documents\\github\\e-co-ai\\models\\multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 테스트 비디오 파일 경로
video_file_path = "C:\\Users\\admin\\Documents\\github\\e-co-ai\\dataset\\temp\\temp2\\temp_video.mp4"
cap = cv2.VideoCapture(video_file_path)

if not cap.isOpened():
    sys.stdout.write(f"Error: Cannot open video file {video_file_path}\n")
    sys.exit()

seq = []
action_seq = []
last_action = None

consecutive_threshold = 3  # 동일한 예측이 몇 프레임 이상 연속되어야 하는지

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    img = detector.findHolistic(img, draw=True)
    _, right_hand_lmList = detector.findRighthandLandmark(img)

    if right_hand_lmList is not None:
        joint = np.zeros((42, 2))
        for j, lm in enumerate(right_hand_lmList.landmark):
            joint[j] = [lm.x, lm.y]

        vector, angle_label = Vector_Normalization(joint)
        d = np.concatenate([vector.flatten(), angle_label.flatten()])

        seq.append(d)

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        input_data = np.array(input_data, dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        y_pred = interpreter.get_tensor(output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]

        if conf >= 0.5:
            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) >= consecutive_threshold and all(x == action for x in action_seq[-consecutive_threshold:]):
                last_action = action
                action_seq = []  # 예측 결과를 갱신한 후 시퀀스를 초기화

                # 액션을 stdout으로 출력
                sys.stdout.write(f'Action: {last_action}\n')
                sys.stdout.flush()

    time.sleep(0.1)  # 100ms 대기
