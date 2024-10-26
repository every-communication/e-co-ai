from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image
import io

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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

seq = []
action_seq = []
last_action = None

# 이미지를 처리하고 예측 결과를 반환하는 함수
def process_image(image_data):
    global last_action, seq, action_seq
    consecutive_threshold = 3  # 동일한 예측이 몇 프레임 이상 연속되어야 하는지

    # 이미지를 디코딩하고 전처리
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
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
            return None  # 충분한 시퀀스가 쌓이지 않으면 None 반환

        # TensorFlow Lite 모델에 입력
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

    return last_action

# WebSocket을 통한 이미지 수신 및 처리
@socketio.on('image')
def handle_image(data):
    image_data = data['image']  # 클라이언트에서 전송된 이미지를 가져옴
    result = process_image(image_data)
    emit('response', {'result': result})  # 처리 결과를 클라이언트로 반환

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
