from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image
from modules import unicode
import io

import os
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

actions = [ 'Back', 'Clear', 'Double', 'Space',
           'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

seq_length = 10
detector = hm.HolisticDetector(min_detection_confidence=0.5)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="..\\models\\multi_hand_gesture_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sen = ""
res = ""
seq = []
action_seq = []
last_action = None

emitted_result = None

# 이미지를 처리하고 예측 결과를 반환하는 함수
def process_image(image_data):
    global sen, res, action_seq, seq, last_action
    consecutive_threshold = 2  # 동일한 예측이 몇 프레임 이상 연속되어야 하는지

    # 이미지를 디코딩하고 전처리
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    """# START
    output_directory = "test_directory"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"debug_image_{current_time}.jpg"
    output_path = os.path.join(output_directory, output_filename)
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    cv2.imwrite(output_path, img)
    print("Image shape", img.shape)
    # END"""

    if img is None:
        print("이미지 디코딩 실패")
        return None

    img = detector.findHolistic(img, draw=True)
    _, right_hand_lmList = detector.findRighthandLandmark(img)

    if right_hand_lmList is not None:

        print("랜드마크 인식 완료")
        joint = np.zeros((42, 2))

        for j, lm in enumerate(right_hand_lmList.landmark):
            joint[j] = [lm.x, lm.y]

        vector, angle_label = Vector_Normalization(joint)
        d = np.concatenate([vector.flatten(), angle_label.flatten()])

        seq.append(d)

        if len(seq) < seq_length:
            return

        # TensorFlow Lite 모델에 입력
        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        input_data = np.array(input_data, dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        y_pred = interpreter.get_tensor(output_details[0]['index'])
        #print(f"모델 출력: {y_pred}")

        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]

        # 모델 자신도 떨어지면 패스
        if conf < 0.95:
            print("모델 자신도 부족")
            return
        
        action = actions[i_pred]
        action_seq.append(action)

        # 충분한 중복이 버퍼 쌓아
        if len(action_seq) < 3:
            return
        print(f'인식 결과 : {action_seq[-1]}')
        # 충분히 쌓였을 때 마지막 3개가 같아야 함
        this_action = '?'
        if action_seq[-1] == action_seq[-2]: # == action_seq[-3]:
            this_action = action
            sen, res = unicode.process_word(sen, action_seq[-1])   

            if last_action != this_action:
                last_action = this_action
            action_seq = []

    else:
        print("랜드마크 인식 실패")
    return res

# WebSocket을 통한 이미지 수신 및 처리
@socketio.on('image')
def handle_image(data):
    global emitted_result

    image_data = data['image']  # 클라이언트에서 전송된 이미지를 가져옴
    print("이미지 수신 완료!")

    result = process_image(image_data)
    print(f"처리 결과 문장: {result}")

    if result == emitted_result:
        print("전송하지 않음")
    else:
        emitted_result = result
        emit('response', {'result': result})  # 처리 결과를 클라이언트로 반환

@socketio.on('reset_session')
def reset_session():
    global res, sen, emitted_result, action_seq, last_action, seq
    res = ""
    sen = ""
    emitted_result = None
    action_seq = []
    last_action = None
    seq = []

    print("세션 초기화 완료")
    emit('session_reset', {'message': 'Session reset successful.'})


    """
    이런 식으로 프론트에서 화상통화에 접속할 때 한번 쏴주면 좋을 듯
    // 소켓 연결이 되어 있는 상태에서 이벤트 전송
    socket.emit('reset_session');

    // 서버에서의 응답 처리
    socket.on('session_reset', (data) => {
    console.log(data.message); // "Session reset successful." 메시지를 출력
    });
    """

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000)
