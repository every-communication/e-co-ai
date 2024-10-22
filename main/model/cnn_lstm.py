import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM, Dropout

class CNN_LSTM(tf.keras.Model):
    def __init__(self, config):
        super(CNN_LSTM, self).__init__()

        self.num_classes = config['num_classes']

        # LSTM 및 Dense 레이어 정의
        self.lstm = LSTM(256, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.dropout = Dropout(0.5)
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(self.num_classes, activation='softmax')  # num_classes를 정수형으로 설정

    def call(self, x_3d):
        # x_3d.shape는 (batch_size, frames, keypoints, 2d-coordinate) 형태
        # 각 프레임의 keypoints와 2D 좌표를 결합
        batch_size = tf.shape(x_3d)[0]
        time_steps = tf.shape(x_3d)[1]
        num_keypoints = tf.shape(x_3d)[2]
        coord_dim = tf.shape(x_3d)[3]  # 2

        # LSTM 입력을 위한 차원 조정
        # Reshape: (batch_size, frames, keypoints * 2)
        x_reshaped = tf.reshape(x_3d, (batch_size, time_steps, num_keypoints * coord_dim))

        # LSTM 레이어
        lstm_output, hidden_state, cell_state = self.lstm(x_reshaped)

        # 마지막 시간 단계의 출력
        x = self.fc1(lstm_output[:, -1, :])
        x = self.dropout(x)
        x = self.fc2(x)
        return x
