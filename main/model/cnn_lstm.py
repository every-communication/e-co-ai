import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from .resnet import ResNet101

class CNN_LSTM:
    def __init__(self, num_classes=70):
        super(CNN_LSTM, self).__init__()
        
        # Using ResNet101 as a feature extractor
        self.resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        self.resnet.trainable = False  # Freeze the ResNet layers

        # LSTM 및 Dense 레이어 정의
        self.lstm = LSTM(256, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(num_classes, activation='softmax')

    def forward(self, x_3d):
        # 각 시간 단계별로 루프
        outputs = []
        for t in range(x_3d.shape[1]):
            # ResNet을 사용하여 특징 추출
            x = self.resnet(x_3d[:, t, :, :, :])  # Shape: (batch_size, height, width, channels)
            outputs.append(x)  # 출력 저장

        # LSTM 입력을 위한 출력 스택
        outputs = tf.stack(outputs, axis=1)  # Shape: (batch_size, time_steps, features)

        # LSTM 레이어
        lstm_output, hidden_state, cell_state = self.lstm(outputs)

        # 마지막 시간 단계의 출력
        x = self.fc1(lstm_output[:, -1, :])
        x = self.fc2(x)
        return x
    