import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras import ResNet101
from base.base_model import BaseModel

class CNN_LSTM(BaseModel):
    def __init__(self, num_classes=70):
        super(CNN_LSTM, self).__init__()
        
        # Using ResNet101 as a feature extractor
        self.resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        self.resnet.trainable = False  # Freeze the ResNet layers

        # Define the additional layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(300, activation='relu')
        self.lstm = layers.LSTM(256, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(num_classes, activation='softmax')

    def call(self, x_3d):
        # Loop through each time step
        outputs = []
        for t in range(x_3d.shape[1]):
            # Extract features using ResNet
            x = self.resnet(x_3d[:, t, :, :, :])  # Shape: (batch_size, height, width, channels)
            x = self.flatten(x)  # Flatten the output
            x = self.dense1(x)  # Apply dense layer
            outputs.append(x)  # Collect the outputs

        # Stack the outputs for LSTM input
        outputs = tf.stack(outputs, axis=1)  # Shape: (batch_size, time_steps, features)

        # LSTM Layer
        lstm_output, hidden_state, cell_state = self.lstm(outputs)

        # Take the last time step's output
        x = self.fc1(lstm_output[:, -1, :])
        x = self.fc2(x)
        return x

# Example usage
if __name__ == "__main__":
    model = CNN_LSTM(num_classes=70)
    model.build((None, 30, 224, 224, 3))  # Example input shape (batch_size, time_steps, height, width, channels)
    model.summary_with_params()  # Print the summary with trainable parameters