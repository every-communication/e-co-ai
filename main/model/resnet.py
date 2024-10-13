# /model/resnet.py
import tensorflow as tf
from tensorflow.python.keras import layers, models

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        return tf.nn.relu(x)

class ResNet101(tf.keras.Model):
    def __init__(self, num_classes=70):
        super(ResNet101, self).__init__()
        self.in_channels = 64
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self.build_layer(64, 3, stride=1)
        self.layer2 = self.build_layer(128, 4, stride=2)
        self.layer3 = self.build_layer(256, 23, stride=2)
        self.layer4 = self.build_layer(512, 3, stride=2)
        
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def build_layer(self, filters, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != filters:
            downsample = layers.Conv2D(filters, kernel_size=1, strides=stride)

        layers_list = [ResidualBlock(filters, stride=stride, downsample=downsample)]
        self.in_channels = filters
        for _ in range(1, blocks):
            layers_list.append(ResidualBlock(filters))

        return tf.keras.Sequential(layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
