import tensorflow as tf
from tensorflow.python.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_size, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_size)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_size)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_transformer_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = TransformerBlock(embed_size=64, num_heads=4, ff_dim=64)(inputs)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# Parameters
input_shape = (10, 64)  # Example shape: 10 words with 64-dimensional embeddings
num_classes = 20  # Number of output classes (e.g., vocabulary size)

# Model creation
transformer_model = create_transformer_model(input_shape, num_classes)
transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
transformer_model.summary()
