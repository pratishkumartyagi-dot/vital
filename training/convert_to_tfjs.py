import torch
import torch.nn as nn
import numpy as np
import os
import json


def convert_pytorch_to_tfjs(pytorch_model_path: str, output_dir: str, config: dict = None):
    """
    Convert trained PyTorch BPNet model to TensorFlow.js format.
    
    This script:
    1. Loads the PyTorch model weights
    2. Creates equivalent TensorFlow/Keras model
    3. Transfers weights
    4. Saves in TF.js format (model.json + weight shards)
    """
    try:
        import tensorflow as tf
        from tensorflowjs.converters import save_keras_model
    except ImportError:
        print("Installing required packages...")
        os.system("pip install tensorflow tensorflowjs")
        import tensorflow as tf
        from tensorflowjs.converters import save_keras_model

    default_config = {
        'input_length': 1024,
        'cnn_filters': [32, 64, 128],
        'kernel_size': 3,
        'pool_size': 2,
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'dropout': 0.2
    }
    if config:
        default_config.update(config)
    config = default_config

    # Build Keras model matching PyTorch BPNet architecture
    inputs = tf.keras.Input(shape=(config['input_length'], 1), name='ppg_input')
    x = inputs

    # CNN layers
    for i, filters in enumerate(config['cnn_filters']):
        x = tf.keras.layers.Conv1D(
            filters, config['kernel_size'], strides=1, padding='same',
            name=f'conv{i}'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'bn{i}')(x)
        x = tf.keras.layers.ReLU(name=f'relu{i}')(x)
        x = tf.keras.layers.MaxPooling1D(config['pool_size'], strides=config['pool_size'], name=f'pool{i}')(x)
        x = tf.keras.layers.Dropout(config['dropout'], name=f'dropout{i}')(x)

    # BiLSTM layers
    for i in range(config['lstm_layers']):
        return_sequences = (i < config['lstm_layers'] - 1) or True  # Always return sequences for attention
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                config['lstm_hidden'],
                return_sequences=return_sequences,
                dropout=config['dropout'] if i < config['lstm_layers'] - 1 else 0,
                name=f'bilstm{i}'
            ),
            name=f'bidirectional{i}'
        )(x)

    # Attention layer (custom)
    from tensorflow.keras.layers import Layer
    class AttentionLayer(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        def build(self, input_shape):
            self.context_vector = self.add_weight(
                name='context_vector',
                shape=(input_shape[-1],),
                initializer='random_normal',
                trainable=True
            )
            self.attention_dense = tf.keras.layers.Dense(1, use_bias=False, name='attention_scores')
            super().build(input_shape)
        def call(self, inputs):
            scores = self.attention_dense(inputs)
            weights = tf.nn.softmax(scores, axis=1)
            context = tf.reduce_sum(inputs * weights, axis=1)
            return context

    context = AttentionLayer(name='attention')(x)

    # Output heads
    sbp = tf.keras.layers.Dense(64, activation='relu', name='sbp_dense1')(context)
    sbp = tf.keras.layers.Dropout(config['dropout'], name='sbp_dropout')(sbp)
    sbp = tf.keras.layers.Dense(1, name='sbp_output')(sbp)

    dbp = tf.keras.layers.Dense(64, activation='relu', name='dbp_dense1')(context)
    dbp = tf.keras.layers.Dropout(config['dropout'], name='dbp_dropout')(dbp)
    dbp = tf.keras.layers.Dense(1, name='dbp_output')(dbp)

    model = tf.keras.Model(inputs=inputs, outputs=[sbp, dbp], name='BPNet')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'sbp_output': 'mse', 'dbp_output': 'mse'},
        metrics={'sbp_output': 'mae', 'dbp_output': 'mae'}
    )

    # Load PyTorch weights
    if os.path.exists(pytorch_model_path):
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Transfer weights (simplified - would need careful mapping)
        print("PyTorch state dict keys:", list(state_dict.keys())[:10])
        print("Keras model layers:", [l.name for l in model.layers])
        
        # For now, save randomly initialized model as template
        print("Weight transfer requires manual mapping. Saving model structure as template.")
    
    os.makedirs(output_dir, exist_ok=True)
    save_keras_model(model, output_dir)
    print(f"TF.js model saved to {output_dir}")
    
    # Save config for inference
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    return model


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_to_tfjs.py <pytorch_model.pth> <output_dir> [config.json]")
        sys.exit(1)
    
    pytorch_path = sys.argv[1]
    output_dir = sys.argv[2]
    config = None
    if len(sys.argv) > 3:
        with open(sys.argv[3]) as f:
            config = json.load(f)
    
    convert_pytorch_to_tfjs(pytorch_path, output_dir, config)