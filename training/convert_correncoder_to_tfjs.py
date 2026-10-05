import os
import json
import numpy as np


def convert_correncoder_to_tfjs(pytorch_model_path: str, output_dir: str, config: dict = None):
    """
    Convert trained PyTorch Correncoder model to TensorFlow.js format.
    
    Architecture: 
    - Conv1d(1, 8, kernel=150, padding=20) + ReLU + Dropout(0.5)
    - Conv1d(8, 8, kernel=75, padding=20) + ReLU + Dropout(0.5)
    - Conv1d(8, 8, kernel=50, padding=10) + Sigmoid + Dropout(0.5)
    - ConvTranspose1d(8, 8, kernel=50, padding=10) + Sigmoid
    - ConvTranspose1d(8, 8, kernel=75, padding=20) + ReLU
    - ConvTranspose1d(8, 1, kernel=150, padding=20)
    """
    try:
        import torch
        import tensorflow as tf
        from tensorflowjs.converters import save_keras_model
    except ImportError:
        print("Installing required packages...")
        os.system("pip install torch tensorflow tensorflowjs")
        import torch
        import tensorflow as tf
        from tensorflowjs.converters import save_keras_model

    default_config = {
        'input_length': 1024,
        'n_kernels': [8, 8, 8],
        'kernel_sizes': [150, 75, 50],
        'paddings': [20, 20, 10],
        'dropout': 0.5
    }
    if config:
        default_config.update(config)
    config = default_config

    # Load PyTorch model
    pytorch_model = torch.load(pytorch_model_path, map_location='cpu')
    state_dict = pytorch_model.get('model_state_dict', pytorch_model)
    
    print("PyTorch state dict keys:")
    for k, v in state_dict.items():
        print(f"  {k}: {v.shape}")

    # Build Keras model matching PyTorch architecture
    inputs = tf.keras.Input(shape=(config['input_length'], 1), name='ppg_input')
    x = inputs

    # Encoder
    for i in range(3):
        x = tf.keras.layers.Conv1D(
            config['n_kernels'][i], config['kernel_sizes'][i],
            strides=1, padding='valid',
            name=f'conv1d_{i}'
        )(x)
        if i < 2:
            x = tf.keras.layers.ReLU(name=f'relu_{i}')(x)
        else:
            x = tf.keras.layers.Activation('sigmoid', name=f'sigmoid_{i}')(x)
        x = tf.keras.layers.Dropout(config['dropout'], name=f'dropout_{i}')(x)

    # Decoder
    for i in range(3):
        idx = 2 - i
        x = tf.keras.layers.Conv1DTranspose(
            config['n_kernels'][idx] if i < 2 else 1,
            config['kernel_sizes'][idx],
            strides=1, padding='valid',
            name=f'conv1d_transpose_{i}'
        )(x)
        if i == 0:
            x = tf.keras.layers.Activation('sigmoid', name=f'sigmoid_dec_{i}')(x)
        elif i == 1:
            x = tf.keras.layers.ReLU(name=f'relu_dec_{i}')(x)
        # Last layer (i=2) has no activation

    model = tf.keras.Model(inputs=inputs, outputs=x, name='Correncoder')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # Try to transfer weights (requires careful mapping)
    print("\nKeras model layers:")
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            print(f"  {layer.name}: {[w.shape for w in weights]}")

    # Save model structure (weights need manual transfer)
    os.makedirs(output_dir, exist_ok=True)
    save_keras_model(model, output_dir)
    print(f"\nTF.js model structure saved to {output_dir}")
    print("Note: Weight transfer from PyTorch requires manual mapping due to layer name differences.")
    print("Use the PyTorch model directly for training, then export final weights.")
    
    # Save config for inference
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    return model


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_correncoder_to_tfjs.py <pytorch_model.pth> <output_dir> [config.json]")
        sys.exit(1)
    
    pytorch_path = sys.argv[1]
    output_dir = sys.argv[2]
    config = None
    if len(sys.argv) > 3:
        with open(sys.argv[3]) as f:
            config = json.load(f)
    
    convert_correncoder_to_tfjs(pytorch_path, output_dir, config)