import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import tensorflow as tf

# Recreate the architectures for loading weights
class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class KWS_MLP(nn.Module):
    def __init__(self):
        super(KWS_MLP, self).__init__()
        self.fc1 = nn.Linear(1000, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 1000)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def benchmark_pytorch(model, input_tensor, num_runs=1000):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
        
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = model(input_tensor)
        end_time = time.perf_counter()
    
    avg_time_us = ((end_time - start_time) / num_runs) * 1_000_000
    return avg_time_us

def benchmark_tflite(pytorch_model, input_tensor, num_runs=1000):
    # Convert PyTorch to TFLite (Requires exporting to SavedModel first, simplified here via TF)
    # For a purely objective test, we build the equivalent Keras model and copy weights
    input_shape = input_tensor.shape[1]
    
    keras_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    # Transfer weights
    with torch.no_grad():
        keras_model.layers[0].set_weights([pytorch_model.fc1.weight.numpy().T, pytorch_model.fc1.bias.numpy()])
        keras_model.layers[1].set_weights([pytorch_model.fc2.weight.numpy().T, pytorch_model.fc2.bias.numpy()])
        keras_model.layers[2].set_weights([pytorch_model.fc3.weight.numpy().T, pytorch_model.fc3.bias.numpy()])

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tf_input = input_tensor.numpy()
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], tf_input)
        interpreter.invoke()
        
    start_time = time.perf_counter()
    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], tf_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.perf_counter()

    avg_time_us = ((end_time - start_time) / num_runs) * 1_000_000
    return avg_time_us

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mnist", "kws"], required=True)
    args = parser.parse_args()
    
    if args.model == "mnist":
        model = MNIST_MLP()
        model.load_state_dict(torch.load("model_weights.MNIST.pt", weights_only=True))
        test_input = torch.randn(1, 784)
    else:
        model = KWS_MLP()
        model.load_state_dict(torch.load("model_weights.KWS.pt", weights_only=True))
        test_input = torch.randn(1, 1000)

    pt_time = benchmark_pytorch(model, test_input)
    tflite_time = benchmark_tflite(model, test_input)
    
    print(f"PYTORCH_US:{int(pt_time)}")
    print(f"TFLITE_US:{int(tflite_time)}")