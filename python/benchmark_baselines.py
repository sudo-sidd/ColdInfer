import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import resource
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

def get_rss_kb():
    """Return current RSS in KB from /proc/self/status (Linux) or getrusage."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    # Fallback: ru_maxrss is KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def benchmark_pytorch(model, input_tensor, num_runs=1000):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
        
        rss_before = get_rss_kb()
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        wall_start = time.perf_counter()

        for _ in range(num_runs):
            _ = model(input_tensor)

        wall_end = time.perf_counter()
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        rss_after = get_rss_kb()

    avg_wall_us  = ((wall_end - wall_start) / num_runs) * 1_000_000
    avg_rss_kb   = max(0, rss_after - rss_before)          # delta RSS
    peak_rss_kb  = rss_after                                # absolute peak after run
    cpu_user_us  = (cpu_after.ru_utime - cpu_before.ru_utime) * 1_000_000 / num_runs
    cpu_sys_us   = (cpu_after.ru_stime - cpu_before.ru_stime) * 1_000_000 / num_runs

    return int(avg_wall_us), int(peak_rss_kb), int(cpu_user_us), int(cpu_sys_us)

def benchmark_tflite(pytorch_model, input_tensor, num_runs=1000):
    input_shape = input_tensor.shape[1]
    
    keras_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    # Transfer weights from PyTorch
    with torch.no_grad():
        keras_model.layers[0].set_weights([pytorch_model.fc1.weight.numpy().T, pytorch_model.fc1.bias.numpy()])
        keras_model.layers[1].set_weights([pytorch_model.fc2.weight.numpy().T, pytorch_model.fc2.bias.numpy()])
        keras_model.layers[2].set_weights([pytorch_model.fc3.weight.numpy().T, pytorch_model.fc3.bias.numpy()])

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tf_input = input_tensor.numpy()
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], tf_input)
        interpreter.invoke()

    rss_before = get_rss_kb()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.perf_counter()

    for _ in range(num_runs):
        interpreter.set_tensor(input_details[0]['index'], tf_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])

    wall_end   = time.perf_counter()
    cpu_after  = resource.getrusage(resource.RUSAGE_SELF)
    rss_after  = get_rss_kb()

    avg_wall_us = ((wall_end - wall_start) / num_runs) * 1_000_000
    peak_rss_kb = rss_after
    cpu_user_us = (cpu_after.ru_utime - cpu_before.ru_utime) * 1_000_000 / num_runs
    cpu_sys_us  = (cpu_after.ru_stime - cpu_before.ru_stime) * 1_000_000 / num_runs

    return int(avg_wall_us), int(peak_rss_kb), int(cpu_user_us), int(cpu_sys_us)

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

    pt_time, pt_rss, pt_cpu_user, pt_cpu_sys   = benchmark_pytorch(model, test_input)
    tf_time, tf_rss, tf_cpu_user, tf_cpu_sys   = benchmark_tflite(model, test_input)
    
    print(f"PYTORCH_US:{pt_time}")
    print(f"PYTORCH_RSS_KB:{pt_rss}")
    print(f"PYTORCH_CPU_USER_US:{pt_cpu_user}")
    print(f"PYTORCH_CPU_SYS_US:{pt_cpu_sys}")
    print(f"TFLITE_US:{tf_time}")
    print(f"TFLITE_RSS_KB:{tf_rss}")
    print(f"TFLITE_CPU_USER_US:{tf_cpu_user}")
    print(f"TFLITE_CPU_SYS_US:{tf_cpu_sys}")