import torch
import torch.nn as nn
import time
import argparse
import numpy as np
import resource
import tensorflow as tf

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128); self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64);  self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

class KWS_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 128); self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64);   self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = x.view(-1, 1000)
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

def get_rss_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def benchmark_pytorch(model, input_tensor, num_runs=1000):
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(input_tensor)  # warmup
        rss_before  = get_rss_kb()
        cpu_before  = resource.getrusage(resource.RUSAGE_SELF)
        wall_start  = time.perf_counter()
        for _ in range(num_runs): model(input_tensor)
        wall_end    = time.perf_counter()
        cpu_after   = resource.getrusage(resource.RUSAGE_SELF)
    avg_us       = int(((wall_end - wall_start) / num_runs) * 1_000_000)
    peak_rss     = get_rss_kb()
    cpu_user_us  = int((cpu_after.ru_utime - cpu_before.ru_utime) * 1_000_000 / num_runs)
    cpu_sys_us   = int((cpu_after.ru_stime - cpu_before.ru_stime) * 1_000_000 / num_runs)
    return avg_us, peak_rss, cpu_user_us, cpu_sys_us

def benchmark_tflite(pytorch_model, input_tensor, num_runs=1000):
    input_shape = input_tensor.shape[1]
    keras_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64,  activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    with torch.no_grad():
        keras_model.layers[0].set_weights([pytorch_model.fc1.weight.numpy().T, pytorch_model.fc1.bias.numpy()])
        keras_model.layers[1].set_weights([pytorch_model.fc2.weight.numpy().T, pytorch_model.fc2.bias.numpy()])
        keras_model.layers[2].set_weights([pytorch_model.fc3.weight.numpy().T, pytorch_model.fc3.bias.numpy()])
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(keras_model).convert()
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp  = interp.get_input_details()
    out  = interp.get_output_details()
    tf_in = input_tensor.numpy()
    for _ in range(10):  # warmup
        interp.set_tensor(inp[0]['index'], tf_in); interp.invoke()
    cpu_before  = resource.getrusage(resource.RUSAGE_SELF)
    wall_start  = time.perf_counter()
    for _ in range(num_runs):
        interp.set_tensor(inp[0]['index'], tf_in)
        interp.invoke()
        interp.get_tensor(out[0]['index'])
    wall_end    = time.perf_counter()
    cpu_after   = resource.getrusage(resource.RUSAGE_SELF)
    avg_us      = int(((wall_end - wall_start) / num_runs) * 1_000_000)
    peak_rss    = get_rss_kb()
    cpu_user_us = int((cpu_after.ru_utime - cpu_before.ru_utime) * 1_000_000 / num_runs)
    cpu_sys_us  = int((cpu_after.ru_stime - cpu_before.ru_stime) * 1_000_000 / num_runs)
    return avg_us, peak_rss, cpu_user_us, cpu_sys_us

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mnist", "kws"], required=True)
    args = parser.parse_args()

    if args.model == "mnist":
        model = MNIST_MLP()
        model.load_state_dict(torch.load("weights/model_weights.MNIST.pt", weights_only=True))
        test_input = torch.randn(1, 784)
    else:
        model = KWS_MLP()
        model.load_state_dict(torch.load("weights/model_weights.KWS.pt", weights_only=True))
        test_input = torch.randn(1, 1000)

    pt_us, pt_rss, pt_cpu_u, pt_cpu_s  = benchmark_pytorch(model, test_input)
    tf_us, tf_rss, tf_cpu_u, tf_cpu_s  = benchmark_tflite(model, test_input)

    print(f"PYTORCH_US:{pt_us}")
    print(f"PYTORCH_RSS_KB:{pt_rss}")
    print(f"PYTORCH_CPU_USER_US:{pt_cpu_u}")
    print(f"PYTORCH_CPU_SYS_US:{pt_cpu_s}")
    print(f"TFLITE_US:{tf_us}")
    print(f"TFLITE_RSS_KB:{tf_rss}")
    print(f"TFLITE_CPU_USER_US:{tf_cpu_u}")
    print(f"TFLITE_CPU_SYS_US:{tf_cpu_s}")