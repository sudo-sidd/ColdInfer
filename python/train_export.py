import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import argparse
import os

def export_c_headers(model, test_image):
    with open('model_weights.h', 'w') as f:
        f.write("#pragma once\n\n")
        for name, param in model.named_parameters():
            c_name = name.replace('.', '_')
            flat_data = param.detach().numpy().flatten().astype(np.float32)
            
            if len(param.shape) == 2:
                rows, cols = param.shape
                f.write(f"const float {c_name}[{rows}][{cols}] = {{\n")
                for r in range(rows):
                    row_data = flat_data[r*cols:(r+1)*cols]
                    row_str = ", ".join(map(str, row_data))
                    f.write(f"  {{{row_str}}},\n")
                f.write("};\n\n")
            else:
                length = param.shape[0]
                row_str = ", ".join(map(str, flat_data))
                f.write(f"const float {c_name}[{length}] = {{{row_str}}};\n\n")
                
    with open('test_image_data.h', 'w') as f:
        f.write("#pragma once\n\n")
        flat_data = test_image.detach().numpy().flatten().astype(np.float32)
        row_str = ", ".join(map(str, flat_data))
        f.write(f"const float input_image[{len(flat_data)}] = {{{row_str}}};\n")

# ==========================================
# MODEL 1: MNIST DIGIT CLASSIFICATION
# ==========================================
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

def train_and_export_mnist():
    print("--- Training MNIST Model ---")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = MNIST_MLP()
    
    if os.path.exists("model_weights.MNIST.pt"):
        print("Loading pre-trained MNIST weights...")
        model.load_state_dict(torch.load("model_weights.MNIST.pt"))
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(1): # Reduced to 1 epoch for faster testing
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
                if batch_idx % 200 == 0:
                    print(f"MNIST Epoch {epoch+1} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
        torch.save(model.state_dict(), "model_weights.MNIST.pt")

    model.eval()
    with torch.no_grad():
        test_image, test_target = next(iter(test_loader))
    export_c_headers(model, test_image)
        
    print(f"MNIST True Label: {test_target.item()}")

# ==========================================
# MODEL 2: KEYWORD SPOTTING (KWS)
# ==========================================
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

def train_and_export_kws():
    print("--- Training Keyword Spotting Model (Organic Data) ---")
    import zipfile
    import urllib.request
    import wave
    
    # 1. Download and Extract the Mini Speech Commands Dataset
    data_dir = './data'
    zip_path = os.path.join(data_dir, 'mini_speech_commands.zip')
    extract_dir = os.path.join(data_dir, 'mini_speech_commands')
    
    if not os.path.exists(extract_dir):
        os.makedirs(data_dir, exist_ok=True)
        url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
        print(f"Downloading mini speech commands dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)
            print("Dataset extracted successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise e

    # 2. Scan classes and load WAV files
    all_classes = sorted([d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d)) and not d.startswith('_')])
    print(f"Detected classes: {all_classes}")
    
    model = KWS_MLP()
    
    if os.path.exists("model_weights.KWS.pt"):
        print("Loading pre-trained KWS weights...")
        model.load_state_dict(torch.load("model_weights.KWS.pt"))
        # We still need one test sample to export headers
        class_path = os.path.join(extract_dir, all_classes[0])
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        wav_path = os.path.join(class_path, wav_files[0])
        with wave.open(wav_path, 'rb') as w:
            frames = w.readframes(w.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < 16000:
                audio = np.pad(audio, (0, 16000 - len(audio)), 'constant')
            else:
                audio = audio[:16000]
            test_input = torch.tensor(audio.reshape(1, 1000, 16).mean(axis=2), dtype=torch.float32)
        y_test = torch.tensor([0], dtype=torch.long)
    else:
        X_list = []
        y_list = []
        
        # We will limit to 200 samples per class for training speed (total 1600 samples)
        max_samples_per_class = 200
        
        for label_idx, class_name in enumerate(all_classes):
            class_path = os.path.join(extract_dir, class_name)
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            print(f"Loading class '{class_name}' ({len(wav_files)} files total, using first {max_samples_per_class})...")
            
            count = 0
            for wav_file in wav_files:
                if count >= max_samples_per_class:
                    break
                wav_path = os.path.join(class_path, wav_file)
                try:
                    with wave.open(wav_path, 'rb') as w:
                        n_channels = w.getnchannels()
                        sampwidth = w.getsampwidth()
                        frames = w.readframes(w.getnframes())
                        
                        if sampwidth == 2:
                            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sampwidth == 1:
                            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                        else:
                            continue
                        
                        if n_channels > 1:
                            audio = audio[::n_channels]
                        
                        # Pad or truncate to 16000 samples
                        if len(audio) < 16000:
                            audio = np.pad(audio, (0, 16000 - len(audio)), 'constant')
                        else:
                            audio = audio[:16000]
                        
                        # Downsample to 1000 features
                        features = audio.reshape(1000, 16).mean(axis=1)
                        X_list.append(features)
                        y_list.append(label_idx)
                        count += 1
                except Exception as e:
                    # Skip corrupt files
                    continue
                    
        X_train = torch.tensor(np.array(X_list), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_list), dtype=torch.long)
        
        print(f"Loaded dataset: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
        
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(1):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"KWS Epoch {epoch+1} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
        torch.save(model.state_dict(), "model_weights.KWS.pt")
        
        test_input = X_train[0:1]
        y_test = y_train[0:1]

    model.eval()
    export_c_headers(model, test_input)
    print(f"KWS Sample Label Index: {y_test.item()} (Class: {all_classes[y_test.item()]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mnist", "kws"], required=True)
    args = parser.parse_args()
    
    if args.model == "mnist":
        train_and_export_mnist()
    elif args.model == "kws":
        train_and_export_kws()