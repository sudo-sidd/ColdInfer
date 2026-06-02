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
    print("--- Training Keyword Spotting Model ---")
    num_samples = 5000
    X_train = torch.randn(num_samples, 1000)
    y_train = torch.randint(0, 10, (num_samples,))
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    
    model = KWS_MLP()
    
    if os.path.exists("model_weights.KWS.pt"):
        print("Loading pre-trained KWS weights...")
        model.load_state_dict(torch.load("model_weights.KWS.pt"))
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(1):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()
                if batch_idx % 20 == 0:
                    print(f"KWS Epoch {epoch+1} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
        torch.save(model.state_dict(), "model_weights.KWS.pt")

    model.eval()
    X_test = torch.randn(1, 1000)
    y_test = torch.randint(0, 10, (1,)) # We just generate a random one for header export
    export_c_headers(model, X_test)
        
    print(f"KWS True Label: {y_test.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mnist", "kws"], required=True)
    args = parser.parse_args()
    
    if args.model == "mnist":
        train_and_export_mnist()
    elif args.model == "kws":
        train_and_export_kws()