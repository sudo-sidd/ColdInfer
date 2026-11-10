"""
scratch/test_train.py — Quick test of Q&A dataset and char-LSTM generation quality.
"""

import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.insert(0, "/data/projects/Coldinfer/python")

from vocab import CharVocab
from train_chatbot import TinyLSTM

QA_PAIRS = [
    # Greetings & Common variations
    ("hello", "Hello! I am ColdBot, a tiny AI running on an ESP32 microcontroller."),
    ("hi", "Hello! I am ColdBot, a tiny AI running on an ESP32 microcontroller."),
    ("hlo", "Hello! I am ColdBot, a tiny AI running on an ESP32 microcontroller."),
    ("hey", "Hey! I am ColdBot, a tiny AI running on an ESP32 microcontroller."),
    ("good morning", "Good morning! I am ColdBot, running bare-metal on ESP32."),
    ("greetings", "Greetings! I am ColdBot, your embedded AI assistant."),
    
    # Identity & Creator
    ("who are you", "I am ColdBot, a tiny character-level LSTM inference engine on ESP32."),
    ("whi are you", "I am ColdBot, a tiny character-level LSTM inference engine on ESP32."),
    ("who r u", "I am ColdBot, a tiny character-level LSTM inference engine on ESP32."),
    ("what are you", "I am ColdBot, an autoregressive character LSTM running bare-metal."),
    ("who made you", "I was created by the ColdInfer project for edge AI research."),
    ("who built you", "I was built by the ColdInfer project team as a research prototype."),
    
    # ColdInfer & Features
    ("what is coldinfer", "ColdInfer is a bare-metal C++ inference engine with zero dynamic allocation."),
    ("how does coldinfer work", "ColdInfer runs INT8 sparse GEMM on static SRAM arenas with rolling KV buffers."),
    ("what can you do", "I can answer questions, run fast inference, and prove that TinyML fits on 520 KB SRAM."),
    ("what is your purpose", "To demonstrate high-efficiency autoregressive neural inference on low-cost microcontrollers."),
    
    # Hardware & Memory (ESP32)
    ("what is esp32", "The ESP32 is a dual-core microcontroller with 520 KB SRAM and 4 MB flash."),
    ("how much memory", "I use under 50 kilobytes of flash for weights and about 12 kilobytes of RAM for buffers."),
    ("how much ram", "I use under 12 kilobytes of internal SRAM at runtime. Zero heap allocations!"),
    ("how much flash", "My INT8 quantized weights take about 40 kilobytes of flash storage."),
    ("what is sram", "SRAM is fast on-chip memory. ESP32 has 520 KB, and ColdInfer leaves most of it free."),
    ("what is flash", "Flash is non-volatile storage where firmware and INT8 weights live permanently."),
    
    # TinyML & Compression
    ("what is tinyml", "TinyML is the field of running machine learning on microcontrollers and edge sensors."),
    ("what is quantization", "Quantization converts 32-bit float weights into 8-bit integers, shrinking size by 4x."),
    ("what is int8", "INT8 stores weights as 8-bit signed integers from -128 to 127, saving 75% memory."),
    ("what is pruning", "Pruning zeroes out small weights so sparse matrix math skips them, saving compute."),
    ("what is an lstm", "An LSTM is a recurrent neural network that remembers sequence context over time."),
    ("how fast are you", "On desktop I generate tokens in 250 microseconds, on ESP32 about 300 milliseconds."),
    ("are you accurate", "I am a compact demonstration model optimized for embedded microcontrollers."),
    ("goodbye", "Goodbye! Have a great day and keep exploring TinyML!"),
    ("bye", "Goodbye! Let me know if you have more questions about ColdInfer!"),
    ("thank you", "You are welcome! Let me know if you need anything else."),
    ("thanks", "You are welcome! Glad I could help."),
]

def build_corpus():
    lines = []
    # Repeat the pairs with slight casing and punctuation variations
    for q, a in QA_PAIRS:
        lines.append(f"Q: {q}\nA: {a}\n")
        lines.append(f"Q: {q}?\nA: {a}\n")
        lines.append(f"Q: {q.capitalize()}\nA: {a}\n")
        lines.append(f"Q: {q.capitalize()}?\nA: {a}\n")
    return "".join(lines)

corpus = build_corpus()
print(f"Corpus length: {len(corpus)} characters")

vocab = CharVocab()
data = torch.tensor(vocab.encode(corpus), dtype=torch.long)

class SlidingDataset(Dataset):
    def __init__(self, tensor_data, seq_len=64):
        self.data = tensor_data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len], self.data[idx+1 : idx + self.seq_len + 1]

dataset = SlidingDataset(data, seq_len=64)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TinyLSTM(vocab.size, embed_dim=64, hidden_size=64)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-4)

for epoch in range(40):
    total_loss = 0
    tokens = 0
    for x, y in loader:
        optimizer.zero_grad()
        logits, _, _ = model(x)
        loss = criterion(logits.reshape(-1, vocab.size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.numel()
        tokens += y.numel()
    scheduler.step()
    avg_loss = total_loss / tokens
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/40 | loss={avg_loss:.4f} | ppl={math.exp(avg_loss):.2f}")

# Test generation
@torch.no_grad()
def test_gen(prompt):
    model.eval()
    full_prompt = f"Q: {prompt}\nA: "
    ids = vocab.encode(full_prompt)
    x = torch.tensor([ids], dtype=torch.long)
    _, h1, h2 = model(x)
    
    # Autoregressive
    out = []
    # Next token starts from logits of the last input
    # In PyTorch: we pass the last token
    next_id = ids[-1]
    curr_h1, curr_h2 = h1, h2
    for _ in range(120):
        logits, curr_h1, curr_h2 = model(torch.tensor([[next_id]]), curr_h1, curr_h2)
        next_id = torch.argmax(logits[0, 0]).item()
        ch = vocab.idx2char[next_id]
        if ch == '\n' or ch == '<EOS>':
            break
        out.append(ch)
    return "".join(out)

print("\n--- Test Generations ---")
for test_q in ["hlo", "whi are you", "what is coldinfer", "what is esp32", "how much memory"]:
    print(f"[{test_q}] -> {test_gen(test_q)}")
