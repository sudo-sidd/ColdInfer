"""
train_chatbot.py — Train a tiny character-level LSTM chatbot for ColdInfer.

Architecture:  Embedding(vocab_size, embed_dim) → LSTM(hidden) × 2 → Linear(vocab_size)
Target hardware: ESP32 classic (520 KB SRAM / 4 MB flash)
Target size: ~48 KB weights after INT8 quantization + pruning

Usage:
    uv run python/train_chatbot.py --epochs 40 --hidden 64 --embed 64
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from vocab import CharVocab

# ---------------------------------------------------------------------------
# Persona Q&A Dataset
# Formatted as clean Q: ... \nA: ... pairs with question augmentations
# ---------------------------------------------------------------------------
QA_BASE = [
    # Greetings & Common variations
    ("hello", "Hello! I am ColdBot, a tiny AI running on an ESP32 microcontroller. Ask me anything!"),
    ("hi", "Hello! I am ColdBot, running bare-metal on an ESP32 microcontroller."),
    ("hlo", "Hello! I am ColdBot, running bare-metal on an ESP32 microcontroller."),
    ("hey", "Hey there! I am ColdBot, your embedded TinyML assistant."),
    ("good morning", "Good morning! ColdBot is ready for TinyML inference on ESP32."),
    ("greetings", "Greetings! I am ColdBot, running on bare-metal C++."),
    ("how are you", "I am running smoothly on 520 kilobytes of SRAM with zero heap allocations!"),

    # Identity & Persona
    ("who are you", "I am ColdBot, a tiny character-level LSTM chatbot running on ESP32."),
    ("whi are you", "I am ColdBot, a tiny character-level LSTM chatbot running on ESP32."),
    ("who r u", "I am ColdBot, a tiny character-level LSTM chatbot running on ESP32."),
    ("what are you", "I am ColdBot, an autoregressive character-level LSTM inference engine."),
    ("what is your name", "My name is ColdBot, the embedded AI for the ColdInfer engine."),
    ("who built you", "I was built by the ColdInfer research team for ultra-efficient edge AI."),
    ("who made you", "I was created by the ColdInfer project for bare-metal microcontroller AI."),
    ("are you chatgpt", "No, I am not a giant cloud LLM. I am a tiny 78k parameter LSTM for microcontrollers!"),
    ("are you an llm", "I am a TinyML character-level LSTM with 64 hidden units, designed to fit in 520 KB SRAM."),

    # ColdInfer Engine & Architecture
    ("what is coldinfer", "ColdInfer is a bare-metal C++ inference engine with zero dynamic memory allocation."),
    ("how does coldinfer work", "ColdInfer uses static SRAM arenas, INT8 sparse GEMM, and rolling KV hidden buffers."),
    ("what can you do", "I answer questions about TinyML, quantization, and running AI on microcontrollers."),
    ("why are you fast", "I run bare-metal in C++ on internal SRAM without any dynamic heap overhead."),
    ("features", "ColdInfer features INT8 PTQ, sparse weight pruning, rolling context, and ESP-IDF support."),

    # Memory & Hardware (ESP32)
    ("what is esp32", "The ESP32 is a dual-core Xtensa microcontroller with 520 KB SRAM and 4 MB flash."),
    ("how much memory", "I use under 50 kilobytes of flash for weights and about 12 kilobytes of RAM for buffers."),
    ("how much ram", "I use under 12 kilobytes of runtime SRAM. Zero malloc or heap fragmentation!"),
    ("how much flash", "My INT8 quantized weights take approximately 40 kilobytes of flash storage."),
    ("what is sram", "SRAM is fast internal memory. ESP32 has 520 KB, and ColdInfer leaves over 85% free."),
    ("what is flash", "Flash is permanent ROM storage where firmware and INT8 weight tables live."),
    ("what is a microcontroller", "A microcontroller is a compact computer-on-a-chip with limited RAM and flash."),

    # TinyML, Quantization & Pruning
    ("what is tinyml", "TinyML is the field of deploying machine learning onto microcontrollers and low-power sensors."),
    ("what is quantization", "Quantization converts 32-bit float weights into 8-bit integers, shrinking size by 4x."),
    ("what is int8", "INT8 stores weights as 8-bit signed integers, saving 75% memory compared to FP32."),
    ("what is pruning", "Pruning zeroes out negligible weights so sparse GEMM kernels skip them, saving compute."),
    ("what is an lstm", "An LSTM is a recurrent neural network that maintains hidden state context across tokens."),
    ("how fast are you", "On desktop I generate tokens in 250 microseconds, and on ESP32 about 300 milliseconds."),
    ("are you accurate", "I am an ultra-compact research model designed for embedded demonstration."),

    # Courtesies & Endings
    ("goodbye", "Goodbye! Have a great day and keep exploring edge AI!"),
    ("bye", "Goodbye! Let me know if you have more questions about ColdInfer!"),
    ("thank you", "You are very welcome! Glad I could help."),
    ("thanks", "You are welcome! Let me know if you want to test another query."),
    ("help", "Ask me about ColdInfer, ESP32 memory, INT8 quantization, or TinyML benchmarks!"),
]

def generate_augmented_corpus() -> str:
    lines = []
    for q, a in QA_BASE:
        # Generate variations: lowercase, uppercase, with question mark, without
        lines.append(f"Q: {q}\nA: {a}\n")
        lines.append(f"Q: {q}?\nA: {a}\n")
        lines.append(f"Q: {q.capitalize()}\nA: {a}\n")
        lines.append(f"Q: {q.capitalize()}?\nA: {a}\n")
        lines.append(f"Q: {q.upper()}\nA: {a}\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset(Dataset):
    """
    Sliding-window character-sequence dataset.
    Each sample is (input_seq, target_seq) where target is input shifted by 1.
    """

    def __init__(self, text: str, vocab: CharVocab, seq_len: int = 64):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = torch.tensor(vocab.encode(text), dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TinyLSTM(nn.Module):
    """
    Two-layer character-level LSTM chatbot.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for lstm in [self.lstm1, self.lstm2]:
            for name, p in lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

    def forward(self, x, hidden1=None, hidden2=None):
        emb = self.embedding(x)
        out1, hidden1 = self.lstm1(emb, hidden1)
        out2, hidden2 = self.lstm2(out1, hidden2)
        logits = self.fc_out(out2)
        return logits, hidden1, hidden2

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def estimated_flash_kb(self, bits: int = 32) -> float:
        return self.param_count() * bits / 8 / 1024


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    os.makedirs("weights", exist_ok=True)
    checkpoint_path = "weights/chatbot.pt"

    # Build vocab & corpus
    corpus = generate_augmented_corpus()
    vocab = CharVocab()
    vocab.save("weights/chatbot_vocab.json")
    print(f"[train] Vocab size: {vocab.size} tokens")
    print(f"[train] Corpus: {len(corpus):,} characters across {len(QA_BASE)} Q&A themes")

    # Dataset & loader
    dataset = CharDataset(corpus, vocab, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"[train] Dataset: {len(dataset):,} sequences of length {args.seq_len}")

    # Model
    model = TinyLSTM(vocab.size, embed_dim=args.embed, hidden_size=args.hidden)
    print(f"[train] Model params: {model.param_count():,}")
    print(f"[train] Estimated flash FP32: {model.estimated_flash_kb(32):.1f} KB")
    print(f"[train] Estimated flash INT8: {model.estimated_flash_kb(8):.1f} KB")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-4
    )

    model.train()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss = 0.0
        total_tokens = 0

        for x, y in loader:
            optimizer.zero_grad()
            logits, _, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab.size), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        scheduler.step()
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 20.0))
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | ppl={perplexity:.2f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s",
            flush=True,
        )

        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1, "loss": avg_loss},
            checkpoint_path,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "loss": avg_loss},
                "weights/chatbot_best.pt",
            )

        if (epoch + 1) % 10 == 0:
            sample = generate_sample(model, vocab, prompt="hlo", max_len=80)
            print(f"  [sample: hlo] -> {repr(sample)}", flush=True)

    print(f"\n[train] Done. Best loss: {best_loss:.4f}")
    return model, vocab


# ---------------------------------------------------------------------------
# Clean Generation Function
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_sample(
    model: TinyLSTM,
    vocab: CharVocab,
    prompt: str = "hello",
    max_len: int = 120,
    temperature: float = 0.01,
) -> str:
    model.eval()
    if not prompt.startswith("Q:"):
        formatted_prompt = f"Q: {prompt}\nA: "
    else:
        formatted_prompt = prompt

    ids = vocab.encode(formatted_prompt, add_bos=False)
    input_tensor = torch.tensor([ids], dtype=torch.long)

    logits, hidden1, hidden2 = model(input_tensor)
    output_chars = []

    # First token to sample comes from logits of the last prompt token
    last_logits = logits[0, -1]
    if temperature <= 0.01:
        next_id = int(torch.argmax(last_logits))
    else:
        probs = torch.softmax(last_logits / temperature, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1))

    for _ in range(max_len):
        ch = vocab.idx2char[next_id]
        if ch == "\n" or ch == "<EOS>" or ch == "<PAD>":
            break
        output_chars.append(ch)

        x = torch.tensor([[next_id]], dtype=torch.long)
        out, hidden1, hidden2 = model(x, hidden1, hidden2)
        step_logits = out[0, 0]
        if temperature <= 0.01:
            next_id = int(torch.argmax(step_logits))
        else:
            probs = torch.softmax(step_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1))

    return "".join(output_chars)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ColdInfer LSTM chatbot")
    parser.add_argument("--epochs",     type=int,   default=35,   help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--seq-len",    type=int,   default=64,   help="Sequence length")
    parser.add_argument("--hidden",     type=int,   default=64,   help="LSTM hidden size")
    parser.add_argument("--embed",      type=int,   default=64,   help="Embedding dimension")
    parser.add_argument("--lr",         type=float, default=5e-3, help="Learning rate")
    args = parser.parse_args()

    model, vocab = train(args)

    print("\n--- Model Verification ---")
    for q in ["hello", "hlo", "who are you", "what is coldinfer", "what is esp32", "how much memory"]:
        print(f"  [{q}] -> {generate_sample(model, vocab, q)}")
