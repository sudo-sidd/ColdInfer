import torch
import numpy as np
import sys
sys.path.insert(0, "/data/projects/Coldinfer/python")
from vocab import CharVocab
from train_chatbot import TinyLSTM
from export_chatbot import prune_tensor, quantize_tensor_int8, dequantize_int8

vocab = CharVocab.load("weights/chatbot_vocab.json")
state = torch.load("weights/chatbot_best.pt", weights_only=True)
model = TinyLSTM(vocab.size, embed_dim=64, hidden_size=64)
model.load_state_dict(state["model"])
model.eval()

# Let's test with:
# A) Unpruned FP32
# B) Pruned 50% FP32
# C) Quantized INT8 (no pruning)
# D) Quantized INT8 + Pruned 50%

def test_weights(prune=0.0, quant=False):
    m = TinyLSTM(vocab.size, embed_dim=64, hidden_size=64)
    m.load_state_dict(state["model"])
    
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "weight" in name and "embedding" not in name:
                w = p.numpy()
                if prune > 0:
                    w, _ = prune_tensor(w, prune)
                if quant:
                    q, scale = quantize_tensor_int8(w)
                    w = dequantize_int8(q, scale)
                p.copy_(torch.from_numpy(w))
    
    # Run test
    prompt = "Q: hlo\nA: "
    ids = [vocab.char2idx[c] for c in prompt]
    h1, h2 = None, None
    for token_id in ids:
        x = torch.tensor([[token_id]])
        logits, h1, h2 = m(x, h1, h2)
    
    out = []
    next_id = torch.argmax(logits[0, -1]).item()
    for _ in range(80):
        ch = vocab.idx2char[next_id]
        if ch == '\n': break
        out.append(ch)
        logits, h1, h2 = m(torch.tensor([[next_id]]), h1, h2)
        next_id = torch.argmax(logits[0, -1]).item()
    return "".join(out)

print("FP32 unpruned:           ", test_weights(0.0, False))
print("FP32 50% pruned:         ", test_weights(0.5, False))
print("INT8 quantized unpruned: ", test_weights(0.0, True))
print("INT8 + 50% pruned:       ", test_weights(0.5, True))
