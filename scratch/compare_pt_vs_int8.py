import torch
import sys
sys.path.insert(0, "/data/projects/Coldinfer/python")
from vocab import CharVocab
from train_chatbot import TinyLSTM, generate_sample

vocab = CharVocab.load("weights/chatbot_vocab.json")
state = torch.load("weights/chatbot_best.pt", weights_only=True)
model = TinyLSTM(vocab.size, embed_dim=64, hidden_size=64)
model.load_state_dict(state["model"])
model.eval()

# Let's inspect step-by-step in PyTorch for "Q: hlo\nA: "
prompt = "Q: hlo\nA: "
ids = [vocab.char2idx[c] for c in prompt]

h1, h2 = None, None
for i, token_id in enumerate(ids):
    x = torch.tensor([[token_id]])
    logits, h1, h2 = model(x, h1, h2)
    pred_id = torch.argmax(logits[0, -1]).item()
    print(f"Fed '{prompt[i]}' (idx {token_id}), top pred: '{vocab.idx2char[pred_id]}'")

print("\nGeneration in PyTorch:")
gen_ids = []
next_id = torch.argmax(logits[0, -1]).item()
for _ in range(80):
    ch = vocab.idx2char[next_id]
    if ch == '\n':
        break
    gen_ids.append(ch)
    x = torch.tensor([[next_id]])
    logits, h1, h2 = model(x, h1, h2)
    next_id = torch.argmax(logits[0, -1]).item()

print("".join(gen_ids))
