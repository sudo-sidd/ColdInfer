"""
vocab.py — Character-level vocabulary for ColdInfer LSTM chatbot.

Builds a compact ASCII vocab, encodes/decodes strings to/from integer indices,
and exports a C++ header with the lookup table baked in for the inference engine.
"""

import json
import os
from typing import List, Optional

# Default character set: printable ASCII (32–126) + newline + padding
DEFAULT_CHARS = (
    "\n "
    + "".join(chr(i) for i in range(33, 127))  # ! through ~
)

PAD_TOKEN = 0      # index 0 reserved for padding
UNK_TOKEN = 1      # index 1 reserved for unknown chars
BOS_TOKEN = 2      # beginning of sequence
EOS_TOKEN = 3      # end of sequence

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


class CharVocab:
    """
    Minimal character-level vocabulary.
    Maps each unique character to an integer index.
    """

    def __init__(self, chars: Optional[str] = None):
        chars = chars or DEFAULT_CHARS
        # Build index: special tokens first, then chars
        self.idx2char: List[str] = SPECIAL_TOKENS + list(chars)
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}
        self.size = len(self.idx2char)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode a string to a list of integer indices."""
        ids = []
        if add_bos:
            ids.append(BOS_TOKEN)
        for ch in text:
            ids.append(self.char2idx.get(ch, UNK_TOKEN))
        if add_eos:
            ids.append(EOS_TOKEN)
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """Decode a list of integer indices back to a string."""
        chars = []
        for i in ids:
            if i < len(self.idx2char):
                c = self.idx2char[i]
                if strip_special and c in SPECIAL_TOKENS:
                    continue
                chars.append(c)
        return "".join(chars)

    def save(self, path: str):
        """Serialize vocab to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"idx2char": self.idx2char}, f, ensure_ascii=False, indent=2)
        print(f"[vocab] Saved vocab ({self.size} tokens) → {path}")

    @classmethod
    def load(cls, path: str) -> "CharVocab":
        """Load vocab from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls.__new__(cls)
        vocab.idx2char = data["idx2char"]
        vocab.char2idx = {c: i for i, c in enumerate(vocab.idx2char)}
        vocab.size = len(vocab.idx2char)
        print(f"[vocab] Loaded vocab ({vocab.size} tokens) ← {path}")
        return vocab

    def export_c_header(self, path: str = "chatbot_vocab.h"):
        """
        Export this vocabulary as a C++ header file.
        Produces a const char array where index i → character.
        Used by the inference engine for decoding logit outputs.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write("#pragma once\n\n")
            f.write(f"// ColdInfer character vocabulary — {self.size} tokens\n")
            f.write(f"constexpr int VOCAB_SIZE = {self.size};\n\n")
            f.write(f"// Special token indices\n")
            f.write(f"constexpr int PAD_IDX = {PAD_TOKEN};\n")
            f.write(f"constexpr int UNK_IDX = {UNK_TOKEN};\n")
            f.write(f"constexpr int BOS_IDX = {BOS_TOKEN};\n")
            f.write(f"constexpr int EOS_IDX = {EOS_TOKEN};\n\n")

            f.write("// idx2char: index → ASCII character (or special token marker)\n")
            f.write(f"const char* const idx2char[{self.size}] = {{\n")
            for i, ch in enumerate(self.idx2char):
                if ch in SPECIAL_TOKENS:
                    escaped = ch  # keep as-is for special tokens
                    f.write(f'  /* {i:3d} */ "{escaped}",\n')
                else:
                    # Escape characters that are special in C strings
                    if ch == '"':
                        escaped = '\\"'
                    elif ch == '\\':
                        escaped = '\\\\'
                    elif ch == '\n':
                        escaped = '\\n'
                    elif ch == '\r':
                        escaped = '\\r'
                    elif ch == '\t':
                        escaped = '\\t'
                    else:
                        escaped = ch
                    f.write(f'  /* {i:3d} */ "{escaped}",\n')
            f.write("};\n\n")

            # Also export char2idx as a 128-entry lookup table for ASCII input
            f.write("// char2idx: ASCII ordinal → vocab index (UNK for unmapped chars)\n")
            f.write(f"const int char2idx[128] = {{\n")
            lookup = [UNK_TOKEN] * 128
            for ch, idx in self.char2idx.items():
                if len(ch) == 1:
                    ordinal = ord(ch)
                    if 0 <= ordinal < 128:
                        lookup[ordinal] = idx
            for i in range(0, 128, 8):
                chunk = lookup[i:i+8]
                f.write("  " + ", ".join(f"{v:3d}" for v in chunk) + ",\n")
            f.write("};\n")
        print(f"[vocab] Exported C++ header → {path}")

    def __repr__(self):
        return f"CharVocab(size={self.size})"


if __name__ == "__main__":
    # Quick sanity check
    vocab = CharVocab()
    print(vocab)
    test = "Hello, World!\nHow are you?"
    encoded = vocab.encode(test, add_bos=True, add_eos=True)
    decoded = vocab.decode(encoded)
    print(f"Original : {repr(test)}")
    print(f"Encoded  : {encoded[:10]}...")
    print(f"Decoded  : {repr(decoded)}")
    assert decoded == test, "Round-trip encode/decode failed!"
    print("Round-trip OK ✓")
