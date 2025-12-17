import torch
import torchaudio
from transformers import AutoModel, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True).to(device)

print("Checking Tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
    print("Tokenizer loaded.")
except Exception as e:
    print(f"Tokenizer failed: {e}")

waveform = torch.randn(1, 32000).to(device) # 2 seconds

langs = ["hi", "hin", "hindi"]
decodings = ["greedy"]

for l in langs:
    for d in decodings:
        try:
            print(f"Testing {l} {d}")
            out = model(waveform, l, d)
            print(f"Result: {out}")
        except Exception as e:
            print(f"Failed: {e}")
