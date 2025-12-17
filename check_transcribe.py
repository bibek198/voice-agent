import torch
from transformers import AutoModel
import torchaudio

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True
).to(device)

print(f"Direct methods: {[d for d in dir(model) if 'transcribe' in d]}")
