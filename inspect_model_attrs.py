import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True
)

print("Attributes:", dir(model))
if hasattr(model, 'transcribe'):
    print("Has transcribe method")
if hasattr(model, 'decode'):
    print("Has decode method")
