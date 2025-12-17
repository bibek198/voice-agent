import torch
from transformers import AutoModel

try:
    print("Loading model...")
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True
    )
    print(f"Model class: {type(model)}")
    print(f"Forward signature: {model.forward.__code__.co_varnames}")
except Exception as e:
    print(f"Error: {e}")
