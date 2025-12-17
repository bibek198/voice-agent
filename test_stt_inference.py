import torch
import torchaudio
from transformers import AutoModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True
).to(device)
model.eval()

# Generate dummy audio (1 sec of silence+noise)
waveform = torch.randn(1, 16000).to(device)

print("Running inference...")
try:
    # Trying likely arguments based on inspection
    # lang='hi' ? 
    # decoding='greedy' ?
    # It might return a string or object.
    output = model(waveform, "hi", "greedy") 
    print(f"Output type: {type(output)}")
    print(f"Output: {output}")
except Exception as e:
    print(f"Inference failed: {e}")
