import os
import sys
# sys.path hack removed
import torch
import soundfile as sf
from tts.chatterbox import ChatterboxTTS
from transformers import AutoModel
import torchaudio

device = "mps" if torch.backends.mps.is_available() else "cpu"

def generate_audio():
    print("Generating audio with Chatterbox...")
    tts = ChatterboxTTS(device=device)
    
    prompt = "नमस्ते भारत, यह एक परीक्षण है।"
    output_path = "test_audio.wav"
    
    saved_path = tts.synthesize(prompt, output_path, language_id="hi")
    
    if saved_path:
        audio_data, sr = sf.read(saved_path)
        print(f"Audio saved to {saved_path}. Rate: {sr}")
        return saved_path, sr
    else:
        print("Failed to generate audio.")
        return None, None

def test_stt(path):
    if not path:
        print("Skipping STT test due to TTS failure.")
        return

    print("Testing STT...")
    model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True).to(device)
    
    # Load and Resample
    audio_data, sr = sf.read(path)
    waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0) # Add batch dim [1, samples] if mono
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform = waveform.to(device)
    print(f"Waveform shape: {waveform.shape}")
    
    # Inference
    out = model(waveform, "hi", "greedy")
    print(f"STT Output: {out}")

if __name__ == "__main__":
    path, _ = generate_audio()
    test_stt(path)
