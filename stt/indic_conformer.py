import torch
import soundfile as sf
import os
import torchaudio
from transformers import AutoModel, AutoTokenizer

class IndicSTT:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[STT] Loading Indic Conformer on {self.device}...")
        
        # Load Model
        self.model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Tokenizer is usually needed for decoding if model doesn't return text
        # But IndicASRModel might return text. We'll keep tokenizer just in case or for debug.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "ai4bharat/indic-conformer-600m-multilingual",
                trust_remote_code=True
            )
        except:
            self.tokenizer = None

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes the given audio file to Hindi (Devanagari) text.
        """
        try:
            # Load audio using soundfile
            audio_data, sr = sf.read(audio_path)
            waveform = torch.tensor(audio_data, dtype=torch.float32)
            
            # Handle channels
            if waveform.ndim > 1 and waveform.shape[1] > 1:
                waveform = torch.mean(waveform, dim=1, keepdim=True).squeeze()
            elif waveform.ndim > 1:
                waveform = waveform.squeeze()
            
            # Add batch dim [1, samples]
            waveform = waveform.unsqueeze(0)
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            
            waveform = waveform.to(self.device)

            # Inference
            with torch.no_grad():
                # Correct signature: wav, lang, decoding
                # decoding must be 'ctc' or 'rnnt' (model default 'ctc')
                output = self.model(waveform, "hi", decoding="ctc")
            
            if output is None:
                print("[STT] Model returned None. Check inputs.")
                return ""
            
            # Output is expected to be string or list of strings
            if isinstance(output, list):
                return output[0]
            return str(output)

        except Exception as e:
            print(f"[STT Error] {e}")
            return ""

if __name__ == "__main__":
    # Test block
    stt = IndicSTT()
    print("STT initialized.")
