import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import os

class ChatterboxTTS:
    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        print(f"[TTS] Loading Chatterbox Multilingual TTS on {self.device}...")
        
        try:
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            print("[TTS] Chatterbox Model loaded successfully.")
        except Exception as e:
            print(f"[TTS] Error loading Chatterbox model: {e}")
            raise e

    def synthesize(self, text: str, output_path: str, language_id: str = "hi") -> str:
        """
        Synthesizes text to audio.
        Args:
            text (str): The text to synthesize.
            output_path (str): Path to save the output wav file.
            language_id (str): Language code (e.g., 'hi' for Hindi, 'en' for English).
        Returns:
            str: Path to the saved audio file.
        """
        try:
            print(f"[TTS] Synthesizing: '{text}' in '{language_id}'")
            # Generate audio
            # Note: The generate method returns a tensor
            wav = self.model.generate(text, language_id=language_id)
            
            # The model output is likely a tensor [1, T] or [T]
            # sf.write expects numpy array.
            wav_cpu = wav.cpu().numpy().squeeze()
            
            # Chatterbox usually works at 24khz or 22050hz, let's check model.sr if available, 
            # otherwise default to what docs say or common practice.
            # The docs used `model.sr` in the example: `ta.save("test-french.wav", wav_french, model.sr)`
            # So we should be able to access self.model.sr
            sample_rate = getattr(self.model, 'sr', 24000) 
            
            sf.write(output_path, wav_cpu, sample_rate)
            return output_path

        except Exception as e:
            print(f"[TTS Error] {e}")
            return ""

if __name__ == "__main__":
    tts = ChatterboxTTS()
    tts.synthesize("नमस्ते, यह एक परीक्षण है।", "test_chatterbox.wav", language_id="hi")
    print("Test complete.")
