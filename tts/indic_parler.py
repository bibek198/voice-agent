import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

class IndicParlerTTS:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[TTS] Loading Indic Parler TTS on {self.device}...")
        
        self.model_id = "ai4bharat/indic-parler-tts"
        
        # Load Model
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_id
        ).to(self.device)
        
        # Load Tokenizers
        # Prompt tokenizer (for the input text to be spoken)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Description tokenizer (for the voice description/prompt)
        # Usually corresponds to the text encoder's tokenizer
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )

    def synthesize(self, text: str, output_path: str, speaker_description: str = "Divya, a female speaker, speaks at a normal pace with a clear and natural tone.") -> str:
        """
        Synthesizes text to audio.
        """
        try:
            # Tokenize description
            input_ids = self.description_tokenizer(
                speaker_description, 
                return_tensors="pt"
            ).input_ids.to(self.device)
            
            # Tokenize prompt (text to speak)
            prompt_input_ids = self.tokenizer(
                text, 
                return_tensors="pt"
            ).input_ids.to(self.device)

            # Generate audio
            generation = self.model.generate(
                input_ids=input_ids, 
                prompt_input_ids=prompt_input_ids
            )
            
            audio_arr = generation.cpu().numpy().squeeze()
            
            # Save audio
            sf.write(output_path, audio_arr, self.model.config.sampling_rate)
            return output_path

        except Exception as e:
            print(f"[TTS Error] {e}")
            return ""

if __name__ == "__main__":
    tts = IndicParlerTTS()
    print("TTS initialized.")
