import argparse
import os
from stt.indic_conformer import IndicSTT
from llm.sarvam_client import SarvamLLM
from tts.indic_parler import IndicParlerTTS
from db.vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Run Voice Agent Pipeline")
    parser.add_argument("--input", type=str, default="audio/input.wav", help="Path to input WAV file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print("Initializing pipeline...")
    stt = IndicSTT()
    llm = SarvamLLM()
    tts = IndicParlerTTS()
    db = VectorStore()

    # 1. STT
    print("Transcribing...")
    transcript = stt.transcribe(args.input)
    print(f"Transcript: {transcript}")

    if not transcript:
        print("STT failed.")
        return

    # 2. LLM
    print("Querying LLM...")
    answer = llm.generate_response(transcript)
    print(f"Answer: {answer}")

    # 3. TTS
    output_path = "audio/output.wav"
    print("Synthesizing speech...")
    tts.synthesize(answer, output_path)
    print(f"Audio saved to {output_path}")

    # 4. DB
    db.add_interaction(transcript, answer, output_path)
    print("Interaction stored.")

if __name__ == "__main__":
    main()
