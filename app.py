# Fix for ChromaDB + Streamlit Cloud (SQLite version issue)
import os

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import sys
# sys.path hack removed - chatterbox is now local
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load Secrets for Streamlit Cloud
import streamlit as st
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
    # Also set for specific libs just in case
    os.environ["HUGGING_FACE_HUB_TOKEN"] = st.secrets["HF_TOKEN"]

import streamlit as st
import shutil
import soundfile as sf
import numpy as np
import time

from stt.indic_conformer import IndicSTT
from llm.sarvam_client import SarvamLLM
from tts.chatterbox import ChatterboxTTS
from db.vector_store import VectorStore

# Page Config
st.set_page_config(page_title="Indic Voice Agent", layout="wide")
st.title("🇮🇳 Local Indic Voice Agent")

# Paths
AUDIO_DIR = "audio"
INPUT_AUDIO = os.path.join(AUDIO_DIR, "input.wav")
OUTPUT_AUDIO = os.path.join(AUDIO_DIR, "output.wav")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Initialize Resources (Cached)
@st.cache_resource
def load_resources():
    stt = IndicSTT()
    llm = SarvamLLM()
    tts = ChatterboxTTS()
    db = VectorStore()
    return stt, llm, tts, db

stt, llm, tts, db = load_resources()

# UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Interaction")
    
    # Audio Input (Streamlit 1.39+)
    audio_value = st.audio_input("Record Voice")

    if audio_value:
        st.info("Processing...")
        
        # Save input audio
        with open(INPUT_AUDIO, "wb") as f:
            f.write(audio_value.read())
            
        # 1. STT
        st.text("Transcribing...")
        transcript = stt.transcribe(INPUT_AUDIO)
        
        if transcript:
            st.success(f"Transcript: {transcript}")
            
            # 2. LLM (Non-Streaming due to model limitations)
            st.text("Generating Answer...")
            history = db.get_recent_history()
            
            # Using Non-Streaming 
            full_response = llm.generate_response(transcript, conversation_history=history, stream=False)
            st.markdown(f"**Answer:** {full_response}")
            
            # 3. Streaming/Progressive TTS
            st.text("Synthesizing Audio...")
            
            # Initialize Global Player
            from ui.audio_player import init_player, enqueue_audio, reset_player
            init_player()
            reset_player()
            
            import re
            parts = re.split(r'([.|!|?|।])', full_response)
            
            audio_idx = 0
            current_sentence = ""
            
            for part in parts:
                current_sentence += part
                if part in ['.', '!', '?', '।']:
                    if current_sentence.strip():
                        print(f"[App Debug] Synthesizing sentence: {current_sentence}")
                        chunk_path = os.path.join(AUDIO_DIR, f"chunk_{audio_idx}.wav")
                        tts.synthesize(current_sentence, chunk_path)
                        
                        # Enqueue directly to JS player
                        enqueue_audio(chunk_path)
                        
                        audio_idx += 1
                        current_sentence = ""
            
            if current_sentence.strip():
                 print(f"[App Debug] Synthesizing final fragment: {current_sentence}")
                 chunk_path = os.path.join(AUDIO_DIR, f"chunk_{audio_idx}.wav")
                 tts.synthesize(current_sentence, chunk_path)
                 enqueue_audio(chunk_path)
            
            # 5. Store
            db.add_interaction(transcript, full_response, INPUT_AUDIO)
            st.success("Interaction Saved!")
            
        else:
            st.error("Could not transcribe audio.")

with col2:
    st.subheader("📚 Conversation History")
    history = db.get_recent_history(limit=10)
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Debug/Info
with st.sidebar:
    st.info("System Info")
    st.write(f"STT Device: {stt.device}")
    st.write(f"TTS Device: {tts.device}")
