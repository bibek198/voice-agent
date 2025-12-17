# Local Indic Voice Agent

A local, end-to-end voice agent for Hindi (academic/technical style) running on macOS Apple Silicon.

## Architecture

1.  **STT**: `ai4bharat/indic-conformer-600m-multilingual` (via Transformers AutoModel)
2.  **LLM**: Sarvam AI (`sarvam-2.0` via official SDK)
3.  **TTS**: `ai4bharat/indic-parler-tts` (via Parler-TTS)
4.  **DB**: ChromaDB (Vector Store)
5.  **UI**: Streamlit

## Setup

1.  **Create Environment**:
    ```bash
    conda env create -f env.yml
    conda activate myAgent
    ```

2.  **Set API Key**:
    You must set your Sarvam AI API key:
    ```bash
    export SARVAM_API_KEY="your_api_key_here"
    ```

## Usage

### Run UI (Preferred)
```bash
streamlit run app.py
```
- Click "Record Voice" to speak.
- View transcript and listen to the AI response.

### Run CLI Script
```bash
python run_pipeline.py --input audio/input.wav
```

## Structure
- `stt/`: Speech-to-Text module
- `llm/`: LLM client
- `tts/`: Text-to-Speech module
- `db/`: Vector database
- `audio/`: Audio storage

## Notes
- First run will download models (~2GB+).
- Uses MPS acceleration on Mac if available.
