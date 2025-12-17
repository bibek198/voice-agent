import os
import random
from typing import Optional
from sarvamai import SarvamAI

class SarvamLLM:
    def __init__(self, model_name: str = "sarvam-2.0"):
        # self.api_key = os.getenv("SARVAM_API_KEY")
        self.api_key = "sk_qh0ufyqk_2fNGIK2lJlBWP5JrcfgT0o7L"
        if not self.api_key:
            print("WARNING: SARVAM_API_KEY not set. LLM will fail.")
        
        # Initialize official SDK client
        self.client = SarvamAI(api_subscription_key=self.api_key)
        self.model_name = model_name
        
        self.system_prompt = """
You are a professor explaining concepts verbally to students.

Rules:
- Speak in Hindi as the base language, written in Devanagari.
- Use English only for technical or academic terms.
- Explain in a conversational, lecture-style tone.
- Use short sentences and natural pauses.
- Do not use bullet points, numbering, markdown, or emojis.
- Do not mention that you are an AI.
- The output will be converted directly to speech.
"""

    def generate_response(self, user_text: str, conversation_history: list = None, stream: bool = False):
        """
        Generates a response using Sarvam AI LLM.
        Args:
            user_text (str): The user's input.
            conversation_history (list): Previous conversation turns.
            stream (bool): Whether to stream the response chunks.
        
        Returns:
            str or Generator: The full response text (if stream=False) or a generator of chunks.
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Disable history for now to prevent context pollution from previous errors
            # if conversation_history:
            #    messages.extend(conversation_history[-4:]) 
            
            messages.append({"role": "user", "content": user_text})

            # Call API
            response = self.client.chat.completions(
                messages=messages,
                stream=stream
            )

            if stream:
                return self._stream_response(response)
            
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[LLM Error] {e}")
            return "क्षमा करें, मैं अभी उत्तर नहीं दे सकता।"

    def _stream_response(self, response):
        """Helper to yield content from streaming response."""
        try:
            import json
            for chunk in response:
                try:
                    # Case 1: Raw SSE String
                    if isinstance(chunk, str):
                        print(f"[LLM Debug] String chunk: {chunk.strip()}")
                        if chunk.strip().startswith("data: "):
                            json_str = chunk.strip()[6:]  # Remove 'data: '
                            if json_str == "[DONE]":
                                break
                            
                            data = json.loads(json_str)
                            # Extract content from dict
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                                    
                    # Case 2: Object with attributes (standard SDK behavior)
                    elif hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, 'content') and delta.content:
                                yield delta.content
                                
                    # Case 3: Byte chunk (decode)
                    elif isinstance(chunk, bytes):
                        s_chunk = chunk.decode("utf-8")
                        # Recurse or handle same as str
                        if s_chunk.strip().startswith("data: "):
                            json_str = s_chunk.strip()[6:]
                            if json_str == "[DONE]":
                                break
                            data = json.loads(json_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]

                except Exception as e:
                    print(f"[Stream Error] processing chunk: {e}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[LLM Error] {e}")
            yield "क्षमा करें, मैं अभी उत्तर नहीं दे सकता।"

if __name__ == "__main__":
    # Test
    llm = SarvamLLM()
    print("Sarvam LLM initialized.")
