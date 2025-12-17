from llm.sarvam_client import SarvamLLM
import sys

print("Initializing SarvamLLM...")
llm = SarvamLLM()

# Override system prompt for testing
llm.system_prompt = "You are a helpful assistant. Reply in Hindi."

prompt = "नमस्ते, आप कौन हैं?"
print(f"Generating response for: '{prompt}'")

try:
    # Test Non-Streaming
    # response = llm.generate_response(prompt, stream=False)
    # But generate_response parses choices[0].message.content
    
    # Let's call client directly to be sure
    print("Calling client.chat.completions(stream=False)...")
    response = llm.client.chat.completions(
        messages=[
           {"role": "system", "content": llm.system_prompt},
           {"role": "user", "content": prompt}
        ],
        stream=False
    )
    
    print(f"Response: {response}")
    print(f"Content: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"\nCaught Exception: {e}")
    import traceback
    traceback.print_exc()
