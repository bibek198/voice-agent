from sarvamai import SarvamAI
import os

key = "sk_qh0ufyqk_2fNGIK2lJlBWP5JrcfgT0o7L"
# client = SarvamAI(api_subscription_key=key)

prompt = "Explain quantum physics in 3 sentences in Hindi."

print("Testing streaming...")
# We need to simulate the client structure I found earlier
# or just try to use the client directly if I can import it.
# Actually I'll just use the code from sarvam_client.py pattern but standalone.

client = SarvamAI(api_subscription_key=key)
try:
    response = client.chat.completions(
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    print(f"Response type: {type(response)}")
    
    for chunk in response:
        print(f"Chunk type: {type(chunk)}")
        print(f"Chunk: {chunk}")
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
            if content:
                print(f"Content: {content}", end="|", flush=True)

except Exception as e:
    print(f"Error: {e}")
