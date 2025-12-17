from sarvamai import SarvamAI
import os

key = os.getenv("SARVAM_API_KEY", "test")
client = SarvamAI(api_subscription_key=key)

print(f"Type of client.chat: {type(client.chat)}")
print(f"Dir of client.chat: {dir(client.chat)}")

if hasattr(client.chat, 'completions'):
    print(f"Type of client.chat.completions: {type(client.chat.completions)}")
    print(f"Dir of client.chat.completions: {dir(client.chat.completions)}")
    
    import inspect
    print(f"Signature of client.chat.completions: {inspect.signature(client.chat.completions)}")
else:
    print("client.chat has NO 'completions' attribute")
