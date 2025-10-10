from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("NEO4J_URI"))
print(os.getenv("NEO4J_USER"))
print(os.getenv("NEO4J_PASSWORD"))

# Test if OpenRouter is reachable
import requests

try:
    response = requests.get("https://openrouter.ai/api/v1/models")
    print(f"OpenRouter status: {response.status_code}")
    
    # Test a simple request
    test_payload = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": "Say 'Hello'"}]
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        json=test_payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"Test request status: {response.status_code}")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"Error: {e}")