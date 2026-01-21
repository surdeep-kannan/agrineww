
import requests
import json
import sys

def test_chatbot(question="How much fertilizer should I use for rice?"):
    url = "http://localhost:8001/ask-chatbot"
    payload = {
        "user_id": "test_script_user",
        "question": question
    }
    
    print(f"Sending question: {question}")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What are the common diseases in tomato?"
    test_chatbot(q)
