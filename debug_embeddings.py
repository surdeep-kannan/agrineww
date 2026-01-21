
import os
import glob
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("VOYAGE_API_KEY")

class VoyageEmbeddings:
    def __init__(self, api_key, model="voyage-2"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.voyageai.com/v1/embeddings"
    
    def embed_documents(self, texts):
        print(f"Embedding {len(texts)} texts...")
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "input": texts,
                "model": self.model
            }
        )
        if response.status_code != 200:
            print(f"Voyage AI Error: {response.status_code}")
            print(response.text)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

# Load one file content
files = glob.glob("knowledge_base/*.txt")
if not files:
    print("No files found!")
    exit(1)

print(f"Loading {files[0]}...")
with open(files[0], 'r', encoding='utf-8') as f:
    content = f.read()

# Split into chunks (simple split for testing)
chunks = [content[:1000]] # First 1000 chars

print("Testing embeddings...")
embedder = VoyageEmbeddings(api_key)
try:
    embeddings = embedder.embed_documents(chunks)
    print("Success!")
except Exception as e:
    print(f"Failed: {e}")
