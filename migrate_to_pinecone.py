"""
Simplified migration script using local HuggingFace embeddings for Pinecone upload.
This version uses local CPU for embeddings (free, no API quota needed).
"""

import os
import sys
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import glob

# Load environment variables
load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agri-knowledge-base")

def load_text_files(directory):
    """Load all .txt files from directory"""
    documents = []
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": os.path.basename(file_path)}
                )
                documents.append(doc)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return documents

def main():
    print("=" * 60)
    print("AgriXVision Knowledge Base Migration to Pinecone")
    print("Using Local HuggingFace Embeddings (Free)")
    print("=" * 60)
    
    # Validate API keys
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        sys.exit(1)
    
    print(f"\n✓ Pinecone API key found")
    print(f"✓ Index name: {PINECONE_INDEX_NAME}")
    
    # Step 1: Load documents
    print(f"\n[1/5] Loading documents from {KNOWLEDGE_BASE_DIR}...")
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"❌ Error: {KNOWLEDGE_BASE_DIR} directory not found")
        sys.exit(1)
    
    documents = load_text_files(KNOWLEDGE_BASE_DIR)
    print(f"✓ Loaded {len(documents)} document(s)")
    
    if not documents:
        print("❌ No documents found to migrate")
        sys.exit(1)
    
    # Step 2: Split documents into chunks
    print("\n[2/5] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    print(f"✓ Created {len(texts)} chunks")
    
    # Step 3: Initialize embeddings
    print("\n[3/5] Initializing embeddings...")
    print("Loading local model 'all-MiniLM-L6-v2' (may take a moment)...")
    
    # Using local embeddings (runs on CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✓ Embeddings model initialized")
    
    # Step 4: Initialize Pinecone
    print("\n[4/5] Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' exists. Checking dimensions...")
        index_info = pc.describe_index(PINECONE_INDEX_NAME)
        if index_info.dimension != 384:
            print(f"⚠️ Dimension mismatch! Index has {index_info.dimension}, but model uses 384.")
            print(f"Deleting and recreating index '{PINECONE_INDEX_NAME}'...")
            pc.delete_index(PINECONE_INDEX_NAME)
            time.sleep(10) # Wait for deletion
            
            # Recreate
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("Waiting for new index to be ready...")
            time.sleep(15)
            print("✓ Index recreated successfully")
        else:
            print(f"✓ Using existing index: {PINECONE_INDEX_NAME} (Dimensions match)")
    else:
        print(f"Creating new index: {PINECONE_INDEX_NAME}...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Waiting for index to be ready...")
        time.sleep(15)
        print("✓ Index created successfully")
    
    # Step 5: Upload vectors to Pinecone
    print("\n[5/5] Uploading vectors to Pinecone...")
    print("This may take a few minutes...")
    
    try:
        vectorstore = PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        print(f"✓ Successfully uploaded {len(texts)} vectors to Pinecone!")
        
    except Exception as e:
        print(f"❌ Error uploading to Pinecone: {str(e)}")
        print("If error is 'dimension mismatch', please delete the index in Pinecone console and retry.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify upload
    print("\n" + "=" * 60)
    print("Migration Complete! ✅")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Documents processed: {len(documents)}")
    print(f"  - Chunks created: {len(texts)}")
    print(f"  - Pinecone index: {PINECONE_INDEX_NAME}")
    print(f"  - Embedding model: Local all-MiniLM-L6-v2")
    print(f"\nYou can now restart the backend to use Pinecone for retrieval.")
    print("=" * 60)

if __name__ == "__main__":
    main()
