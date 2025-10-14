"""
Cloud setup: OpenAI embeddings + Qdrant Cloud storage
"""

import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Validate credentials
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file!")
if not QDRANT_URL:
    raise ValueError("❌ QDRANT_URL not found in .env file!")
if not QDRANT_API_KEY:
    raise ValueError("❌ QDRANT_API_KEY not found in .env file!")

print("✅ All credentials loaded successfully!")

CSV_FILE = "enhanced_activities_manual.csv"
COLLECTION_NAME = "business_activities"

# Setup
Settings.embed_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

def create_searchable_text(row):
    parts = [
        f"Activity: {row['Activity Name ']}",
        f"Category: {row['Category']}",
    ]
    
    if pd.notna(row.get('Description')) and row['Description'] != '-':
        parts.append(f"Description: {row['Description']}")
    
    if pd.notna(row.get('Generated Keywords')) and row['Generated Keywords'] not in ['ERROR', '-']:
        parts.append(f"Keywords: {row['Generated Keywords']}")
    
    return "\n".join(parts)

def main():
    print("""
    ====================================================================
       Qdrant Cloud Embedding Setup                                  
       Uploading embeddings to cloud...                       
    ====================================================================
    """)
    
    # Load data
    print(f"\n📂 Loading {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, encoding='latin-1')
    print(f"✅ Loaded {len(df)} activities")
    
    # Create documents
    print(f"\n📝 Creating documents...")
    documents = []
    
    for idx, row in df.iterrows():
        doc = Document(
            text=create_searchable_text(row),
            metadata={
                "code": str(row['Code']),
                "sn": str(row.get('SN', '')),
                "activity_name": str(row['Activity Name ']),
                "category": str(row['Category']),
                "group": str(row.get('Group', '')),
                "description": str(row.get('Description', '')),
                "keywords": str(row.get('Generated Keywords', '')),
                "risk_rating": str(row.get('Risk Rating', 'Low')),
                "related_activities": str(row.get('Related Business Activities', '')),
                "when": str(row.get('When', '')),
                "third_party": str(row.get('Third Party', '')),
            },
            id_=str(row['Code'])
        )
        documents.append(doc)
        
        if (idx + 1) % 100 == 0:
            print(f"   ✓ Created {idx + 1}/{len(df)} documents...")
    
    print(f"✅ Created {len(documents)} documents")
    
    # Connect to Qdrant Cloud
    print(f"\n☁️  Connecting to Qdrant Cloud...")
    print(f"   URL: {QDRANT_URL}")
    
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Generate embeddings and upload
    print(f"\n🚀 Generating embeddings and uploading to Qdrant Cloud...")
    print(f"   Collection name: '{COLLECTION_NAME}'")
    print(f"   Using OpenAI text-embedding-3-small")
    print(f"   This will take 5-10 minutes...")
    print(f"   Please wait...\n")
    
    try:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print(f"\n" + "="*70)
        print(f"✅ SUCCESS! Embeddings uploaded to Qdrant Cloud!")
        print(f"✅ Collection '{COLLECTION_NAME}' is ready!")
        print(f"✅ You can now deploy to Streamlit Cloud!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check your Qdrant API key is correct")
        print(f"2. Check your Qdrant URL format (should include https:// and port)")
        print(f"3. Ensure your Qdrant cluster is running")

if __name__ == "__main__":
    main()