from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Step 1 - Load PDF
print("🔄 Loading PDF...")
extracted_data = load_pdf_file("data/")
print(f"✅ Loaded {len(extracted_data)} pages")

# Step 2 - Split into chunks
print("🔄 Splitting text...")
text_chunks = text_split(extracted_data)
print(f"✅ Created {len(text_chunks)} chunks")

# Step 3 - Load embeddings model
print("🔄 Loading embeddings model...")
embeddings = download_hugging_face_embeddings()
print("✅ Embeddings ready")

# Step 4 - Upload to Pinecone
print("🔄 Uploading to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name="medical-chatbot"
)
print("✅ Done! Data uploaded to Pinecone successfully.")