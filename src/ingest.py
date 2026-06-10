import os
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from config import DATA_DIR, CHROMA_DB_DIR, EMBEDDING_MODEL

def extract_text_preserving_layout(pdf_path):
    """
    Extracts text using pdfplumber to maintain whitespace and spatial layout, 
    which is critical for preventing financial tables from turning into gibberish.
    """
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text preserving visual spacing
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
    return full_text

def chunk_text(text, chunk_size=2000, overlap=400):
    """
    Splits the text into overlapping chunks. 
    A larger chunk size (2000 chars) helps ensure entire tables stay in one chunk.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move forward, but step back by 'overlap' amount to preserve context context
        start += chunk_size - overlap 
    return chunks

def get_metadata(filename):
    """
    Extracts the company ticker from the filename.
    Example: 'aapl-20250927.pdf' -> 'AAPL'
    """
    basename = os.path.splitext(filename)[0]
    company_ticker = basename.split('-')[0].upper()
    return {"company": company_ticker, "source_file": filename}

def main():
    print("Initializing ChromaDB...")
    # Initialize local persistent vector database
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    
    # Set up the local embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    collection = client.get_or_create_collection(
        name="sec_10k_filings",
        embedding_function=embedding_func
    )
    
    # Process each PDF in the data directory
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".pdf"):
            continue
            
        pdf_path = DATA_DIR / filename
        print(f"Processing {filename}...")
        
        metadata = get_metadata(filename)
        text = extract_text_preserving_layout(pdf_path)
        chunks = chunk_text(text)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append(metadata) # Attach company metadata to every single chunk
            ids.append(f"{metadata['company']}_chunk_{i}")
            
        # Add to Vector DB in batches to avoid memory overflow
        batch_size = 150
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            
    print("\nIngestion complete! Database saved to chroma_db/")

if __name__ == "__main__":
    main()