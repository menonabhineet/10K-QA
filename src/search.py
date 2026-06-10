import chromadb
from chromadb.utils import embedding_functions
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL

# Initialize the database client and embedding function once
# so we don't reload the model on every single query
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

try:
    collection = client.get_collection(
        name="sec_10k_filings", 
        embedding_function=embedding_func
    )
except ValueError:
    print("Error: Collection not found. Did you run the ingestion script?")
    collection = None

def retrieve_context(query: str, top_k: int = 5, company_filter: str = None) -> str:
    """
    Searches the vector database for chunks most similar to the query.
    Optionally filters by a specific company ticker.
    Returns a formatted string of the context to inject into the LLM prompt.
    """
    if collection is None:
        return "System Error: Database not initialized."

    # Prepare search parameters
    search_params = {
        "query_texts": [query],
        "n_results": top_k,
    }
    
    # If we want to restrict search to a specific company, we use ChromaDB's where filter
    if company_filter:
        search_params["where"] = {"company": company_filter.upper()}

    # Execute the search
    results = collection.query(**search_params)

    # If nothing is found, return an empty context
    if not results['documents'] or not results['documents'][0]:
        return ""

    # Format the retrieved chunks into a clean, readable string with citations
    formatted_context = ""
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        company = meta.get("company", "UNKNOWN")
        source_file = meta.get("source_file", "Unknown File")
        
        # We explicitly wrap the chunks in XML-like tags. 
        # LLMs like DeepSeek respond very well to this structure for grounding.
        formatted_context += f"\n--- Source {i+1} ---\n"
        formatted_context += f"Company: {company}\n"
        formatted_context += f"File: {source_file}\n"
        formatted_context += f"Excerpt:\n{doc}\n"
        formatted_context += "-" * 20 + "\n"

    return formatted_context

# Quick test block to ensure it works
if __name__ == "__main__":
    test_query = "What was the total revenue for Apple?"
    print(f"Testing search for: '{test_query}'\n")
    
    # Notice we can filter exactly to 'AAPL' metadata to improve accuracy
    context = retrieve_context(test_query, top_k=3, company_filter="AAPL")
    print(context)