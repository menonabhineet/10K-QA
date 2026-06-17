import chromadb
from chromadb.utils import embedding_functions
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL

# Initialize database client and local embedding engine
client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

ALL_COMPANIES = ["AAPL", "CAT", "JPM", "KO", "NVDA", "WMT"]

try:
    collection = client.get_collection(
        name="sec_10k_filings", 
        embedding_function=embedding_func
    )
except ValueError:
    print("Error: Collection not found. Ensure ingestion was run successfully.")
    collection = None

def retrieve_context_multi(query: str, target_tickers: list) -> str:
    """
    Retrieves chunks with dynamic context allocation:
    - If 1 company is targeted: fetches top_k=5 chunks.
    - If multiple companies or ALL are targeted: loops through each target 
      and fetches top_k=3 chunks per company to prevent context starvation.
    """
    if collection is None:
        return "System Error: Database not initialized."

    # If the router or UI specifies a global search across all entities
    if not target_tickers or "ALL" in target_tickers:
        tickers_to_search = ALL_COMPANIES
        chunks_per_company = 3
    else:
        # Clean and filter input tickers to match our valid corpus
        tickers_to_search = [t.upper() for t in target_tickers if t.upper() in ALL_COMPANIES]
        # Apply the rule: 5 chunks for a single company lookup, 3 chunks per company for multi-comparisons
        chunks_per_company = 5 if len(tickers_to_search) == 1 else 3

    # If no valid tickers resolved, fall back to global search space
    if not tickers_to_search:
        tickers_to_search = ALL_COMPANIES
        chunks_per_company = 3

    formatted_context = ""
    source_counter = 1

    # Execute deterministic, isolated vector searches per company
    for ticker in tickers_to_search:
        search_params = {
            "query_texts": [query],
            "n_results": chunks_per_company,
            "where": {"company": ticker}
        }
        
        results = collection.query(**search_params)
        
        if not results['documents'] or not results['documents'][0]:
            continue
            
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]

        for doc, meta in zip(documents, metadatas):
            company = meta.get("company", "UNKNOWN")
            source_file = meta.get("source_file", "Unknown File")
            
            formatted_context += f"\n--- Source {source_counter} ---\n"
            formatted_context += f"Company: {company}\n"
            formatted_context += f"File: {source_file}\n"
            formatted_context += f"Excerpt:\n{doc}\n"
            formatted_context += "-" * 20 + "\n"
            source_counter += 1

    return formatted_context

# Local execution test
if __name__ == "__main__":
    print("Testing single-company routing (Should yield 5 chunks):")
    ctx_single = retrieve_context_multi("What was Apple's total net sales?", ["AAPL"])
    print(f"Total sources fetched: {ctx_single.count('--- Source')}\n")

    print("Testing multi-company comparison routing (Should yield 3 chunks per company -> 6 total):")
    ctx_multi = retrieve_context_multi("Compare R&D for Apple and NVIDIA", ["AAPL", "NVDA"])
    print(f"Total sources fetched: {ctx_multi.count('--- Source')}\n")