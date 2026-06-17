import json
from openai import OpenAI
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL
from src.search import retrieve_context_multi

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

ROUTER_PROMPT = """You are a precise query routing agent for an SEC 10-K database. Your only job is to analyze a user query and determine which companies' financial filings must be searched.

The valid corporate tickers in our corpus are: ["AAPL", "CAT", "JPM", "KO", "NVDA", "WMT"].

Rules:
1. If the query clearly targets one specific company, return a JSON list containing only that ticker, e.g., ["AAPL"].
2. If the query compares multiple specific companies, return a JSON list containing those tickers, e.g., ["AAPL", "NVDA"].
3. If the query is global, aggregate, or mentions no specific company (e.g., "Which company spent the most on R&D?" or "What was the highest revenue recorded?"), return exactly ["ALL"].
4. If the query asks about a company completely outside our corpus (e.g., Tesla, Amazon, Target), return exactly ["ALL"]. This allows our downstream generation engine to securely verify the entire database and state "I don't know."

You must output ONLY a valid JSON list of strings. Do not include markdown formatting, backticks, block formatting, or conversational prose."""

SYSTEM_PROMPT = """You are an expert, precision-focused financial analyst chatbot. Your task is to answer user questions about specific SEC 10-K filings using ONLY the provided context blocks.

Strict Operational Rules:
1. GROUNDING: Base your answer exclusively on the clear facts directly mentioned in the context. Do not make up facts, extrapolate figures, or assume trends.
2. HONESTY ("I don't know"): If the context does not contain the answer, or if the context is empty, or if the user asks about a company not present in the provided context, you MUST reply exactly with: "I don't know. The requested information is not available in the provided 10-K filings." Do not attempt to use outside training knowledge.
3. CITATIONS: When providing facts, you must cleanly state which company's filing or source block you derived the information from.
4. TABLE LITERACY: The context contains tables where columns represent years or categories. Match rows and columns carefully to extract exact figures.

Format your output professionally using markdown."""

def route_query(query: str) -> list:
    """
    Uses the LLM to classify the query intent and extract target tickers 
    for isolated vector database querying.
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )
        
        # Clean potential whitespace or formatting anomalies from the response string
        raw_output = response.choices[0].message.content.strip()
        
        # Strip codeblock backticks if the model accidentally generates them
        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n")[1:-1]
            raw_output = "".join(raw_output).strip()
            
        tickers = json.loads(raw_output)
        if isinstance(tickers, list):
            return tickers
        return ["ALL"]
    except Exception as e:
        # Fall back safely to searching all filings if routing parses incorrectly
        print(f"Routing error encountered: {str(e)}. Falling back to comprehensive search.")
        return ["ALL"]

def generate_grounded_answer_dynamic(query: str, ui_filter: str = None):
    """
    Orchestrates the entire RAG pipeline:
    1. Determines whether to route dynamically or respect a hard UI filter.
    2. Fetches layout-aware context chunks with dynamic context allocation.
    3. Streams the verified response back along with the raw source context.
    """
    # Step 1: Resolve targets
    if ui_filter:
        targets = [ui_filter]
    else:
        targets = route_query(query)
        
    print(f"DEBUG - Query routed to database partitions: {targets}")
    
    # Step 2: Fetch precisely allocated text chunks
    context = retrieve_context_multi(query, target_tickers=targets)
    
    # Step 3: Run streaming generation loop
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context Chunks:\n{context}\n\nUser Question: {query}"}
    ]
    
    response_stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        stream=True
    )
    
    # Yield both the token stream and the generated context string for the UI expander
    return response_stream, context