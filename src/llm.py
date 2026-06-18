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

REWRITE_PROMPT = """You are a precise query rewriting agent. Your job is to take a user's latest question and rewrite it into a fully standalone search query, using the provided conversation history to resolve any pronouns (like "they", "it", "their") or missing context.

Rules:
1. If the latest question is already fully self-contained (e.g., "What was Apple's revenue in 2025?"), output it exactly as is.
2. If the latest question relies on history (e.g., "What about their net income?"), combine it with the previous context (e.g., "What was Apple's net income in 2025?").
3. Do not answer the question. ONLY output the rewritten query.
4. Do not include conversational filler, formatting, or quotes."""

SUGGESTION_PROMPT = """You are an expert financial analyst guiding a user through complex SEC 10-K filings. 
Based on the retrieved context and the answer you just provided, generate exactly 3 highly relevant, insightful follow-up questions the user should ask to dive deeper into the topic.

Rules:
1. Ensure the questions are distinct from one another.
2. Output ONLY a valid JSON list of 3 strings. Do not include markdown blocks, numbering, or conversational filler.
Example: ["How did their R&D spend impact this?", "What are the key risk factors associated with this growth?", "Compare this margin to their competitors."]"""

SYSTEM_PROMPT = """You are an expert, precision-focused financial analyst chatbot. Your task is to answer user questions about specific SEC 10-K filings using ONLY the provided context blocks.

Strict Operational Rules:
1. GROUNDING: Base your answer exclusively on the clear facts directly mentioned in the context. Do not make up facts, extrapolate figures, or assume trends.
2. HONESTY ("I don't know"): If the context does not contain the answer, or if the context is empty, or if the user asks about a company not present in the provided context, you MUST reply exactly with: "I don't know. The requested information is not available in the provided 10-K filings." Do not attempt to use outside training knowledge.
3. CITATIONS: When providing facts, you must cleanly state which company's filing or source block you derived the information from.
4. TABLE LITERACY: The context contains tables where columns represent years or categories. Match rows and columns carefully to extract exact figures.

Format your output professionally using markdown."""

def rewrite_query_with_history(query: str, chat_history: list) -> str:
    """
    Uses the chat history to resolve pronouns and output a standalone search string.
    """
    # If this is the very first question, no rewrite is needed
    if not chat_history:
        return query
    
    # Format the history cleanly so the LLM doesn't get overwhelmed by massive context chunks
    history_text = ""
    
    # We only need the last 4 interactions to establish conversational context
    recent_history = chat_history[-4:] 
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant responses to just the first 200 characters to keep latency ultra-low
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        history_text += f"{role}: {content}\n"

    prompt_content = f"Conversation History:\n{history_text}\n\nLatest User Question: {query}\n\nRewritten Query:"
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.0
        )
        rewritten_query = response.choices[0].message.content.strip()
        print(f"DEBUG - Raw Query: '{query}' | Rewritten: '{rewritten_query}'")
        return rewritten_query
    except Exception as e:
        print(f"Query rewrite failed: {str(e)}. Falling back to original query.")
        return query

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

def generate_grounded_answer_dynamic(query: str, ui_filter: str = None, chat_history: list = None):
    # Step 1: Pre-process the query using conversational memory
    standalone_query = rewrite_query_with_history(query, chat_history) if chat_history else query
    
    # Step 2: Resolve targets using the NEW STANDALONE query
    if ui_filter:
        targets = [ui_filter]
    else:
        targets = route_query(standalone_query)
        
    print(f"DEBUG - Query routed to database partitions: {targets}")
    
    # Step 3: Fetch precisely allocated text chunks using the STANDALONE query
    context = retrieve_context_multi(standalone_query, target_tickers=targets)
    
    # Step 4: Run streaming generation loop using the STANDALONE query
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context Chunks:\n{context}\n\nUser Question: {standalone_query}"}
    ]
    
    response_stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        stream=True
    )
    
    # Yield both the token stream and the generated context string for the UI expander
    return response_stream, context

def generate_suggested_questions(answer: str, context: str) -> list:
    """
    Analyzes the generated answer and context to suggest 3 contextual follow-up questions.
    """
    # If there was no context (e.g., an out-of-corpus question), don't suggest questions
    if not context or "I don't know" in answer:
        return []
        
    prompt_content = f"Context:\n{context}\n\nGenerated Answer:\n{answer}\n\nGenerate 3 follow-up questions as a JSON list:"
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SUGGESTION_PROMPT},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.7 # Slight temperature increase for question variety
        )
        
        raw_output = response.choices[0].message.content.strip()
        if raw_output.startswith("```"):
            raw_output = raw_output.split("\n")[1:-1]
            raw_output = "".join(raw_output).strip()
            
        questions = json.loads(raw_output)
        return questions[:3] if isinstance(questions, list) else []
    except Exception as e:
        print(f"DEBUG - Suggestion generation failed: {str(e)}")
        return []