from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL
from search import retrieve_context

# Initialize the OpenAI client configured to point to DeepSeek's API endpoint
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

SYSTEM_PROMPT = """You are an expert, precision-focused financial analyst chatbot. Your task is to answer user questions about specific SEC 10-K filings using ONLY the provided context blocks.

Strict Operational Rules:
1. GROUNDING: Base your answer exclusively on the clear facts directly mentioned in the context. Do not make up facts, extrapolate figures, or assume trends.
2. HONESTY ("I don't know"): If the context does not contain the answer, or if the context is empty, or if the user asks about a company not present in the provided context (e.g., Tesla), you MUST reply exactly with: "I don't know. The requested information is not available in the provided 10-K filings." Do not attempt to use outside training knowledge.
3. CITATIONS: When providing facts, you must cleanly state which company's filing or source block you derived the information from.
4. TABLE LITERACY: The context contains tables where columns represent years or categories. Match rows and columns carefully to extract exact figures.

Format your output professionally using markdown."""

def generate_grounded_answer(query: str, company_filter: str = None):
    """
    Retrieves context from the vector DB, structures the prompt, 
    and yields a stream of responses from the DeepSeek model.
    """
    # 1. Retrieve relevant source passages
    context = retrieve_context(query, top_k=5, company_filter=company_filter)
    
    # 2. Structure messages for the model
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context Chunks:\n{context}\n\nUser Question: {query}"}
    ]
    
    # 3. Request a streaming response for a real-time chat feel
    response_stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,  # Zero temperature minimizes creative drift and hallucination
        stream=True
    )
    
    # Yield content chunks as they arrive from the API
    for chunk in response_stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Quick test routine
if __name__ == "__main__":
    test_query = "What was the total net sales for Apple in 2025?"
    print(f"Query: {test_query}\nAnswer: ", end="")
    
    # Stream the output directly to the console
    for token in generate_grounded_answer(test_query, company_filter="AAPL"):
        print(token, end="")
    print("\n" + "="*40)
    
    test_out_of_bounds = "What was Tesla's revenue in 2025?"
    print(f"Query: {test_out_of_bounds}\nAnswer: ", end="")
    for token in generate_grounded_answer(test_out_of_bounds):
        print(token, end="")
    print()