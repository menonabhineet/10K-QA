import streamlit as st
from src.search import retrieve_context
from src.llm import generate_grounded_answer

# Page Configuration
st.set_page_config(
    page_title="10-K Financial QA Bot",
    layout="centered"
)

# Application Header
st.title("10-K Financial Assistant")
st.markdown(
    "Query parsed SEC 10-K filings using layout-aware retrieval. "
    "Select a specific company to apply strict metadata filtering."
)

# Sidebar Configuration for Filtering
st.sidebar.header("Search Filters")
company_options = {
    "All Companies": None,
    "Apple (AAPL)": "AAPL",
    "Caterpillar (CAT)": "CAT",
    "JPMorgan Chase (JPM)": "JPM",
    "Coca-Cola (KO)": "KO",
    "NVIDIA (NVDA)": "NVDA",
    "Walmart (WMT)": "WMT"
}

selected_label = st.sidebar.selectbox("Target Company Filings", list(company_options.keys()))
company_filter = company_options[selected_label]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "### System Specs\n"
    "- **LLM:** `deepseek-v4-flash`\n"
    "- **Vector DB:** ChromaDB (Embedded)\n"
    "- **Embeddings:** `all-MiniLM-L6-v2`"
)

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Existing Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("🔍 View Verifiable Source Passages"):
                st.text(message["context"])

# Accept User Input
if user_query := st.chat_input("Ask a question about the filings (e.g., What was Apple's R&D spend in 2025?)"):
    
    # Append user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Response
    with st.chat_message("assistant"):
        # 1. Fetch chunks visibly with a clean loading spinner
        with st.spinner("Searching document corpus..."):
            context = retrieve_context(user_query, top_k=5, company_filter=company_filter)
        
        response_placeholder = st.empty()
        
        try:
            # 2. Run the streaming generation engine
            response_stream = generate_grounded_answer(user_query, context)
            full_response = st.write_stream(response_stream)
            
            # 3. Surface raw passages under the stream for explicit verification
            if context:
                with st.expander("🔍 View Verifiable Source Passages"):
                    st.text(context)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "context": context
            })
            
        except Exception as e:
            st.error(f"An error occurred connecting to the inference engine: {str(e)}")