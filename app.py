import streamlit as st
from src.llm import generate_grounded_answer_dynamic

st.set_page_config(
    page_title="10-K Financial QA Bot",
    layout="centered"
)

st.title("10-K Financial Assistant")
st.markdown(
    "Query parsed SEC 10-K filings using layout-aware retrieval. "
    "Select a specific company to apply strict metadata filtering or select 'All Companies' to activate dynamic intent routing."
)

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
    "- **Intent Routing:** Dynamic Multi-Query\n"
    "- **Vector DB:** ChromaDB (Embedded)\n"
    "- **Embeddings:** `all-MiniLM-L6-v2`"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("context"):
            with st.expander("🔍 View Verifiable Source Passages"):
                st.text(message["context"])

if user_query := st.chat_input("Ask a cross-company question or single lookup..."):
    
    # Capture the existing conversation history to pass to the rewriter
    history_to_pass = [msg for msg in st.session_state.messages]
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and pulling cross-company context..."):
            
            # Pass the history_to_pass argument into your pipeline
            response_stream, context = generate_grounded_answer_dynamic(
                query=user_query, 
                ui_filter=company_filter,
                chat_history=history_to_pass
            )
        
        response_placeholder = st.empty()
        
        try:
            full_response = st.write_stream(response_stream)
            
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