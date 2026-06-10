# 10-K Grounded Financial Assistant

A QA chatbot built to perform hyper-accurate, grounded question-answering over dense, table-heavy SEC 10-K financial filings. 

This implementation bypasses high-abstraction agentic frameworks in favor of an explicit, layout-aware data pipeline designed to prevent financial data hallucinations and preserve the visual structure of corporate balance sheets.

---

## Quickstart & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Clone and Initialize Environment
```bash
# Navigate to project directory
cd 10K-QA

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install exact dependencies
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a .env file in the root directory:
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

I have used DeepSeek because I had previously used it in one of my projects and got good results and also had some credits leftover.

### 4. Run Ingestion Pipeline
The six raw 10-K documents I downloaded from the EDGAR website were these - 
* **Apple Inc. (AAPL):** Fiscal year ended September 27, 2025 (`aapl-20250927.pdf`)
* **Caterpillar Inc. (CAT):** Fiscal year ended December 31, 2025 (`cat-20251231.pdf`)
* **JPMorgan Chase & Co. (JPM):** Fiscal year ended December 31, 2025 (`jpm-20251231.pdf`)
* **The Coca-Cola Company (KO):** Fiscal year ended December 31, 2025 (`ko-20251231.pdf`)
* **NVIDIA Corporation (NVDA):** Fiscal year ended January 25, 2026 (`nvda-20260125.pdf`)
* **Walmart Inc. (WMT):** Fiscal year ended January 31, 2026 (`wmt-20260131.pdf`)

Place the six raw 10-K PDFs inside the data/ directory and execute the ingestion script to build the local vector store:
```bash
python src/ingest.py
```

### 5. Launch the Web Interface

```bash
streamlit run app.py
```

## Decision Log: Architecture & Trade-offs

### 1. Parsing & Chunking Strategy
**Choice:** pdfplumber + Character-based overlapping chunks (Size: 2000, Overlap: 400).

**Why:** Standard PDF parsers extract text sequentially, merging multi-column financial tables into unreadable, single-paragraph strings. pdfplumber was chosen because it extracts text while rigidly preserving visual whitespace layout. Combined with a larger chunk size (2000 characters), this ensures entire multi-year financial tables remain structurally intact within a single chunk, allowing the LLM to successfully perform row-column alignment.

### 2. Vector Store & Embedding Model
**Choice:** ChromaDB (Embedded) + all-MiniLM-L6-v2 (Local Sentence-Transformers).

**Why:** To guarantee a zero-configuration setup for the reviewer, an embedded database was mandatory. ChromaDB runs locally without Docker or cloud instances. The all-MiniLM-L6-v2 model runs entirely on local CPU, offering lightning-fast embedding generation at zero cost, while natively handling financial terminology mapping (e.g., matching "revenue" with "net sales").

### 3. Retrieval Strategy & Omission of Hybrid/Reranking
**Choice:** Bi-Encoder Retrieval + Hard Metadata Filtering.

**Why:** I intentionally omitted BM25 hybrid search and Cross-Encoder rerankers to optimize compute and precision. By utilizing deterministic metadata tagging ({"company": TICKER}) at ingestion, the search space is instantly isolated to the target document. A standard bi-encoder searching exclusively within Apple's chunks achieves near-perfect precision, rendering complex hybrid merging or high-latency local rerankers redundant.

### 4. Design Trade-off: Evaluation Transparency over Minimalism
**Choice:** Surfacing the full, raw text of the retrieved chunks inside a Streamlit expander rather than hiding them or heavily stylizing the citations.

**Why:** In a consumer-facing application, dumping 10,000 characters of raw 10-K text into the UI is poor design. However, for this assessment, the primary goal is verifiable grounding. I intentionally exposed the raw context blocks so reviewers can instantly inspect the pipeline's output. By opening the expander, a reviewer can visually verify that the `pdfplumber` chunking successfully preserved the spatial alignment of the financial tables, and confirm the LLM is strictly synthesizing from the provided text rather than hallucinating.

## Current System Weaknesses & Vulnerabilities
Through comprehensive stress testing, two critical structural limits of the current architecture were exposed:

### 1. Global Context Aggregation (The "Highest Revenue" Problem):
When asking cross-company aggregate questions (e.g., "Which company had the highest revenue in 2025?"), a flat vector search with top_k=5 fails. It can only fetch chunks from the 2 or 3 documents that most closely match the query terms semantically, leaving out the remaining companies completely.

### 2. Information Truncation via Top-K Limits:
When asking queries that require checking an exhaustive list across all files (e.g., "Which of the six companies list AI as a risk?"), the system experiences information bias. It surfaces the most intense risk-prose chunks from 3 companies but cuts off the others due to the hard top-k boundary.

## What I'd Do With Another Week

To evolve this prototype, I would prioritize the following upgrades:

1. **Structured Metadata Extraction (Hybrid SQL + Vector Search):** To permanently resolve global aggregation failures (like comparing the highest revenues across all files), I would introduce an LLM-powered extraction pipeline at ingestion. Key financial metrics would be parsed directly into a structured database schema. Global quantitative queries would route to deterministic SQL, while open-ended narrative queries would route to the vector store.
2. **Parent-Child (Hierarchical) Chunking:** I would decouple the retrieval index from the generation context. By indexing small semantic child chunks (sentences/individual rows) but retrieving their larger parent blocks (full pages/entire sections), the system would drastically improve its ability to surface needle-in-a-haystack data points (like obscure footnotes) without sacrificing systemic context.
3. **Automated CI/CD Evaluation Loop (RAGAS):** Manual verification does not scale. I would integrate an automated grading framework to evaluate changes to chunk profiles and system prompts continuously, measuring mathematical Faithfulness and Context Recall programmatically on every commit.

## Comprehensive Evaluation Report
I checked 21 QA pairs by hand to evaluate the system, I divided the questions into 4 categories, Single-Fact Lookups, Cross-Company Comparisons, Out-of-Corpus, Complex/Deliberate Failure Candidates. 

The system answered 19 out of the 21 questions correctly, out of which 10 were single fact lookups, 4 were cross company comparisons, 4 were out of corpus where the system said "I don't know". Out of the 3 complex questions asked, the system answered 1 where it tested advanced reasoning across disjointed context blocks and for the other two it wasn't able to fetch enough context with a topk=5 constraint.

The QA pairs can be seen in the `QA Pairs.txt` file in the repository.