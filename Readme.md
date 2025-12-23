```markdown
# Infra KB Quality Checker 🧠🔧

An end-to-end Retrieval-Augmented Generation (RAG) project for evaluating and improving an **Infrastructure (Infra) Knowledge Base**.  

The app lets you:

- Ask real-world SCOM/SCCM/Windows infra questions
- Retrieve the most relevant KB articles from a vector database
- Generate an answer using a local LLM (Ollama)
- Score the quality of that answer using multiple evaluation metrics
- Visualize everything in an interactive Streamlit UI

This is designed as a **portfolio-grade** project that looks and behaves like something you’d build for a real ops/team environment.

---

## 1. Problem Statement

Infra teams usually have tons of scattered documentation: OneNote pages, Word docs, wikis, Confluence, random text files, etc.  

Common problems:

- Hard to know if the KB actually covers the most frequent issues
- No objective way to measure KB quality
- New engineers don’t know which docs to trust
- “Ask the senior guy” becomes the default support flow

This project answers two questions:

1. **Can the KB answer this question right now?**
2. **If not, what’s missing and how bad is the gap?**

---

## 2. High-Level Architecture

**Components:**

- **Embedding Model**: `BAAI/bge-small-en-v1.5` (384-dim sentence embeddings)
- **Vector DB**: Qdrant (local server) for semantic search
- **LLM**: Ollama running `llama3.2` locally
- **App**: Streamlit UI for interactive Q&A + metrics
- **Eval Layer**: LLM-based evaluation of retrieval and answer quality

**Flow:**

1. Offline:
   - Load KB text file
   - Split into snippets
   - Embed with BGE model
   - Upsert into Qdrant collection

2. Online (Dashboard):
   - User asks a question in the UI
   - Embed question → search Qdrant → get top‑K docs
   - LLM generates an answer using only those docs as context
   - Evaluation layer scores:
     - Context Relevancy
     - Answer Completeness
     - Faithfulness (no hallucinations)
     - Precision@K (retrieval quality)
   - UI shows answer, sources, and metrics

---

## 3. Features

### Core

- 🔎 **Semantic search** over infra KB (SCOM/SCCM/Windows troubleshooting)
- 🤖 **Local LLM answering** via Ollama (`llama3.2`)
- 📚 **Source transparency** – shows which KB snippets were used
- 📊 **Quality metrics per question**:
  - Context relevancy
  - Answer completeness
  - Faithfulness
  - Precision@3
  - Overall KB quality score (weighted)

### Dashboard

- Chat-style interface for asking infra questions
- Per-question metrics and source snippets
- Session stats (questions asked, average quality score)
- Export session as JSON (for analysis or training data)
- Configurable:
  - Number of retrieved sources
  - Toggle metrics and source visibility

---

## 4. Tech Stack

- **Python 3**
- **Qdrant** (self-hosted, vector DB)
- **SentenceTransformers** (`BAAI/bge-small-en-v1.5`)
- **Ollama** (`llama3.2` model)
- **Streamlit** for the UI

---

## 5. Project Structure

```
infra-kb-quality-checker/
├── app/
│   └── streamlit_dashboard.py       # Interactive Q&A + metrics dashboard
├── data/
│   └── kb_docs/
│       └── infra_sample_kb.txt      # Infra KB snippets (SCOM, SCCM, Windows)
├── scripts/
│   ├── load_kb_into_qdrant.py       # One-time KB load into Qdrant
│   ├── check_kb_quality.py          # CLI-based evaluation and report
│   └── evaluation_metrics.py        # Core metric calculations
├── reports/                         # Generated JSON reports (gitignored)
├── qdrant_storage/                  # Qdrant local data (gitignored)
├── requirements.txt
└── README.md
```

---

## 6. Setup & Installation

### 6.1. Prerequisites

- Python 3.10+  
- Qdrant server running locally (e.g. Docker)  
- Ollama installed and working  
- GPU is optional but recommended (for embeddings + LLM)

**Run Qdrant via Docker:**

```
docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Install Ollama (Windows/macOS/Linux)**  
Follow instructions from https://ollama.com and make sure this works:

```
ollama --version
ollama pull llama3.2
```

### 6.2. Clone & Install Dependencies

```
git clone <your-repo-url> infra-kb-quality-checker
cd infra-kb-quality-checker

pip install -r requirements.txt
```

---

## 7. Load the Knowledge Base into Qdrant

This script:

- Reads `infra_sample_kb.txt`
- Splits it into snippets
- Embeds with BGE model
- Writes vectors + payload into Qdrant

```
python scripts/load_kb_into_qdrant.py
```

Expected console output (example):

```
Collection infra_kb already exists
Loaded 10 snippets from KB
Model moved to GPU
Inserted 10 points into infra_kb

Search results for: SCOM agent heartbeat failure troubleshooting
Score: 0.8390846
Text: [DOC1] Title: SCOM Heartbeat Failure ...
...
```

If you see 3 reasonable matches and similarity scores, your retrieval pipeline is working.

---

## 8. Run the Interactive Dashboard

Start the Streamlit app:

```
streamlit run app/streamlit_dashboard.py
```

This should open at `http://localhost:8501`.

### Example Usage

1. Type a question like:
   - `Why is my SCOM agent not sending heartbeats?`
   - `What should I do if SCOM database backup fails?`
2. The app will:
   - Retrieve top‑K KB snippets from Qdrant
   - Ask `llama3.2` to answer based only on those snippets
   - Compute metrics
3. You’ll see:
   - The generated answer
   - Key metrics (Overall, Relevancy, Completeness, Faithfulness, Precision@3)
   - Optional: the source documents used

---

## 9. Evaluation Metrics (What They Mean)

The interesting part of this project is the **evaluation layer**, not just the RAG pipeline.

For each question, the system computes:

### 1. Context Relevancy

> How much of the retrieved content is actually relevant to the question?

Rough intuition:  
Relevant sentences / total sentences in retrieved documents.

### 2. Answer Completeness

> Does the KB content fully answer the question?

The evaluator:

- Breaks the question into 3–5 sub-questions
- Labels each as ANSWERED / PARTIAL / NOT_ANSWERED
- Converts that into a score between 0 and 1

### 3. Faithfulness

> Is the answer grounded in the KB, or is the LLM hallucinating?

Checks whether the claims in the answer can be traced back to KB snippets.

### 4. Precision@3

> Out of the top 3 retrieved snippets, how many were actually relevant?

Simple retrieval quality metric:
- 1.0 → all top 3 are useful
- 0.0 → none of them are useful

### 5. Overall Quality Score

A weighted combination of the above:

- Context Relevancy – 25%
- Answer Completeness – 35%
- Faithfulness – 20%
- Precision@3 – 20%

This gives a single number per question that you can track over time.

---

## 10. Offline CLI Evaluation (Optional)

Besides the UI, you can also run batch evaluation from the command line:

```
python scripts/check_kb_quality.py
```

This:

- Runs a fixed set of test questions
- Logs metrics per question
- Writes:
  - `reports/kb_quality_detailed_YYYYMMDD_HHMMSS.json`
  - `reports/kb_quality_summary_YYYYMMDD_HHMMSS.json`

These are what the dashboard originally used before the live Q&A mode.

---

## 11. Ideas for Further Improvement

If you want to extend this project further:

- Add more KB documents and show before/after metrics
- Support multiple collections (SCOM, SCCM, Windows, Linux, etc.)
- Add authentication for the dashboard
- Log anonymized real queries from users (if deployed in a team)
- Add a “Suggest new KB article” button when completeness is low

---

## 12. Why This Project Matters

This project isn’t just a toy chatbot.

It demonstrates:

- Practical use of RAG for internal tooling
- Realistic infra troubleshooting domain (SCOM/SCCM/Windows)
- Evaluation of both retrieval and generation, not just “does it answer?”
- Integration of:
  - Vector DB (Qdrant)
  - Embeddings (BGE)
  - Local LLM (Ollama)
  - Web UI (Streamlit)

That combination is exactly the kind of end-to-end system people expect from an engineer building AI tools for ops/SRE/support teams.