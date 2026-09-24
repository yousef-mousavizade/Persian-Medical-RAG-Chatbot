# Persian Pharmaceutical RAG Chatbot

A prototype Persian-language chatbot that answers medication-related questions using a **Retrieval-Augmented Generation (RAG)** architecture. Built as the final project of team **Kavinari Code** at Fakhra Growth Center, University of Qom, in support of the Sanadagah system's need for a pharmaceutical Q&A module.

## Features

- 🔍 Semantic retrieval over a Persian pharmaceutical knowledge base (4,700+ records, 420 drugs)
- 🧠 Fully local execution — no API key or external cloud service required
- 💬 Short-term conversation memory (handles follow-up questions like "Is this drug safe during pregnancy too?")
- 🛑 Abstention mechanism: refuses to answer instead of fabricating a response when evidence is insufficient
- 🌐 Persian, right-to-left web chat interface

## Architecture

```
Web Chat UI (Persian)
        │
        ▼
Preprocessing  (pandas + hazm)
        │
        ▼
Chunking  (RecursiveCharacterTextSplitter, chunk_size=250)
        │
        ▼
Embedding  (multilingual-e5-large, local)
        │
        ▼
FAISS  (vector database)
        │
        ▼
Retrieval (k=5) + conversation memory
        │
        ▼
Abstention  (retrieval confidence threshold)
        │
        ▼
Generation  (Ollama, qwen2.5:7b-instruct)
        │
        ▼
Answer + sources  (FastAPI)
```

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.13 |
| Preprocessing | pandas, hazm |
| Text splitting | LangChain (`RecursiveCharacterTextSplitter`) |
| Embedding | `intfloat/multilingual-e5-large` (HuggingFace, local) |
| Vector store | FAISS |
| LLM | `qwen2.5:7b-instruct` via Ollama (local) |
| Backend | FastAPI, Uvicorn |
| Frontend | Vanilla HTML / CSS / JavaScript |

## Project Structure

```
Persian-Medical-RAG-Chatbot/
├── data/
│   ├── raw/                  # raw dataset
│   └── processed/            # cleaned data, chunks.jsonl, faiss_index/
├── src/
│   ├── document_processing/  # data cleaning and normalization
│   ├── chunking/             # text splitting
│   ├── embedding/            # FAISS index building
│   ├── retrieval/            # vector search
│   ├── generation/           # answer generation with Ollama
│   ├── fastAPI/              # FastAPI backend
│   └── evaluation/           # evaluation and benchmark scripts
├── frontend/
│   └── UI.html            # chat UI
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and set up Ollama
```bash
# Download from https://ollama.com or https://github.com/ollama/ollama/releases
ollama pull qwen2.5:7b-instruct
```

### 3. Build the knowledge base (first time only)
```bash
cd src/document_processing && python clean_dataset.py
cd ../chunking && python chunk_dataset.py
cd ../embedding && python build_index.py
```

### 4. Run the backend
```bash
cd src/fastAPI
uvicorn main:app --reload
```
The backend runs at `http://127.0.0.1:8000`; interactive API docs are available at `/docs`.

### 5. Run the frontend
Open `frontend/index.html` in a browser (the backend must be running at the same time).

## Evaluation

| Metric | Target | Result |
|---|---|---|
| Retrieval accuracy (Hit Rate@5) | ≥ 65% | **87.7%** (on data representative of real usage) |
| Response time | < 5 seconds | 18.18s average (requires a GPU-equipped server) |

Full evaluation details, including results on the PersianMedQA benchmark and a comparison of Abstention methods, are documented in the project's final report.

## Known Limitations

- Response time on local hardware (4GB GPU) does not yet meet the project's target.
- Automatic citation validation (verifying that answers match their cited sources) is not implemented.
- Conversation memory is kept only in server RAM and is lost on server restart.

## Team

**Kavinari Code** — Seyed Yousef Mousavizadeh, Seyed Ali Fatemehpanah
Supervisor: Dr. Mirarab | Mentors: Mozaffarnia, Elahimanesh
Fakhra Growth Center, University of Qom

## Disclaimer

⚠️ This chatbot is not a substitute for professional medical or pharmacist advice, and answers are based solely on the information available in its dataset.
