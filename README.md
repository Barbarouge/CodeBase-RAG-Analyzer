# Automatic Code Base Analysis and Architectural Querying System Using RAG & LLMs

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Local_Inference-Ollama_%7C_Qwen_2.5-purple.svg)](https://ollama.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

> **BSc Computer Science and Digitisation Dissertation Project**  
> **Author:** Barbaros Yeşilova

## 📌 Overview & Problem Statement
While Large Language Models (LLMs) excel at code generation and explanation, deploying them on enterprise-scale codebases faces two major hurdles:
1. **Context Fragmentation in Traditional RAG:** Standard character/token-based text splitters break syntactic boundaries (classes, functions), diluting semantic context and causing hallucinations.
2. **The Privacy vs. Performance Dilemma:** Sending proprietary source code to third-party cloud APIs introduces severe intellectual property and data leakage risks.

This repository implements a **privacy-first, model-agnostic Retrieval-Augmented Generation (RAG) architecture** that parses source code into an **Abstract Syntax Tree (AST)**. Instead of treating code as flat text, it indexes structural entities (`FunctionDef`, `ClassDef`, `AsyncFunctionDef`) with an adaptive recursive fallback mechanism for monolithic blocks (>4,000 characters).

---

## 🏗️ System Architecture
* **AST Ingestion & Hybrid Chunking:** Uses Python's native `ast` module to isolate complete syntactic blocks. Monolithic nodes exceeding 4,000 characters gracefully degrade via `RecursiveCharacterTextSplitter` (`chunk_size=4000`, `chunk_overlap=200`) while retaining hierarchical metadata (`{node.name}_part{i+1}`).
* **Vectorization & Storage:** Encodes structural chunks locally using `nomic-embed-text` via Ollama and persists them into **ChromaDB**.
* **Model-Agnostic Generation Layer:** Built on **LangChain (LCEL)** and **Streamlit**, allowing seamless runtime switching between:
  * 🔒 **On-Premises (Air-Gapped):** `Qwen 2.5 (7B)` via Ollama for 100% data sovereignty.
  * ☁️ **Cloud API:** `Google Gemini 3.5 Flash` for low-latency, high-capacity pedagogical design pattern mapping.

---

## 📊 Experimental Results & Benchmarks
The system was benchmarked against a naive baseline (`RecursiveCharacterTextSplitter` with 1,000-character chunks and 200-character overlap) on the open-source **Flask** repository across a curated 15-question architectural ground-truth dataset.

### 1. Retrieval Performance ($k=3$)
| Evaluation Metric | Baseline (Naive Recursive) | AST-Aware (Proposed) |
| :--- | :---: | :---: |
| **Hit Rate (@3)** | 86.67% | **86.67%** |
| **Mean Reciprocal Rank (MRR)** | 0.678 | **0.789** |
| **Query Execution Time** | 0.99s | **0.31s** |

*Key Finding:* While both methods achieved identical Hit Rates at $k=3$, the **AST-aware parser improved MRR by +16.3% (0.789 vs. 0.678)**, consistently ranking the exact target module at **Rank 1** by preserving syntactic boundaries.

### 2. Downstream Generation & Local RAGAS Evaluation
Evaluated using the **RAGAS** framework configured with a privacy-preserving local LLM-as-a-judge (`Qwen 2.5 7B` via `ChatOllama`) on an Apple Silicon (M4 Pro) host:

| Evaluation Metric | Qwen 2.5 7B (Local / Ollama) | Gemini 3.5 Flash (Cloud API) |
| :--- | :---: | :---: |
| **Mean Inference Latency** | 27.35s | **20.02s** |
| **Answer Relevance** | **79.6%** | 73.4% |
| **Faithfulness (Grounding)** | **81.3%** | 55.1% |

*Key Finding:* Although Gemini 3.5 Flash offered faster inference and rich pedagogical formatting, it relied heavily on pre-trained assumptions outside the retrieved context (55.1% Faithfulness). Conversely, **local Qwen 2.5 (7B) achieved 81.3% Faithfulness**, proving that on-premises SLMs paired with AST-RAG strictly adhere to retrieved codebase evidence without leaking proprietary data.

---

## 🚀 Getting Started

### Prerequisites
1. **Python 3.11+**
2. **[Ollama](https://ollama.com/)** installed and running locally.
3. Pull the required local embedding and inference models via Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:7b
   ollama pull qwen2.5-coder:7b
