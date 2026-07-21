# CodeBase-RAG-Analyzer

This repository contains the official implementation of the retrieval-augmented generation (RAG) system developed for the dissertation: *"Automatic Code Base Analysis and Architectural Querying System Using RAG and Large Language Models"*.

## Overview
CodeBase-RAG-Analyzer is a structural, model-agnostic decision-support system designed to query complex software architectures. Unlike traditional naive text-chunking RAG systems, it utilizes Abstract Syntax Tree (AST) parsing to maintain the semantic and hierarchical integrity of the codebase. It implements a hybrid inference architecture, balancing data privacy via local models (Qwen 2.5) with high pedagogical depth via cloud models (Gemini 3.5 Flash).

## Key Features
* **AST-Based Chunking:** Parses Python code into structured `FunctionDef`, `ClassDef`, and `AsyncFunctionDef` nodes.
* **Hybrid Model Architecture:** Seamlessly switches between local inference (Ollama/Qwen 2.5) and cloud endpoints (Google Gemini 3.5 Flash) via LangChain.
* **Privacy-Preserving:** Allows enterprise-grade architectural analysis entirely on-premises without exposing proprietary code to third-party APIs.
* **Interactive UI:** Built with Streamlit for real-time architectural querying and design pattern mapping.

## Prerequisites
* Python 3.9+
* [Ollama](https://ollama.ai/) installed and running locally.
* A valid Google Gemini API Key.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Barbarouge/CodeBase-RAG-Analyzer.git
cd CodeBase-RAG-Analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Clone the target codebase (Flask) for testing:**
```bash
git clone https://github.com/pallets/flask.git flask
```

4. **Environment Variables:**
Create a `.env` file in the root directory and securely add your API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

5. **Provision the Local LLM:**
Ensure the Ollama server is running, then pull the local inference model:
```bash
ollama pull qwen2.5:7b
```

## Usage

1. **Launch the User Interface:**
```bash
streamlit run app.py
```

2. **Build the Vector Database:**
On the Streamlit interface, click the **Parse & Save (Build DB)** button. This will trigger the AST parser to traverse the `flask` directory and build the `chroma_db` vectors locally.

3. **Query the System:**
Select your preferred model (Gemini or Local LLM) from the dropdown and enter your architectural queries (e.g., *"What is the core method that acts as the WSGI application interface?"*).

## Evaluation
To reproduce the quantitative findings (e.g., 85.71% Hit Rate, 0.762 MRR) discussed in Chapter 4 of the dissertation, run the automated evaluation pipeline:
```bash
python evaluate_retrieval.py
```
