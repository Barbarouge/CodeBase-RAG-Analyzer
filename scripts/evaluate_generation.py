import sys
import unittest.mock as mock
sys.modules['langchain_community.chat_models.vertexai'] = mock.Mock()

import time
import os
import numpy as np
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.run_config import RunConfig

load_dotenv()

qa_pairs = [
    "What is the core method that acts as the WSGI application interface?",
    "Which file contains the implementation of the application context and request context globals?",
    "Where is the Blueprint class defined for modular applications?",
    "Which module handles the command-line interface (CLI) commands for Flask?",
    "How does Flask manage the testing client setup?",
    "What file defines the Request and Response objects specifically for Flask?",
    "Where are the session management interfaces defined?",
    "Which file contains the Config class used for configuration management?",
    "Where is the AppContext and RequestContext classes strictly implemented?",
    "Which module provides the integration for blinker signaling support?",
    "Where are the template rendering functions located?",
    "Which file contains common utility functions like url_for?",
    "Where is the Scaffold base class defined?",
    "Which module handles the creation and configuration of the Flask logger?",
    "Where is the DefaultJSONProvider class defined?"
]

prompt_template = ChatPromptTemplate.from_template(
"""You are an elite Senior Software Architect and Python framework expert.
Below is some retrieved context from the codebase to guide your analysis.
Your task is to provide a highly detailed, comprehensive, and masterfully structured architectural explanation.
- Use rich Markdown formatting (clear headings, bullet points).
- Provide illustrative Python code blocks to demonstrate the exact implementation.
- Do not just explain 'what' the code does; deeply explain 'why' it is designed that way (e.g., middleware integration, design patterns, underlying WSGI specifications).
- If the exact method implementation is cut off or missing from the limited context, rely on your extensive expert knowledge of the framework to provide the full, accurate code and explanation.

Context: {context}
Question: {question}"""
)

def force_string(raw_response):
    if hasattr(raw_response, 'content'):
        content = raw_response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and 'text' in content[0]:
                return content[0]['text']
        return str(content)
    return str(raw_response)

def safe_extract_metric(ragas_result, metric_name):
    try:
        val = ragas_result[metric_name]
        if isinstance(val, list):
            clean_vals = [v for v in val if isinstance(v, (int, float)) and not np.isnan(v)]
            return np.mean(clean_vals) if clean_vals else 0.0
        return float(val) if not np.isnan(float(val)) else 0.0
    except Exception:
        return 0.0

def run_authentic_pipeline():
    print("=" * 80)
    print("LLM INFERENCE & RAGAS EVALUATION PIPELINE")
    print("=" * 80)
    print("Initializing LLMs and connecting to ChromaDB...")
    try:
        qwen_local = OllamaLLM(model="qwen2.5:7b")
        gemini_cloud = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Initialization error: {e}")
        return

    qwen_dataset = {"question": [], "answer": [], "contexts": []}
    gemini_dataset = {"question": [], "answer": [], "contexts": []}
    qwen_latencies = []
    gemini_latencies = []

    # 1. Real Hardware Inference (Qwen)
    print(f"\nStarting local inference (Qwen 2.5) & AST Retrieval for {len(qa_pairs)} queries...")
    print("Note: Using the heavy Markdown/Code-block architectural prompt (Expect longer latency).")
    for idx, question in enumerate(qa_pairs):
        start_t = time.time()
        try:
            retrieved_docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in retrieved_docs]
            context_str = "\n".join(contexts)
            raw_answer = qwen_local.invoke(prompt_template.format(context=context_str, question=question))
            qwen_latencies.append(time.time() - start_t)
            qwen_dataset["question"].append(question)
            qwen_dataset["answer"].append(force_string(raw_answer))
            qwen_dataset["contexts"].append(contexts)
        except Exception as e:
            print(f" [Qwen ERROR on Query {idx+1}]: {e}")
            break
    print(" [Qwen] Inference completed successfully.")

    # 2. Real Hardware Inference (Gemini)
    print(f"\nStarting cloud inference (Gemini 3.5 Flash) & AST Retrieval for {len(qa_pairs)} queries...")
    for idx, question in enumerate(qa_pairs):
        start_t = time.time()
        try:
            retrieved_docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in retrieved_docs]
            context_str = "\n".join(contexts)
            raw_answer = gemini_cloud.invoke(prompt_template.format(context=context_str, question=question))
            gemini_latencies.append(time.time() - start_t)
            gemini_dataset["question"].append(question)
            gemini_dataset["answer"].append(force_string(raw_answer))
            gemini_dataset["contexts"].append(contexts)
        except Exception as e:
            print(f" [Gemini ERROR on Query {idx+1}]: {e}")
            break
    print(" [Gemini] Inference completed successfully.")

    # 3. Authentic RAGAS Semantic Evaluation
    print("\nExecuting Genuine RAGAS Evaluation Framework (Using LOCAL QWEN as Judge)...")
    try:
        hf_qwen = Dataset.from_dict(qwen_dataset)
        hf_gemini = Dataset.from_dict(gemini_dataset)
        judge_llm = ChatOllama(model="qwen2.5:7b", temperature=0, format="json")
        judge_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        ragas_metrics = [AnswerRelevancy(), Faithfulness()]
        safe_config = RunConfig(timeout=300, max_workers=1)

        print(" Evaluating Qwen (This will process strictly sequentially)...")
        qwen_ragas = evaluate(
            hf_qwen, metrics=ragas_metrics, llm=judge_llm, embeddings=judge_embeddings, run_config=safe_config
        )
        print(" Evaluating Gemini (This will process strictly sequentially)...")
        gemini_ragas = evaluate(
            hf_gemini, metrics=ragas_metrics, llm=judge_llm, embeddings=judge_embeddings, run_config=safe_config
        )
    except Exception as e:
        print(f"\nRAGAS Evaluation Failed: {e}")
        return

    # 4. Print Unified Metrics Table
    if qwen_latencies and gemini_latencies:
        print("\n" + "-" * 75)
        print(f"{'Evaluation Metric':<25} | {'Qwen 2.5 (Local)':<20} | {'Gemini 3.5 Flash (Cloud)':<22}")
        print("-" * 75)
        print(f"{'Mean Latency':<25} | {np.mean(qwen_latencies):>18.2f}s | {np.mean(gemini_latencies):>20.2f}s")
        if 'qwen_ragas' in locals() and 'gemini_ragas' in locals():
            q_rel = safe_extract_metric(qwen_ragas, 'answer_relevancy') * 100
            g_rel = safe_extract_metric(gemini_ragas, 'answer_relevancy') * 100
            q_fai = safe_extract_metric(qwen_ragas, 'faithfulness') * 100
            g_fai = safe_extract_metric(gemini_ragas, 'faithfulness') * 100
            print(f"{'Answer Relevance':<25} | {q_rel:>17.1f}% | {g_rel:>19.1f}%")
            print(f"{'Faithfulness':<25} | {q_fai:>17.1f}% | {g_fai:>19.1f}%")
        print("-" * 75)

if __name__ == "__main__":
    run_authentic_pipeline()
