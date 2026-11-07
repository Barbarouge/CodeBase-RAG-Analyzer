import time
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Ground Truth Dataset mapped to Flask Source Code Files
ground_truth_dataset = [
    {"question": "What is the core method that acts as the WSGI application interface?", "expected_file": "app.py"},
    {"question": "Which file contains the implementation of the application context and request context globals?", "expected_file": "globals.py"},
    {"question": "Where is the Blueprint class defined for modular applications?", "expected_file": "blueprints.py"},
    {"question": "Which module handles the command-line interface (CLI) commands for Flask?", "expected_file": "cli.py"},
    {"question": "How does Flask manage the testing client setup?", "expected_file": "testing.py"},
    {"question": "What file defines the Request and Response objects specifically for Flask?", "expected_file": "wrappers.py"},
    {"question": "Where are the session management interfaces defined?", "expected_file": "sessions.py"},
    {"question": "Which file contains the Config class used for configuration management?", "expected_file": "config.py"},
    {"question": "Where is the AppContext and RequestContext classes strictly implemented?", "expected_file": "ctx.py"},
    {"question": "Which module provides the integration for blinker signaling support?", "expected_file": "signals.py"},
    {"question": "Where are the template rendering functions located?", "expected_file": "templating.py"},
    {"question": "Which file contains common utility functions like url_for?", "expected_file": "helpers.py"},
    {"question": "Where is the Scaffold base class defined?", "expected_file": "scaffold.py"},
    {"question": "Which module handles the creation and configuration of the Flask logger?", "expected_file": "logging.py"},
    {"question": "Where is the DefaultJSONProvider class defined?", "expected_file": "provider.py"}
]

def calculate_mrr_and_hit_rate(retriever, dataset, k=3):
    """Executes genuine retrieval and calculates strict Hit Rate and MRR metrics."""
    hit_count = 0
    mrr_sum = 0.0
    for idx, item in enumerate(dataset):
        question = item["question"]
        expected = item["expected_file"]
        retrieved_docs = retriever.invoke(question)
        rank = 0
        for i, doc in enumerate(retrieved_docs[:k]):
            source_data = doc.metadata.get("source", doc.metadata.get("file_path", doc.page_content))
            if expected in source_data:
                rank = i + 1
                break
        if rank > 0:
            hit_count += 1
            mrr_sum += (1.0 / rank)
            print(f"[HIT]  Query {idx+1:02d}: Found '{expected}' at Rank {rank}")
        else:
            print(f"[MISS] Query {idx+1:02d}: Could not find '{expected}' in top {k} results.")
            
    hit_rate = hit_count / len(dataset)
    mrr = mrr_sum / len(dataset)
    return hit_rate, mrr

def run_authentic_retrieval_evaluation():
    print("=" * 80)
    print(" RETRIEVAL (AST vs BASELINE) EVALUATION")
    print("=" * 80)
    print("Connecting to Vector Databases...")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        ast_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        ast_retriever = ast_store.as_retriever(search_kwargs={"k": 3})
        
        baseline_path = "./chroma_db_baseline"
        baseline_exists = os.path.exists(baseline_path)
        if baseline_exists:
            baseline_store = Chroma(persist_directory=baseline_path, embedding_function=embeddings)
            baseline_retriever = baseline_store.as_retriever(search_kwargs={"k": 3})
        else:
            print(f"[!] Warning: Baseline database not found at '{baseline_path}'")
    except Exception as e:
        print(f"Database connection error: {e}")
        return

    # 1. Baseline Test
    start_time = time.time()
    if baseline_exists:
        print("\n--- Evaluating Baseline (Recursive) Retrieval ---")
        base_hit_rate, base_mrr = calculate_mrr_and_hit_rate(baseline_retriever, ground_truth_dataset)
    else:
        base_hit_rate, base_mrr = 0.0, 0.0
    base_latency = time.time() - start_time

    # 2. AST Test
    start_time = time.time()
    print("\n--- Evaluating AST-Aware (Proposed) Retrieval ---")
    ast_hit_rate, ast_mrr = calculate_mrr_and_hit_rate(ast_retriever, ground_truth_dataset)
    ast_latency = time.time() - start_time

    # 3. Results Table
    print("\n" + "-" * 75)
    print(f"{'Evaluation Metric':<25} | {'Baseline (Recursive)':<20} | {'AST-Aware (Proposed)':<22}")
    print("-" * 75)
    print(f"{'Hit Rate (@3)':<25} | {base_hit_rate*100:>19.2f}% | {ast_hit_rate*100:>21.2f}%")
    print(f"{'Mean Reciprocal Rank':<25} | {base_mrr:>19.3f}  | {ast_mrr:>21.3f}")
    print("-" * 75)
    print(f"Evaluation completed. Baseline: {base_latency:.2f}s | AST: {ast_latency:.2f}s")

if __name__ == "__main__":
    run_authentic_retrieval_evaluation()
