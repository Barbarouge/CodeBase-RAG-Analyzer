import streamlit as st
import os
import time
import ast
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Environment Variables 
load_dotenv()

# 2. Streamlit Page Configuration
st.set_page_config(
    page_title="RAG & LLM Dissertation Prototype", 
    page_icon="🤖", 
    layout="centered"
)

TARGET_DIR = "./flask/src/flask"
VECTOR_DB_PATH = "./chroma_db"

# 3. Cache the Vector Database (Performance Optimization)
@st.cache_resource
def load_retriever():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(VECTOR_DB_PATH):
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    return None

retriever = load_retriever()

# --- AST PARSE & SAVE FUNCTION ---
def rebuild_vector_db():
    st.session_state.is_building = True
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("1/3: Reading and Parsing AST...")
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    
    for root, _, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()
                try:
                    tree = ast.parse(source)
                    for node in tree.body:
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            chunk_code = ast.get_source_segment(source, node)
                            if chunk_code:
                                if len(chunk_code) > 4000:
                                    sub_chunks = text_splitter.split_text(chunk_code)
                                    for i, sub_chunk in enumerate(sub_chunks):
                                        documents.append(Document(
                                            page_content=sub_chunk,
                                            metadata={"source": file, "type": type(node).__name__, "name": f"{node.name}_part{i+1}"}
                                        ))
                                else:
                                    documents.append(Document(
                                        page_content=chunk_code,
                                        metadata={"source": file, "type": type(node).__name__, "name": node.name}
                                    ))
                except Exception as e:
                    pass
                    
    progress_bar.progress(50)
    status_text.text(f"2/3: Extracted {len(documents)} AST blocks. Vectorizing now...")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=VECTOR_DB_PATH)
    
    progress_bar.progress(100)
    status_text.text("3/3: Success! Vector DB saved to disk.")
    time.sleep(1.5)
    st.cache_resource.clear()
    st.rerun()

# --- MAIN UI ---
st.title("RAG and LLM Integration Interface")
st.markdown("The API key is securely loaded from the environment. You can enter your query directly below.")

# Model Configuration
st.subheader("Model Configuration")
selected_model = st.selectbox(
    "Select the model you want to use:",
    ("Gemini 3.5 Flash", "Local LLM")
)
st.divider()

# Database Management
with st.expander("⚙️ Knowledge Base Management (Optional)"):
    st.markdown("If you haven't built the vector database yet, or if the source code has changed, build it here.")
    if st.button("🔄 Parse & Save (Build DB)"):
        rebuild_vector_db()
    if not retriever:
        st.warning("No Vector DB found. Please click 'Parse & Save' to initialize the RAG system.")

# RAG Logic Prompts
def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source')} | Code:\n{doc.page_content}" for doc in docs)

template = """You are an elite Senior Software Architect and Python framework expert.
Below is some retrieved context from the codebase to guide your analysis. 

Your task is to provide a highly detailed, comprehensive, and masterfully structured architectural explanation. 
- Use rich Markdown formatting (clear headings, bullet points).
- Provide illustrative Python code blocks to demonstrate the exact implementation.
- Do not just explain 'what' the code does; deeply explain 'why' it is designed that way (e.g., middleware integration, design patterns, underlying WSGI specifications).
- If the exact method implementation is cut off or missing from the limited context, rely on your extensive expert knowledge of the framework to provide the full, accurate code and explanation.

Context: {context}

Question: {question}
"""
prompt_template = ChatPromptTemplate.from_template(template)

# User Input Area
user_input = st.text_area("Enter your text or query:", height=150)

# Submit Button and Execution
if st.button("Submit"):
    if not retriever:
        st.warning("Please build the Vector DB first using the 'Knowledge Base Management' expander above.")
    elif user_input.strip() == "":
        st.warning("Please enter a valid prompt.")
    else:
        with st.spinner(f"{selected_model} is generating a response..."):
            start_time = time.time()
            
            # Select LLM
            if selected_model == "Local LLM":
                llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)
            elif selected_model == "Gemini 3.5 Flash":
                if not os.getenv("GEMINI_API_KEY"):
                    st.error("GEMINI_API_KEY is missing in your .env file!")
                    st.stop()
                # MODEL ADI BURADA DÜZELTİLDİ:
                llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
            
            # LCEL Chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            try:
                retrieved_docs = retriever.invoke(user_input)
                response = rag_chain.invoke(user_input)
                inference_time = time.time() - start_time
                
                # Display the Response
                st.success("Operation Successful!")
                
                if selected_model == "Local LLM":
                    st.markdown("### Local Model Response:")
                else:
                    st.markdown("### Gemini Response:")
                    
                st.write(response)
                
                # Metrics (Süre ve Bulunan Dosyalar)
                st.divider()
                st.markdown(f"**⏱️ Inference Time:** {inference_time:.2f} seconds")
                st.markdown("**📂 Retrieved Source Documents (AST Blocks):**")
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"- **Match {i+1}:** `{doc.metadata.get('source')}` (Type: {doc.metadata.get('type')}, Name: {doc.metadata.get('name')})")
                    
            except Exception as e:
                st.error(f"An error occurred during inference: {str(e)}")