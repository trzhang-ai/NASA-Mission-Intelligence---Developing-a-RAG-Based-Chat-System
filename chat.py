#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Enhanced version of the simple RAG chat that includes real-time evaluation
and feedback collection for continuous improvement.
"""

import streamlit as st
import os
import math
import ragas_evaluator
import rag_client
import llm_client
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env"
)

# Use the evaluator's RAGAS compatibility check.
RAGAS_AVAILABLE = ragas_evaluator.RAGAS_AVAILABLE

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    return rag_client.discover_chroma_backends()

def initialize_rag_system(
    chroma_dir: str,
    collection_name: str,
    openai_key: str,
):
    """Initialize the RAG system with the selected backend."""
    try:
        return rag_client.initialize_rag_system(
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            openai_api_key=openai_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
        )
    except Exception as error:
        return None, False, str(error)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    try:
        return rag_client.retrieve_documents(collection, query, n_results, mission_filter)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    return rag_client.format_context(documents, metadatas)

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str,
) -> str:
    """Generate a grounded response using the configured LLM."""
    return llm_client.generate_response(
        openai_key=openai_key,
        user_message=user_message,
        context=context,
        conversation_history=conversation_history,
        model=model,
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
    )

def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    openai_key: str,
) -> Dict[str, float | str]:
    """Evaluate response quality using RAGAS metrics."""
    try:
        return ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=contexts,
            evaluator_model=os.getenv(
                "OPENAI_EVALUATOR_MODEL",
                ragas_evaluator.DEFAULT_EVALUATOR_MODEL,
            ),
            openai_api_key=openai_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
        )
    except Exception as error:
        return {
            "error": f"Evaluation failed: {error}"
        }

def display_evaluation_metrics(
    scores: Dict[str, float | str],
):
    """Display valid evaluation metrics in the sidebar."""
    if "error" in scores:
        st.sidebar.error(
            f"Evaluation Error: {scores['error']}"
        )
        return
    st.sidebar.subheader("📊 Response Quality")
    for metric_name, score in scores.items():
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            continue
        label = metric_name.replace("_", " ").title()
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            st.sidebar.warning(
                f"{label} is unavailable."
            )
            continue
        st.sidebar.metric(
            label=label,
            value=f"{numeric_score:.3f}",
            delta=None,
        )
        st.sidebar.progress(
            max(0.0, min(numeric_score, 1.0))
        )

def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        # Discover available backends
        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()
        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info(
                "Please run the embedding pipeline first:\n"
                "`uv run python embedding_pipeline.py`"
            )
            st.stop()
        # Backend selection
        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}
        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(backend_options.keys()),
            format_func=lambda x: backend_options[x],
            help="Choose which document collection to use for retrieval"
        )
        selected_backend = available_backends[selected_backend_key]
        # API Key input
        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()
        # Model selection
        default_model = os.getenv(
            "OPENAI_CHAT_MODEL",
            llm_client.DEFAULT_GENERATOR_MODEL,
        )
        model_choice = st.text_input(
            "OpenAI Model",
            value=default_model,
            help="Enter a chat model supported by your configured API endpoint",
        )
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)
        mission_choice = st.selectbox(
            "Mission filter",
            options=[
                "All missions",
                "Apollo 11",
                "Apollo 13",
                "Challenger",
            ],
        )
        # Evaluation settings
        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=RAGAS_AVAILABLE)
        # Initialize RAG system when backend changes
        if (st.session_state.current_backend != selected_backend_key):
            st.session_state.current_backend = selected_backend_key
            # Clear cache to force reinitialization
            st.cache_resource.clear()
    # Initialize RAG system
    with st.spinner("Initializing RAG system..."):
        collection, success, error = initialize_rag_system(
            chroma_dir=selected_backend["directory"],
            collection_name=selected_backend["collection_name"],
            openai_key=openai_key,
        )
    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()
    # Display evaluation metrics if available
    if st.session_state.last_evaluation and enable_evaluation:
        display_evaluation_metrics(st.session_state.last_evaluation)
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Chat input
    if prompt := st.chat_input("Ask about NASA space missions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                # Retrieve relevant documents
                docs_result = retrieve_documents(
                    collection=collection,
                    query=prompt,
                    n_results=n_docs,
                    mission_filter=mission_choice,
                )
                # Format context
                context = ""
                contexts_list = []
                if docs_result and docs_result.get("documents"):
                    context = format_context(docs_result["documents"][0], docs_result["metadatas"][0])
                    contexts_list = docs_result["documents"][0]
                    st.session_state.last_contexts = contexts_list
                # Generate response
                try:
                    response = generate_response(
                        openai_key,
                        prompt,
                        context,
                        st.session_state.messages[:-1],
                        model_choice,
                    )
                except Exception as error:
                    st.error(f"Error generating response: {error}")
                    st.session_state.messages.pop()
                    st.stop()
                st.markdown(response)
                # Evaluate response quality if enabled
                if enable_evaluation and RAGAS_AVAILABLE:
                    with st.spinner("Evaluating response quality..."):
                        evaluation_scores = evaluate_response_quality(
                            prompt, 
                            response, 
                            contexts_list,
                            openai_key
                        )
                        st.session_state.last_evaluation = evaluation_scores
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if __name__ == "__main__":
    main()
