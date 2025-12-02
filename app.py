# type: ignore # type: ignore
"""
Streamlit Research Agent Web Application

This app provides a web interface for:
1. Research Pipeline - Topic research with clarification, SERP generation, and web search
2. Learning Extraction - Extract structured learnings from search results
"""

import streamlit as st
import asyncio
import os
import json
import time
import tempfile
import pandas as pd
import subprocess
import platform
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables at the module level
load_dotenv()

# Import existing modules
from agents.clarification import get_clarifications
from agents.serp import get_serp_queries
from agents.learn import get_learning_structured
from agents.summarise_csv import summarize_csv_file, save_summarized_csv
from config.searxng_tools import searxng_web_tool, searxng_client
from config.model_config import get_model
from schemas.datamodel import (
    SearchResultsCollection,
    CSVSummarizationMetadata,
    CSVSummarizationHistory,
)
from embeddings import CSVEmbeddingProcessor
from embeddings_processor import JSONEmbeddingProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research.log'),
        logging.StreamHandler()
    ]
)

# Set page configuration for wide layout
st.set_page_config(
    page_title="TI Agent",
    page_icon="🔬",
    layout="wide",  # Use full width of the page
    initial_sidebar_state="expanded"
)

# Custom CSS to change red colors to royal blue
st.markdown("""
    <style>
    /* Change primary button color from red to royal blue */
    .stButton > button[kind="primary"] {
        background-color: #4169E1 !important;
        border-color: #4169E1 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #1E40AF !important;
        border-color: #1E40AF !important;
    }
    
    /* Change error messages from red to royal blue */
    .stAlert[data-baseweb="notification"][kind="error"] {
        background-color: rgba(65, 105, 225, 0.1) !important;
        border-left-color: #4169E1 !important;
    }
    
    /* Change warning colors to royal blue tones */
    .stAlert[data-baseweb="notification"][kind="warning"] {
        background-color: rgba(65, 105, 225, 0.1) !important;
        border-left-color: #4169E1 !important;
    }
    
    /* Change progress bar color to royal blue */
    .stProgress > div > div > div > div {
        background-color: #4169E1 !important;
    }
    
    /* Change download button hover to royal blue */
    .stDownloadButton > button:hover {
        border-color: #4169E1 !important;
        color: #4169E1 !important;
    }
    
    /* Change radio button selected state to royal blue */
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: #4169E1 !important;
    }
    
    /* Change checkbox selected state to royal blue */
    .stCheckbox > label > div[data-baseweb="checkbox"] > div {
        border-color: #4169E1 !important;
        background-color: #4169E1 !important;
    }
    
    /* Change slider to royal blue */
    .stSlider > div > div > div > div {
        background-color: #4169E1 !important;
    }
    
    /* Change number input focus border to royal blue */
    .stNumberInput > div > div > input:focus {
        border-color: #4169E1 !important;
        box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
    }
    
    /* Change text input focus border to royal blue */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4169E1 !important;
        box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
    }
    
    /* Change selectbox focus to royal blue */
    .stSelectbox > div > div > div:focus-within {
        border-color: #4169E1 !important;
        box-shadow: 0 0 0 0.2rem rgba(65, 105, 225, 0.25) !important;
    }
    
    /* Change metric delta positive color to royal blue */
    [data-testid="stMetricDelta"] svg {
        fill: #4169E1 !important;
    }
    
    /* Change links to royal blue */
    a {
        color: #4169E1 !important;
    }
    
    a:hover {
        color: #1E40AF !important;
    }
    
    /* Change spinner to royal blue */
    .stSpinner > div {
        border-top-color: #4169E1 !important;
    }
    
    /* Make sidebar divider line black */
    section[data-testid="stSidebar"] > div {
        border-right: 2px solid #000000 !important;
    }
    
    /* Alternative selector for sidebar border */
    .css-1d391kg, .st-emotion-cache-1d391kg {
        border-right: 2px solid #000000 !important;
    }
    
    /* Sidebar width settings - only when expanded */
    section[data-testid="stSidebar"]:not([aria-hidden="true"]) {
        min-width: 350px !important;
        width: 450px !important;
    }
    
    section[data-testid="stSidebar"]:not([aria-hidden="true"]) > div {
        min-width: 350px !important;
    }
    
    /* Ensure sidebar collapse/expand works properly */
    section[data-testid="stSidebar"][aria-hidden="true"] {
        min-width: 0 !important;
        width: 0 !important;
    }
    
    /* Auto-adjust multiselect widget to use full sidebar width */
    section[data-testid="stSidebar"] .stMultiSelect {
        width: 100% !important;
    }
    
    /* Standardize all multiselect items to same width - fit content */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
        width: 100% !important;
    }
    
    /* Make all tags the same size to match longest item */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        white-space: nowrap !important;
        max-width: none !important;
        font-size: 0.8rem !important;
        min-width: fit-content !important;
        width: auto !important;
        display: inline-flex !important;
    }
    
    /* Ensure tag container expands to fit all tags */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {
        width: 100% !important;
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    
    /* Reduce font size for multiselect items */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 0.75rem !important;
    }
    
    /* Reduce font size in dropdown options */
    section[data-testid="stSidebar"] .stMultiSelect [role="option"] {
        font-size: 0.75rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to format time
def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# Custom logging handler to capture logs for Streamlit display
class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that stores logs in session state."""
    
    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        try:
            msg = self.format(record)
            if 'crawl_logs' not in st.session_state:
                st.session_state.crawl_logs = []
            st.session_state.crawl_logs.append(msg)
            # Keep only last 200 log entries to avoid memory issues
            if len(st.session_state.crawl_logs) > 200:
                st.session_state.crawl_logs = st.session_state.crawl_logs[-200:]
        except Exception:
            self.handleError(record)


# Initialize session state
if 'crawl_logs' not in st.session_state:
    st.session_state.crawl_logs = []
if 'csv_processing' not in st.session_state:
    st.session_state.csv_processing = False
if 'csv_processed_df' not in st.session_state:
    st.session_state.csv_processed_df = None
if 'csv_metadata' not in st.session_state:
    st.session_state.csv_metadata = None
if 'csv_progress' not in st.session_state:
    st.session_state.csv_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
st.session_state.crawl_results = None
st.session_state.crawling_in_progress = False
st.session_state.crawl_cancel_requested = False
st.session_state.crawl_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0, 'start_time': None}
if 'strategy_confirmed' not in st.session_state:
    st.session_state.strategy_confirmed = False
if 'clarifications' not in st.session_state:
    st.session_state.clarifications = None
if 'serp_queries' not in st.session_state:
    st.session_state.serp_queries = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = None
if 'learnings' not in st.session_state:
    st.session_state.learnings = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'processing_in_progress' not in st.session_state:
    st.session_state.processing_in_progress = False
if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
if 'rag_store' not in st.session_state:
    st.session_state.rag_store = None
if 'rag_chat' not in st.session_state:
    st.session_state.rag_chat = []
if 'selected_model_config' not in st.session_state:
    st.session_state.selected_model_config = {'provider': 'azure', 'model_name': None}


def reset_session_state():
    """Reset all session state variables to default values"""
    st.session_state.crawl_logs = []
    st.session_state.csv_processing = False
    st.session_state.csv_processed_df = None
    st.session_state.csv_metadata = None
    st.session_state.csv_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
    st.session_state.crawl_results = None
    st.session_state.crawling_in_progress = False
    st.session_state.crawl_cancel_requested = False
    st.session_state.crawl_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0, 'start_time': None}
    st.session_state.strategy_confirmed = False
    st.session_state.clarifications = None
    st.session_state.serp_queries = None
    st.session_state.search_results = None
    st.session_state.current_stage = None
    st.session_state.learnings = None
    st.session_state.detection_results = None
    st.session_state.processing_in_progress = False
    st.session_state.processing_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
    st.session_state.rag_store = None
    st.session_state.rag_chat = []
    st.session_state.selected_model_config = {'provider': 'azure', 'model_name': None}


def standardize_url(url: str) -> str:
    """Ensure URL has a scheme (https://"""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


# Helper function to run async code in Streamlit
def run_async(coro):
    """Run async coroutine in Streamlit-compatible way"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
        return result
    finally:
        loop.close()


def run_clarification_stage(topic: str):
    """Stage 1: Get clarifications (Streamlit version - no input())"""
    async def _async_clarification():
        from agents.clarification import create_clarification_agent
        from schemas.datamodel import ClarificationResponse, ClarificationQuestion

        clarification_agent = create_clarification_agent()
        response = ClarificationResponse(original_query=topic)

        try:
            # Get clarification questions from agent (no user input)
            result = await clarification_agent.run(topic)
            questions = result.output

            # Parse questions and add them WITHOUT answers
            if questions:
                for i, question in enumerate(questions.split('\n'), 1):
                    if question.strip():
                        qa = ClarificationQuestion(
                            question=question.strip(),
                            answer=""  # Empty - will be filled by Streamlit form
                        )
                        response.questions_and_answers.append(qa)
        except Exception as e:
            logging.error(f"Error getting clarifications: {e}")

        logging.info(f"Clarifications received: {len(response.questions_and_answers)} questions")
        return response

    with st.spinner("Generating clarification questions..."):
        clarifications = run_async(_async_clarification())
        st.session_state.clarifications = clarifications
    return clarifications


def run_serp_generation_stage(topic: str, clarifications):
    """Stage 2: Generate SERP queries"""
    async def _async_serp():
        serp_queries = await get_serp_queries(topic, clarifications)
        logging.info(f"Generated {len(serp_queries)} SERP queries")
        return serp_queries

    with st.spinner("Generating SERP queries..."):
        serp_queries = run_async(_async_serp())
        st.session_state.serp_queries = serp_queries
    return serp_queries


def run_search_stage(serp_queries):
    """Stage 3: Execute web searches"""
    async def _async_search():
        results_collection = SearchResultsCollection()
        total_queries = len(serp_queries)

        for idx, query in enumerate(serp_queries, 1):
            status_text.text(f"Searching [{idx}/{total_queries}]: {query}")
            logging.info(f"Searching [{idx}/{total_queries}]: {query}")

            try:
                results = await searxng_client._search(query)
                results_collection.add_result(query, results)
                logging.info(f"Search successful: {query}")
            except Exception as e:
                st.warning(f"Search failed for '{query}': {e}")
                logging.error(f"Search failed for '{query}': {e}")
                continue

            progress_bar.progress(idx / total_queries)

        return results_collection

    progress_bar = st.progress(0)
    status_text = st.empty()

    results_collection = run_async(_async_search())

    status_text.text(f"Search complete! {results_collection.total_queries} queries executed.")
    st.session_state.search_results = results_collection

    return results_collection


def run_learning_extraction_stage(results_collection):
    """Stage 4: Extract learnings"""
    async def _async_learning():
        learnings_dict = {}
        results_list = list(results_collection.results.items())
        total_queries = len(results_list)

        for idx, (query, search_result) in enumerate(results_list, 1):
            status_text.text(f"Extracting learnings [{idx}/{total_queries}]: {query[:50]}...")
            logging.info(f"Extracting learnings [{idx}/{total_queries}]: {query}")

            try:
                learnings = await get_learning_structured(query, search_result.results)
                learnings_dict[query] = learnings
                logging.info(f"Learning extraction successful: {query}")
            except Exception as e:
                st.warning(f"Learning extraction failed for '{query}': {e}")
                logging.error(f"Learning extraction failed for '{query}': {e}")
                continue

            progress_bar.progress(idx / total_queries)

        return learnings_dict

    progress_bar = st.progress(0)
    status_text = st.empty()

    learnings_dict = run_async(_async_learning())

    status_text.text(f"Learning extraction complete! {len(learnings_dict)} learnings extracted.")
    st.session_state.learnings = learnings_dict

    return learnings_dict


def save_search_results(results_collection, filename):
    """Save search results to JSON file"""
    path = Path("data")
    path.mkdir(parents=True, exist_ok=True)

    output_path = path / f"{filename}.json"
    results_collection.to_file(output_path)
    
    # Upload to S3 if configured
    try:
        from aws_storage import get_storage
        s3_storage = get_storage()
        s3_key = f"research_results/{filename}.json"
        s3_storage.upload_file(str(output_path), s3_key)
        logging.info(f"✓ Uploaded search results to S3: {s3_key}")
    except Exception as e:
        logging.warning(f"⚠️ S3 upload skipped: {e}")

    return output_path


def save_learnings(learnings_dict, filename):
    """Save learnings to markdown file"""
    path = Path("data")
    path.mkdir(parents=True, exist_ok=True)

    output_path = path / f"{filename}.md"

    with open(output_path, "w", encoding="utf-8") as f:
        for query, learnings in learnings_dict.items():
            f.write(f"## {query}\n\n")
            f.write(learnings)
            f.write("\n\n---\n\n")
    
    # Upload to S3 if configured
    try:
        from aws_storage import get_storage
        s3_storage = get_storage()
        s3_key = f"research_results/{filename}.md"
        s3_storage.upload_file(str(output_path), s3_key)
        logging.info(f"✓ Uploaded learnings to S3: {s3_key}")
    except Exception as e:
        logging.warning(f"⚠️ S3 upload skipped: {e}")

    return output_path


def load_search_results(file_path):
    """Load search results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    collection = SearchResultsCollection()
    for query, results in data.items():
        collection.add_result(query, results)

    return collection


# Main App
st.title("Technology Intelligence Tool")
st.markdown("AI-powered tool equipped with discovery of sources, web-crawling, LLM extraction into structured data for a combined database and chatbot")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Mode",
    ["Web Search", "Web Crawler", "LLM Extraction", "Summarization", "Database", "Chatbot", "About", "LinkedIn Home Feed Monitor"],
    label_visibility="collapsed"
)

# Web Search Page
if page == "Web Search":
    st.header("Web Search")
    st.markdown("AI-powered research with clarification, SERP generation, web search, and learning extraction")
    
    # Create tabs for Research Pipeline and Learning Extraction
    tab1, tab2 = st.tabs(["Research Pipeline", "Learning Extraction"])
    
    with tab1:
        st.subheader("Research Pipeline")
        st.markdown("Conduct comprehensive research with automated clarification and web search")

        # Input Section
        st.subheader("1. Research Topic")
        topic = st.text_area(
            "Enter your research topic:",
            placeholder="e.g., What is the expected growth rate for Singapore companies that sell AI solutions?",
            height=100
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            start_research = st.button("Start Research", type="primary", use_container_width=True)
        with col2:
            reset_btn = st.button("Reset", use_container_width=True)

        if reset_btn:
            reset_session_state()
            st.rerun()

        if start_research and topic:
            st.session_state.current_stage = 'clarification'

        # Clarification Stage
        if st.session_state.current_stage == 'clarification' and topic:
            st.subheader("2. Clarification Questions")

            if st.session_state.clarifications is None:
                # Generate clarifications
                clarifications = run_clarification_stage(topic)

                if clarifications.questions_and_answers:
                    st.success(f"Generated {len(clarifications.questions_and_answers)} clarification questions")
                    st.session_state.current_stage = 'answer_clarifications'
                    st.rerun()
                else:
                    st.info("No clarification questions needed. Proceeding to search...")
                    st.session_state.current_stage = 'search'
                    st.rerun()

        # Answer Clarifications
        if st.session_state.current_stage == 'answer_clarifications':
            st.subheader("2. Answer Clarification Questions")

            clarifications = st.session_state.clarifications

            with st.form("clarification_form"):
                st.markdown("Please answer the following questions to refine the research:")

                answers = []
                for idx, qa in enumerate(clarifications.questions_and_answers):
                    st.markdown(f"**Question {idx + 1}:** {qa.question}")
                    answer = st.text_input(
                        f"Your answer:",
                        key=f"answer_{idx}",
                        label_visibility="collapsed"
                    )
                    answers.append(answer)

                submitted = st.form_submit_button("Submit Answers", type="primary")

                if submitted:
                    # Update clarifications with answers
                    for idx, answer in enumerate(answers):
                        clarifications.questions_and_answers[idx].answer = answer

                    st.session_state.clarifications = clarifications
                    st.session_state.current_stage = 'search'
                    st.rerun()

        # Search Stage
        if st.session_state.current_stage == 'search' and topic:
            st.subheader("3. Web Search")

            if st.session_state.serp_queries is None:
                # Generate SERP queries
                serp_queries = run_serp_generation_stage(topic, st.session_state.clarifications)

                if serp_queries:
                    st.success(f"Generated {len(serp_queries)} SERP queries")

                    with st.expander("View SERP Queries"):
                        for idx, query in enumerate(serp_queries, 1):
                            st.text(f"{idx}. {query}")
                else:
                    st.error("No SERP queries generated. Please try again.")
                    st.stop()

            if st.session_state.search_results is None:
                if st.button("Execute Web Search", type="primary"):
                    results_collection = run_search_stage(st.session_state.serp_queries)

                    if results_collection.results:
                        st.success(f"Search complete! Collected {results_collection.total_queries} results.")
                        st.session_state.current_stage = 'save_results'
                        st.rerun()
                    else:
                        st.error("No search results collected.")
            else:
                st.success(f"Search complete! Collected {st.session_state.search_results.total_queries} results.")
                st.session_state.current_stage = 'save_results'

        # Save Results Stage
        if st.session_state.current_stage == 'save_results':
            st.subheader("4. Save Results")

            results_collection = st.session_state.search_results

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Queries Executed", results_collection.total_queries)
            with col2:
                st.metric("Timestamp", results_collection.timestamp.strftime("%Y-%m-%d %H:%M:%S"))

            # Show ALL search results
            st.subheader("Search Results")

            # Add a search/filter box
            search_filter = st.text_input("Filter queries:", placeholder="Type to filter results...")

            # Filter results if search term provided
            filtered_results = results_collection.results.items()
            if search_filter:
                filtered_results = [(q, r) for q, r in results_collection.results.items()
                                   if search_filter.lower() in q.lower()]

            # Display results count
            st.info(f"Showing {len(filtered_results) if search_filter else len(results_collection.results)} results")

            # Display all results in expandable sections
            for idx, (query, result) in enumerate(filtered_results if search_filter else results_collection.results.items(), 1):
                with st.expander(f"**{idx}. {query}**", expanded=False):
                    # Show snippet and full content
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.caption(f"Length: {len(result.results)} chars")
                        st.caption(f"Timestamp: {result.timestamp.strftime('%H:%M:%S')}")

                    # Display the full results
                    st.text_area(
                        "Search Results:",
                        value=result.results,
                        height=300,
                        key=f"search_result_{idx}",
                        label_visibility="collapsed"
                    )

            filename = st.text_input(
                "Enter filename to save results (without extension):",
                value=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            if st.button("Save Results", type="primary"):
                try:
                    output_path = save_search_results(results_collection, filename)
                    st.success(f"Results saved to {output_path}")

                    # Provide download button
                    with open(output_path, 'r', encoding='utf-8') as f:
                        json_data = f.read()

                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{filename}.json",
                        mime="application/json"
                    )

                    st.info("You can now extract learnings from this file in the 'Learning Extraction' tab (above)")

                except Exception as e:
                    st.error(f"Error saving results: {e}")

    with tab2:
        st.subheader("Learning Extraction")
        st.markdown("Extract structured learnings from search results")

        # Option 1: Upload JSON file
        st.subheader("Option 1: Upload Search Results")
        uploaded_file = st.file_uploader(
            "Upload JSON file with search results",
            type=['json'],
            help="Upload a JSON file created from the Research Pipeline"
        )

        # Option 2: Select from existing files
        st.subheader("Option 2: Select Existing File")
        data_path = Path("data")
        if data_path.exists():
            json_files = list(data_path.glob("*.json"))
            if json_files:
                selected_file = st.selectbox(
                    "Select a file:",
                    options=[f.name for f in json_files],
                    index=None
                )
            else:
                st.info("No JSON files found in data/ directory")
                selected_file = None
        else:
            st.info("No data/ directory found")
            selected_file = None

        # Process selected file
        results_dict = None

        if uploaded_file is not None:
            try:
                results_dict = json.load(uploaded_file)
                st.success(f"Loaded {len(results_dict)} queries from uploaded file")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        elif selected_file is not None:
            try:
                file_path = data_path / selected_file
                with open(file_path, 'r', encoding='utf-8') as f:
                    results_dict = json.load(f)
                st.success(f"Loaded {len(results_dict)} queries from {selected_file}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        # Extract learnings
        if results_dict is not None:
            st.subheader("Extract Learnings")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", len(results_dict))

            # Preview queries
            with st.expander("Preview Queries"):
                for idx, query in enumerate(list(results_dict.keys())[:5], 1):
                    st.text(f"{idx}. {query}")
                if len(results_dict) > 5:
                    st.text(f"... and {len(results_dict) - 5} more")

            output_filename = st.text_input(
                "Output filename (without extension):",
                value=f"learnings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            if st.button("Extract Learnings", type="primary"):
                # Create SearchResultsCollection from dict
                collection = SearchResultsCollection()
                for query, results in results_dict.items():
                    collection.add_result(query, results)

                # Extract learnings
                learnings_dict = run_learning_extraction_stage(collection)

                if learnings_dict:
                    st.success(f"Extracted learnings for {len(learnings_dict)} queries")

                    # Display ALL learnings
                    st.subheader("Extracted Learnings")

                    # Add a search/filter box
                    learning_filter = st.text_input(
                        "Filter learnings:",
                        placeholder="Type to filter by query...",
                        key="learning_filter"
                    )

                    # Filter learnings if search term provided
                    filtered_learnings = learnings_dict.items()
                    if learning_filter:
                        filtered_learnings = [(q, l) for q, l in learnings_dict.items()
                                              if learning_filter.lower() in q.lower()]

                    # Display count
                    st.info(f"Showing {len(filtered_learnings) if learning_filter else len(learnings_dict)} learnings")

                    # Display all learnings in expandable sections
                    for idx, (query, learnings) in enumerate(filtered_learnings if learning_filter else learnings_dict.items(), 1):
                        with st.expander(f"**{idx}. {query}**", expanded=False):
                            st.markdown(learnings)
                            st.divider()

                            # Option to copy individual learning
                            st.code(learnings, language=None)

                    # Save learnings
                    try:
                        output_path = save_learnings(learnings_dict, output_filename)
                        st.success(f"Learnings saved to {output_path}")

                        # Provide download button
                        with open(output_path, 'r', encoding='utf-8') as f:
                            md_data = f.read()

                        st.download_button(
                            label="Download Markdown Report",
                            data=md_data,
                            file_name=f"{output_filename}.md",
                            mime="text/markdown"
                        )

                    except Exception as e:
                        st.error(f"Error saving learnings: {e}")
                else:
                    st.warning("No learnings extracted")

# Web Crawler Page
elif page == "Web Crawler":
    st.header("🕷️ Web Crawler & URL Filtering")
    st.markdown("Crawl websites and filter URLs in one streamlined workflow")
    
    # Create tabs for Crawling and Filtering
    tab1, tab2 = st.tabs(["Crawl Websites", "Filter URLs"])
    
    with tab1:
        st.subheader("Website Crawling")
        st.markdown("Crawl websites with intelligent content extraction and robots.txt compliance")
        
        # Import webcrawler components
        import sys
        from pathlib import Path
        
        # Add parent directory to path so webcrawler can be imported as a package
        workspace_path = str(Path(__file__).parent)
        if workspace_path not in sys.path:
            sys.path.insert(0, workspace_path)
        
        from webcrawler.scraper import WebScraper
        
        # Crawl mode selection
        st.subheader("Crawl Mode")
        crawl_mode = st.radio(
            "Choose crawling mode",
            ["Single URL", "Multiple URLs", "Full Site Crawl"],
            help="Single URL crawls one page, Multiple URLs crawls a list, Full Site discovers and crawls all linked pages"
        )
        
        # URL input based on mode
        if crawl_mode == "Single URL":
            url_input = st.text_input(
                "Website URL",
                placeholder="https://www.example.com",
                help="Enter the website URL you want to crawl"
            )
            urls_to_crawl = [url_input] if url_input.strip() else []
        elif crawl_mode == "Multiple URLs":
            url_input = st.text_area(
                "Website URLs (one per line)",
                placeholder="https://www.example.com\nhttps://www.another-site.org\n...",
                help="Enter multiple URLs, one per line",
                height=150
            )
            urls_to_crawl = [url.strip() for url in url_input.split('\n') if url.strip()]
        else:  # Full Site Crawl
            url_input = st.text_input(
                "Starting URL",
                placeholder="https://www.example.com",
                help="The crawler will start here and discover all linked pages"
            )
            urls_to_crawl = [url_input] if url_input.strip() else []
        
        # Auto-correct URLs: add https:// if missing
        if urls_to_crawl:
            corrected_urls = []
            for url in urls_to_crawl:
                if not url.startswith(('http://', 'https://')):
                    corrected_urls.append('https://' + url)
                else:
                    corrected_urls.append(url)
            
            if corrected_urls != urls_to_crawl:
                num_corrected = len([u for u in urls_to_crawl if not u.startswith(('http://', 'https://'))])
                st.info(f"🔗 Auto-corrected {num_corrected} URL(s) to include https://")
            urls_to_crawl = corrected_urls
        
            # Crawl configuration
            st.subheader("Crawl Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                user_agent = st.text_input(
                    "User Agent",
                    value="TI-Tool-Crawler/1.0",
                    help="Identifier for your crawler"
                )
            
            with col2:
                default_delay = st.number_input(
                    "Crawl Delay (seconds)",
                    min_value=0.5,
                    max_value=10.0,
                    value=1.0,
                    step=0.5,
                    help="Delay between requests (respects robots.txt if higher)"
                )
            
            # Full site crawl specific settings
            if crawl_mode == "Full Site Crawl":
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    max_pages = st.number_input(
                        "Max Pages",
                        min_value=1,
                        max_value=10000,
                        value=100,
                        help="Maximum number of pages to crawl"
                    )
                
                with col4:
                    max_depth = st.number_input(
                        "Max Depth",
                        min_value=1,
                        max_value=10,
                        value=3,
                        help="Maximum depth to crawl (0 = start page only)"
                    )
                
                with col5:
                    same_domain_only = st.checkbox(
                        "Same Domain Only",
                        value=True,
                        help="Only crawl pages on the same domain"
                    )
            
            clear_tracking = st.checkbox(
                "Clear Previous Tracking",
                value=False,
                help="Forget previously crawled URLs and start fresh"
            )
            
            # Display URL count
            if urls_to_crawl:
                st.info(f"📊 {len(urls_to_crawl)} URL(s) ready to crawl")
            
            # Start crawl button
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🕷️ Start Crawling", type="primary", use_container_width=True, disabled=len(urls_to_crawl) == 0):
                    # Clear previous logs
                    st.session_state.crawl_logs = []
                    
                    st.session_state.crawling_in_progress = True
                    st.session_state.crawl_cancel_requested = False
                    st.session_state.crawl_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0, 'start_time': time.time()}
                    
                    # Set up logging to capture webcrawler logs
                    streamlit_handler = StreamlitLogHandler()
                    streamlit_handler.setLevel(logging.INFO)
                    
                    # Add handler to webcrawler loggers
                    webcrawler_logger = logging.getLogger('webcrawler')
                    webcrawler_logger.addHandler(streamlit_handler)
                    webcrawler_logger.setLevel(logging.INFO)
                    
                    # Create progress containers
                    progress_container = st.container()
                    status_container = st.container()
                    
                    with progress_container:
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        time_text = st.empty()
                    
                    with status_container:
                        status_text = st.empty()
                    
                    # Start crawling process
                    try:
                        # Initialize scraper
                        scraper = WebScraper(
                            user_agent=user_agent,
                            default_delay=default_delay,
                            clear_tracking=clear_tracking
                        )
                        
                        status_text.text(f"🔧 Initializing Web Scraper (User Agent: {user_agent})")
                        
                        # Perform crawl based on mode
                        start_time = time.time()
                        
                        if crawl_mode == "Full Site Crawl":
                            status_text.text(f"🕷️ Starting full site crawl from {urls_to_crawl[0]}...")
                            
                            # Define a cancellation checker
                            def should_stop():
                                return st.session_state.crawl_cancel_requested
                            
                            summary = scraper.crawl_site(
                                start_url=urls_to_crawl[0],
                                max_pages=max_pages,
                                max_depth=max_depth,
                                same_domain_only=same_domain_only,
                                skip_if_crawled=not clear_tracking,
                                should_stop=should_stop
                            )
                            
                            progress_bar.progress(1.0)
                            progress_text.text(f"✅ Full site crawl complete!")
                        else:
                            # Single or multiple URLs
                            total_urls = len(urls_to_crawl)
                            status_text.text(f"🕷️ Crawling {total_urls} URL(s)...")
                            
                            summary = scraper.crawl_urls(
                                urls=urls_to_crawl,
                                extract_links=False  # Don't extract links for simple crawl
                            )
                            
                            progress_bar.progress(1.0)
                            progress_text.text(f"✅ Crawling complete!")
                        
                        # Store results
                        duration = time.time() - start_time
                        st.session_state.crawl_results = {
                            'summary': summary,
                            'stats': scraper.get_stats(),
                            'mode': crawl_mode,
                            'duration': duration
                        }
                        
                        # Close scraper
                        scraper.close()
                        
                        # Remove the handler to avoid duplicate logging
                        webcrawler_logger.removeHandler(streamlit_handler)
                        
                        # Update progress display
                        time_text.text(f"⏱️ Total time: {format_time(duration)}")
                        status_text.text(f"✅ Crawling completed successfully!")
                        
                        st.session_state.crawling_in_progress = False
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error during crawling: {str(e)}")
                        logging.error(f"Web Crawler error: {e}", exc_info=True)
                        # Remove the handler
                        webcrawler_logger.removeHandler(streamlit_handler)
                        st.session_state.crawling_in_progress = False
            
            with col2:
                stop_clicked = st.button("🛑 Stop", type="secondary", use_container_width=True, disabled=not st.session_state.crawling_in_progress)
                if stop_clicked:
                    st.session_state.crawl_cancel_requested = True
                    st.warning("Cancelling crawl...")
        
        # Results section
        st.divider()
        st.subheader("📊 Crawl Results")
        
        if st.session_state.crawl_results:
            results = st.session_state.crawl_results
            summary = results['summary']
            stats = results['stats']
            mode = results['mode']
            duration = results.get('duration', 0)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total URLs", summary['total'])
            with col2:
                st.metric("Successful", summary['successful'])
            with col3:
                st.metric("Failed", summary['failed'])
            with col4:
                success_rate = (summary['successful'] / summary['total'] * 100) if summary['total'] > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Duration
            st.metric("Duration", f"{duration:.2f}s")
            
            # All-time stats
            st.markdown("### 📈 All-Time Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Crawled", stats['total_crawled'])
            with col2:
                st.metric("All Successful", stats['successful'])
            with col3:
                st.metric("All Failed", stats['failed'])
            
            # Output files
            if stats['output_files']:
                st.markdown("### 💾 Output Files")
                for filepath in stats['output_files']:
                    st.code(filepath, language="text")
                    
                    # Offer download if file exists
                    file_path = Path(filepath)
                    if file_path.exists():
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                f"📥 Download {file_path.name}",
                                data=f.read(),
                                file_name=file_path.name,
                                mime="text/csv"
                            )
            
            # Show failed URLs if any
            if summary['failed'] > 0:
                st.markdown("### ⚠️ Failed URLs")
                failed_results = [r for r in summary['results'] if not r['success']]
                
                with st.expander(f"View {len(failed_results)} Failed URLs"):
                    for result in failed_results:
                        st.markdown(f"- `{result['url']}`  \n  Error: {result.get('error', 'Unknown error')}")
            
            # Show successful results preview
            if summary['successful'] > 0:
                st.markdown("### ✅ Successful Crawls Preview")
                successful_results = [r for r in summary['results'] if r['success']]
                
                preview_count = min(5, len(successful_results))
                for i, result in enumerate(successful_results[:preview_count]):
                    with st.expander(f"✓ {result['url']} ({i+1}/{preview_count})"):
                        if result.get('content'):
                            content_preview = result['content'][:500]
                            st.text_area(
                                "Content Preview",
                                value=content_preview + "..." if len(result['content']) > 500 else content_preview,
                                height=150,
                                disabled=True
                            )
            
            # Clear results button
            if st.button("Clear Results"):
                st.session_state.crawl_results = None
                st.session_state.crawl_logs = []
                st.rerun()
        else:
            st.info("No crawl results available. Run a crawl first.")
    
    with tab2:
        st.subheader("URL Filtering")
        st.markdown("Filter out unwanted URLs from crawled CSV files before LLM extraction")
        
        st.info("This step removes URLs containing common non-article patterns like `/about`, `/author`, `/contact`, etc.")
        
        # Get list of available CSV files from crawled_data
        crawled_data_path = Path("crawled_data")
        crawled_data_path.mkdir(parents=True, exist_ok=True)
        available_csvs = []
        
        if crawled_data_path.exists():
            available_csvs = [
                f.name for f in crawled_data_path.iterdir()
                if f.is_file() and f.suffix == '.csv'
            ]
        
        # If local folder is empty, try to retrieve from S3
        if not available_csvs:
            try:
                from aws_storage import get_storage
                s3_storage = get_storage()
                
                with st.spinner("📥 Checking S3 for crawled data..."):
                    # List all CSV files in S3 crawled_data prefix
                    s3_csv_files = s3_storage.list_files(prefix="crawled_data/", suffix=".csv")
                    
                    if s3_csv_files:
                        st.info(f"Found {len(s3_csv_files)} CSV file(s) in S3. Downloading...")
                        
                        # Download each file
                        for s3_key in s3_csv_files:
                            file_name = s3_key.split('/')[-1]
                            local_path = crawled_data_path / file_name
                            
                            if s3_storage.download_file(s3_key, str(local_path)):
                                available_csvs.append(file_name)
                                st.success(f"✓ Downloaded: {file_name}")
                            else:
                                st.warning(f"⚠️ Failed to download: {file_name}")
                        
                        if available_csvs:
                            st.success(f"✅ Successfully retrieved {len(available_csvs)} file(s) from S3")
                    else:
                        st.info("No CSV files found in S3 crawled_data folder")
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve from S3: {str(e)}")
        
        if available_csvs:
            # Sort files by modification time (newest first)
            available_csvs.sort(
                key=lambda x: (crawled_data_path / x).stat().st_mtime,
                reverse=True
            )
            
            st.subheader("📁 Select CSV File")
            selected_csv = st.selectbox(
                "Choose a CSV file from crawled_data/",
                options=available_csvs,
                help="Select a crawled CSV file to filter",
                key="url_filter_csv"
            )
            
            if selected_csv:
                csv_path = crawled_data_path / selected_csv
                
                # Preview the file
                try:
                    df = pd.read_csv(csv_path)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total URLs", len(df))
                    with col2:
                        if 'url' in df.columns:
                            unique_urls = df['url'].nunique()
                            st.metric("Unique URLs", unique_urls)
                    with col3:
                        st.metric("Columns", len(df.columns))
                    
                    # Filter patterns configuration
                    st.subheader("🔧 Filter Configuration")
                    
                    default_patterns = [
                        "/about", "/author", "/contact", "/supporters", 
                        "/support", "/donate", "/people", "/video", "/podcast", "/issue",
                        "/FAQ", "/terms", "/privacy", "/login", "/signup", "/register", "/subscribe", 
                        "/advertise", "/press", "/careers", "/shop", "/profile", "/settings", "/search",
                        "/cookies", "/sitemap", "/feed", "/news_feed", "/ads", "/ad", "/sponsor", "/sponsored", 
                        "/issues"
                    ]
                    
                    # Allow users to customize patterns
                    custom_patterns = st.text_area(
                        "URL patterns to filter",
                        value="\n".join(default_patterns),
                        height=200,
                        help="Enter URL patterns to exclude. URLs containing any of these patterns will be removed.",
                        label_visibility="collapsed"
                    )
                    
                    # Parse patterns
                    filter_patterns = [p.strip() for p in custom_patterns.split('\n') if p.strip()]
                    
                    # Preview filtering
                    if 'url' in df.columns and filter_patterns:
                        st.subheader("📊 Filter Preview")
                        
                        # Count URLs that will be filtered
                        mask = df['url'].apply(
                            lambda url: any(pattern in str(url).lower() for pattern in filter_patterns)
                        )
                        urls_to_remove = mask.sum()
                        urls_to_keep = len(df) - urls_to_remove
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("URLs to Keep", urls_to_keep, delta=None)
                        with col2:
                            st.metric("URLs to Remove", urls_to_remove, delta=f"-{urls_to_remove}")
                        with col3:
                            removal_pct = (urls_to_remove / len(df) * 100) if len(df) > 0 else 0
                            st.metric("Removal Rate", f"{removal_pct:.1f}%")
                        
                        # Show examples of URLs to be removed
                        if urls_to_remove > 0:
                            with st.expander("🗑️ Preview URLs to be Removed", expanded=False):
                                removed_urls = df[mask]['url'].head(20)
                                for idx, url in enumerate(removed_urls, 1):
                                    # Highlight which pattern matched
                                    matched_patterns = [p for p in filter_patterns if p in str(url).lower()]
                                    st.caption(f"{idx}. {url}")
                                    if matched_patterns:
                                        st.caption(f"   ↳ Matches: {', '.join(matched_patterns)}")
                        
                        # Show examples of URLs to be kept
                        if urls_to_keep > 0:
                            with st.expander("✅ Preview URLs to be Kept", expanded=False):
                                kept_urls = df[~mask]['url'].head(20)
                                for idx, url in enumerate(kept_urls, 1):
                                    st.caption(f"{idx}. {url}")
                        
                        st.divider()
                        
                        # Output configuration
                        st.subheader("💾 Output Configuration")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            output_folder = st.text_input(
                                "Output Folder",
                                value="crawled_data",
                                help="Folder to save filtered CSV"
                            )
                        
                        with col2:
                            # Suggest output filename
                            base_name = csv_path.stem
                            suggested_name = f"{base_name}_filtered.csv"
                            output_filename = st.text_input(
                                "Output Filename",
                                value=suggested_name,
                                help="Name for the filtered CSV file"
                            )
                        
                        # Apply filtering button
                        if st.button("🔧 Apply Filter", type="primary", use_container_width=True):
                            try:
                                # Filter the dataframe
                                df_filtered = df[~mask].copy()
                                
                                # Save to output folder
                                output_path = Path(output_folder)
                                output_path.mkdir(parents=True, exist_ok=True)
                                output_file = output_path / output_filename
                                
                                df_filtered.to_csv(output_file, index=False)
                                
                                # Upload to S3 if configured
                                s3_upload_success = False
                                try:
                                    from aws_storage import get_storage
                                    s3_storage = get_storage()
                                    
                                    # Upload filtered CSV to S3
                                    s3_key = f"crawled_data/{output_filename}"
                                    if s3_storage.upload_file(str(output_file), s3_key):
                                        st.success(f"☁️ Uploaded to S3: s3://{s3_storage.bucket_name}/{s3_key}")
                                        s3_upload_success = True
                                    else:
                                        st.warning("⚠️ S3 upload failed")
                                except Exception as e:
                                    st.warning(f"⚠️ S3 upload skipped: {str(e)}")
                                
                                # Delete original file after successful S3 upload
                                if s3_upload_success and csv_path.exists():
                                    try:
                                        csv_path.unlink()
                                        st.info(f"🗑️ Original file deleted: `{csv_path.name}`")
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not delete original file: {e}")
                                
                                # Show success message
                                st.success(f"✅ Filtered CSV saved to `{output_file}`")
                                st.info(f"📊 Kept {len(df_filtered)} of {len(df)} URLs ({len(df_filtered)/len(df)*100:.1f}%)")
                                
                                # Show download button
                                csv_data = df_filtered.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Filtered CSV",
                                    data=csv_data,
                                    file_name=output_filename,
                                    mime="text/csv",
                                    use_container_width=True
                                )
                                
                                # Log the filtering action
                                log_file = output_path / f"{base_name}_filter_log.txt"
                                with open(log_file, 'w') as f:
                                    f.write(f"URL Filtering Log\n")
                                    f.write(f"=" * 60 + "\n")
                                    f.write(f"Source File: {csv_path}\n")
                                    f.write(f"Output File: {output_file}\n")
                                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                    f.write(f"\n")
                                    f.write(f"Filter Patterns:\n")
                                    for pattern in filter_patterns:
                                        f.write(f"  - {pattern}\n")
                                    f.write(f"\n")
                                    f.write(f"Results:\n")
                                    f.write(f"  Total URLs: {len(df)}\n")
                                    f.write(f"  URLs Kept: {len(df_filtered)}\n")
                                    f.write(f"  URLs Removed: {urls_to_remove}\n")
                                    f.write(f"  Retention Rate: {len(df_filtered)/len(df)*100:.1f}%\n")
                                
                                st.caption(f"📄 Filter log saved to `{log_file}`")
                                
                            except Exception as e:
                                st.error(f"Error applying filter: {e}")
                    
                    else:
                        st.warning("CSV file must have a 'url' column to filter")
                        
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
        else:
            st.warning("⚠️ No CSV files found in `crawled_data/` folder. Please crawl some websites first in the 'Crawl Websites' tab.")


# LLM Extraction Page
elif page == "LLM Extraction":
    st.header("LLM Extraction")
    st.markdown("Use AI models to intelligently extract structured metadata from crawled CSV content")

    # CSV source selection
    st.subheader("Select CSV File from Web Crawler")
    
    selected_source = None
    
    # Get list of available CSV files
    crawled_data_path = Path("crawled_data")
    crawled_data_path.mkdir(parents=True, exist_ok=True)
    available_csvs = []
    
    if crawled_data_path.exists():
        available_csvs = [
            f.name for f in crawled_data_path.iterdir()
            if f.is_file() and f.suffix == '.csv'
        ]
    
    # If local folder is empty, try to retrieve from S3
    if not available_csvs:
        try:
            from aws_storage import get_storage
            s3_storage = get_storage()
            
            with st.spinner("📥 Checking S3 for crawled data..."):
                # List all CSV files in S3 crawled_data prefix
                s3_csv_files = s3_storage.list_files(prefix="crawled_data/", suffix=".csv")
                
                if s3_csv_files:
                    st.info(f"Found {len(s3_csv_files)} CSV file(s) in S3. Downloading...")
                    
                    # Download each file
                    for s3_key in s3_csv_files:
                        file_name = s3_key.split('/')[-1]
                        local_path = crawled_data_path / file_name
                        
                        if s3_storage.download_file(s3_key, str(local_path)):
                            available_csvs.append(file_name)
                            st.success(f"✓ Downloaded: {file_name}")
                        else:
                            st.warning(f"⚠️ Failed to download: {file_name}")
                    
                    if available_csvs:
                        st.success(f"✅ Successfully retrieved {len(available_csvs)} file(s) from S3")
                else:
                    st.info("No CSV files found in S3 crawled_data folder")
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve from S3: {str(e)}")
    
    if available_csvs:
        # Sort files by modification time (newest first)
        available_csvs.sort(
            key=lambda x: (crawled_data_path / x).stat().st_mtime,
            reverse=True
        )
        
        csv_file_name = st.selectbox(
            "Select CSV File",
            options=available_csvs,
            help="Choose a CSV file from crawled_data/ directory",
            key="llm_extraction_csv"
        )
        selected_source = crawled_data_path / csv_file_name
        
        # Show file info
        if selected_source.exists():
            try:
                df_preview = pd.read_csv(selected_source)
                st.caption(f"📁 {len(df_preview)} rows in this CSV file")
                
                # Show column info
                if 'text_content' in df_preview.columns:
                    st.success("✅ Found 'text_content' column")
                else:
                    st.warning(f"⚠️ 'text_content' column not found. Available columns: {', '.join(df_preview.columns)}")
                
                # Preview first few rows
                with st.expander("Preview CSV Data"):
                    st.dataframe(df_preview.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        st.warning("No CSV files found in crawled_data/. Please run the Web Crawler first.")
        
        # Manual file upload option
        uploaded_file = st.file_uploader(
            "Or upload a CSV file",
            type=['csv'],
            help="Upload a CSV file with a 'text_content' column",
            key="llm_csv_upload"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_csv_path = temp_dir / uploaded_file.name
            
            with open(temp_csv_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            selected_source = temp_csv_path
            st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Model configuration (uses your configured LLM provider)
    st.subheader("Model Configuration")
    
    # Get current LLM provider from environment
    current_provider = os.getenv("LLM_PROVIDER", "azure").lower()
    provider_display = {
        "azure": "Azure OpenAI",
        "openai": "OpenAI",
        "lm_studio": "LM Studio (Local)"
    }.get(current_provider, "Azure OpenAI")
    
    # Display provider info
    st.info(f"🤖 Using configured provider: **{provider_display}**")
    
    # For LM Studio, try to get the actual loaded model name
    if current_provider == "lm_studio":
        try:
            import requests
            base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
            models_url = base_url.replace("/v1", "") + "/v1/models"
            
            response = requests.get(models_url, timeout=2)
            if response.status_code == 200:
                models_data = response.json()
                if models_data.get("data") and len(models_data["data"]) > 0:
                    loaded_model = models_data["data"][0].get("id", "Unknown")
                    st.success(f"Loaded Model: **{loaded_model}**")
                else:
                    st.warning("⚠️ No model loaded in LM Studio")
            else:
                st.warning(f"⚠️ Could not fetch model info (Status: {response.status_code})")
        except requests.exceptions.RequestException:
            st.warning("⚠️ Could not connect to LM Studio. Make sure it's running.")
        except Exception as e:
            st.caption(f"Could not fetch model info: {str(e)}")
    
    st.caption("Change provider in sidebar Settings if needed")
    
    # Determine model name based on provider
    if current_provider == "azure":
        model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4")
    elif current_provider == "openai":
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    else:  # lm_studio
        model_name = "local-model"

    # Output folder
    output_folder = st.text_input(
        "Output Folder",
        value="processed_data",
        help="Folder to save processed CSV/JSON files",
        key="llm_output_folder"
    )

    # Checkpointing configuration
    st.subheader("💾 Checkpointing & Resume")
    
    # Check if checkpoint exists for selected file
    checkpoint_info = None
    if selected_source and selected_source.exists():
        from agents.llm_extractor import CheckpointManager
        checkpoint_file = Path(output_folder) / f"{selected_source.stem}_checkpoint.json"
        
        if checkpoint_file.exists():
            try:
                checkpoint_manager = CheckpointManager(checkpoint_file)
                checkpoint_data = checkpoint_manager.checkpoint_data
                
                if checkpoint_data.get('processed_indices'):
                    processed_count = len(checkpoint_data['processed_indices'])
                    total_count = checkpoint_data.get('total_rows', 0)
                    last_save = checkpoint_data.get('last_save_time')
                    
                    if last_save:
                        from datetime import datetime
                        try:
                            last_save_dt = datetime.fromisoformat(last_save.replace('Z', '+00:00'))
                            last_save_str = last_save_dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            last_save_str = last_save
                    
                    checkpoint_info = {
                        'processed': processed_count,
                        'total': total_count,
                        'last_save': last_save_str if 'last_save_str' in locals() else last_save,
                        'progress': processed_count / total_count * 100 if total_count > 0 else 0
                    }
                    
                    st.success(f"📁 **Checkpoint Found!** {processed_count}/{total_count} rows processed ({checkpoint_info['progress']:.1f}%)")
                    st.caption(f"Last saved: {checkpoint_info['last_save']}")
                    
                    # Show resume option
                    resume_from_checkpoint = st.checkbox(
                        "🔄 Resume from checkpoint",
                        value=True,
                        help=f"Continue processing from row {processed_count + 1}. Uncheck to start fresh."
                    )
                else:
                    st.info("ℹ️ No valid checkpoint data found")
                    resume_from_checkpoint = False
            except Exception as e:
                st.warning(f"⚠️ Could not read checkpoint: {e}")
                resume_from_checkpoint = False
        else:
            st.info("ℹ️ No checkpoint found - will start fresh processing")
            resume_from_checkpoint = False
    
    # Checkpoint interval setting
    col1, col2 = st.columns(2)
    with col1:
        checkpoint_interval = st.number_input(
            "Checkpoint Interval",
            min_value=5,
            max_value=100,
            value=10,
            step=5,
            help="Save progress every N rows (recommended: 10-20)"
        )
    
    with col2:
        if checkpoint_info:
            st.metric("Current Progress", f"{checkpoint_info['processed']}/{checkpoint_info['total']}", 
                     f"{checkpoint_info['progress']:.1f}%")

    # Interrupt handling info
    st.info("ℹ️ **Interrupt Handling:** Processing can be safely interrupted with Ctrl+C. Progress will be automatically saved and can be resumed later.")

    # Start processing button
    if st.button("🤖 Start LLM Extraction", type="primary", use_container_width=True):
        if not selected_source or not selected_source.exists():
            st.error(f"Source '{selected_source}' does not exist")
        else:
            # Check if file is CSV
            if not str(selected_source).endswith('.csv'):
                st.error("Selected file is not a CSV file")
                st.stop()
            
            st.info(f"Processing CSV file: {selected_source.name}")

            # Create progress containers
            progress_container = st.container()
            status_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
                time_text = st.empty()

            with status_container:
                status_text = st.empty()

            # Start processing with progress tracking
            try:
                from agents.llm_extractor import (
                    process_csv_with_progress,
                    get_openai_client
                )
                import asyncio

                # Set processing state
                st.session_state.processing_in_progress = True
                st.session_state.processing_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
                st.session_state.processing_start_time = time.time()

                def processing_progress_callback(message, current, total):
                    """Progress callback for processing updates"""
                    if total > 0:
                        progress = current / total
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing: {current}/{total} ({progress*100:.1f}%)")

                        # Calculate time estimates
                        elapsed = time.time() - st.session_state.get('processing_start_time', time.time())
                        remaining = 0
                        if current > 0:
                            estimated_total = elapsed * total / current
                            remaining = max(0, estimated_total - elapsed)

                            time_text.text(f"⏱️ Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining)}")
                        else:
                            time_text.text(f"⏱️ Elapsed: {format_time(elapsed)} | Remaining: Calculating...")

                        # Update session state
                        st.session_state.processing_progress = {
                            'current': current,
                            'total': total,
                            'elapsed': elapsed,
                            'remaining': remaining
                        }

                        status_text.text(f"🔄 {message}")

                # Get client based on current provider
                client = get_openai_client(
                    provider=current_provider,
                    base_url=os.getenv("LM_STUDIO_BASE_URL") if current_provider == "lm_studio" else None
                )

                # Run the LLM extractor for CSV
                df, stats = asyncio.run(process_csv_with_progress(
                    csv_path=selected_source,
                    output_dir=Path(output_folder),
                    client=client,
                    model_name=model_name,
                    text_column="text_content",
                    progress_callback=processing_progress_callback,
                    checkpoint_interval=checkpoint_interval,
                    resume_from_checkpoint=resume_from_checkpoint if 'resume_from_checkpoint' in locals() else True
                ))

                st.session_state.csv_processed_df = df
                st.session_state.csv_metadata = stats

                # Clear processing state
                st.session_state.processing_in_progress = False

                # Show completion message
                progress_bar.progress(1.0)
                progress_text.text("✅ LLM Extraction Complete!")
                total_time = time.time() - st.session_state.processing_start_time
                time_text.text(f"⏱️ Total time: {format_time(total_time)}")
                
                status_text.text(f"🎉 Successfully processed {stats.get('filtered_rows', len(df))} rows (after filtering)")

                # Show completion statistics
                st.success("✅ LLM extraction complete!")
                
                # Show S3 upload status
                if stats.get('output_csv') and stats.get('output_json'):
                    # Extract filename from path
                    import re
                    source_name = selected_source.stem if selected_source else "extracted"
                    date_pattern = r'_\d{8}$'
                    source_name = re.sub(date_pattern, '', source_name)
                    date_str = datetime.now().strftime('%Y%m%d')
                    
                    st.info(f"📤 Files uploaded to S3 bucket in `processed_data/`:\n- `{source_name}_{date_str}.csv`\n- `{source_name}_{date_str}.json`")

                # Display processing stats with filtering information
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Rows", stats.get('total_rows', len(df)))
                with col2:
                    st.metric("Processed", stats.get('processed', 0))
                with col3:
                    st.metric("Failed", stats.get('skipped_error', 0))
                with col4:
                    st.metric("Filtered Out", stats.get('removed_empty_content', 0) + stats.get('removed_old_date', 0))
                with col5:
                    st.metric("Final Rows", stats.get('filtered_rows', len(df)))
                
                # Show filtering details
                if stats.get('removed_empty_content', 0) > 0 or stats.get('removed_old_date', 0) > 0:
                    with st.expander("🔍 Filtering Details"):
                        st.write(f"**Removed empty content:** {stats.get('removed_empty_content', 0)} rows")
                        st.write(f"**Removed old dates (>2 years):** {stats.get('removed_old_date', 0)} rows")
                        st.write(f"**Date threshold:** {stats.get('filter_date_threshold', 'N/A')}")
                        st.caption("Rows with empty extracted content or publication dates older than 2 years were automatically filtered out.")

                # Show results
                st.markdown("---")
                st.subheader("Extraction Results")
                
                # Display with AgGrid for better text wrapping
                from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
                
                gb = GridOptionsBuilder.from_dataframe(df)
                
                # Configure grid options
                gb.configure_default_column(
                    wrapText=True,
                    autoHeight=True,
                    resizable=True,
                    filterable=True,
                    sortable=True,
                    enableCellTextSelection=True
                )
                
                # Configure specific columns if they exist
                if 'content' in df.columns:
                    gb.configure_column(
                        'content',
                        wrapText=True,
                        autoHeight=True,
                        cellStyle={'white-space': 'normal'},
                        minWidth=300,
                        enableCellTextSelection=True
                    )
                
                if 'title' in df.columns:
                    gb.configure_column(
                        'title',
                        wrapText=True,
                        autoHeight=True,
                        minWidth=200,
                        enableCellTextSelection=True
                    )
                
                if 'url' in df.columns:
                    gb.configure_column(
                        'url',
                        wrapText=True,
                        autoHeight=True,
                        minWidth=250,
                        enableCellTextSelection=True
                    )
                
                # Pagination
                gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
                gb.configure_grid_options(domLayout='normal', enableCellTextSelection=True, ensureDomOrder=True)
                
                gridOptions = gb.build()
                
                # Display AgGrid
                AgGrid(
                    df,
                    gridOptions=gridOptions,
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    height=600,
                    theme='streamlit',
                    allow_unsafe_jscode=True,
                    enable_enterprise_modules=False
                )
                
                # Download options
                col1, col2 = st.columns(2)
                
                # Generate filename from source name
                source_name = selected_source.stem if selected_source else "extracted"
                
                # Remove any existing date suffix (e.g., "canarymedia_20251115" -> "canarymedia")
                import re
                date_pattern = r'_\d{8}$'  # Matches "_YYYYMMDD" at the end
                source_name = re.sub(date_pattern, '', source_name)
                
                date_str = datetime.now().strftime('%Y%m%d')
                
                with col1:
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"{source_name}_{date_str}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    json_data = df.to_json(orient='records', indent=2).encode('utf-8')
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"{source_name}_{date_str}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                # Show info about extracted fields
                with st.expander("ℹ️ Extracted Fields Information"):
                    st.markdown("""
                    **Extracted Fields:**
                    - **URL**: Original article URL (if available)
                    - **Title**: Extracted article title
                    - **Publication Date**: Extracted publication date
                    - **Main Content**: Extracted main article content
                    - **Categories**: Extracted article categories/topics
                    
                    The extraction uses your configured LLM provider ({}) to intelligently
                    parse content and extract structured metadata.
                    """.format(provider_display))

            except Exception as e:
                st.session_state.processing_in_progress = False
                st.error(f"LLM extraction failed: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())


# Summarization Page
elif page == "Summarization":
    st.header("Summarization")
    st.markdown("Upload CSV files with a 'content' column to generate tech-intelligence summaries and automatic categorization")

    # Check if processing flag is stuck (interrupted by navigation)
    if st.session_state.csv_processing and st.session_state.csv_processed_df is None:
        st.warning("⚠️ **Previous processing was interrupted.** The task did not complete because you navigated away from this page.")
        if st.button("Clear Interrupted Task", type="secondary"):
            st.session_state.csv_processing = False
            st.session_state.csv_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
            st.rerun()
        st.divider()

    # Create tabs for upload and history
    tab1, tab2 = st.tabs(["Select & Process", "History"])

    with tab1:
        st.subheader("Select CSV File")
        st.markdown("""
        **Requirements:**
        - CSV file must contain a column named `content` or `text_content`
        - The content column should contain text to be summarized
        - Each row will be processed independently
        """)
        
        st.divider()
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            model_provider = st.selectbox(
                "Provider",
                ["Azure OpenAI", "LM Studio (Local)"],
                help="Select the AI model provider"
            )
        
        with col2:
            if model_provider == "Azure OpenAI":
                azure_model_name = st.selectbox(
                    "Model",
                    ["pmo-gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "gpt-4"],
                    help="Select Azure OpenAI model"
                )
                st.session_state.selected_model_config = {
                    'provider': 'azure',
                    'model_name': azure_model_name
                }
            else:  # LM Studio
                lm_studio_url = st.text_input(
                    "LM Studio URL",
                    value="http://127.0.0.1:1234/v1",
                    help="LM Studio API endpoint"
                )
                st.session_state.selected_model_config = {
                    'provider': 'lm_studio',
                    'base_url': lm_studio_url,
                    'model_name': 'local-model'
                }
                
                # Try to get the actual loaded model name
                try:
                    import requests
                    models_url = lm_studio_url.replace("/v1", "") + "/v1/models"
                    
                    response = requests.get(models_url, timeout=2)
                    if response.status_code == 200:
                        models_data = response.json()
                        if models_data.get("data") and len(models_data["data"]) > 0:
                            loaded_model = models_data["data"][0].get("id", "Unknown")
                            st.success(f"✅ **Loaded Model:** `{loaded_model}`")
                        else:
                            st.warning("⚠️ No model loaded in LM Studio")
                    else:
                        st.info("💡 Make sure LM Studio is running and a model is loaded at the specified URL")
                except requests.exceptions.RequestException:
                    st.warning("⚠️ Could not connect to LM Studio. Make sure it's running.")
                except Exception:
                    st.info("💡 Make sure LM Studio is running and a model is loaded at the specified URL")
        
        st.divider()

        # CSV file selection from processed_data folder
        st.subheader("📁 Select CSV File")
        
        processed_data_path = Path("processed_data")
        processed_data_path.mkdir(parents=True, exist_ok=True)
        available_csvs = []
        selected_csv_path = None
        
        if processed_data_path.exists():
            available_csvs = [
                f.name for f in processed_data_path.iterdir()
                if f.is_file() and f.suffix == '.csv'
            ]
        
        # If local folder is empty, try to retrieve from S3
        if not available_csvs:
            try:
                from aws_storage import get_storage
                s3_storage = get_storage()
                
                with st.spinner("📥 Checking S3 for processed data..."):
                    # List all CSV files in S3 processed_data prefix
                    s3_csv_files = s3_storage.list_files(prefix="processed_data/", suffix=".csv")
                    
                    if s3_csv_files:
                        st.info(f"Found {len(s3_csv_files)} CSV file(s) in S3. Downloading...")
                        
                        # Download each file
                        for s3_key in s3_csv_files:
                            file_name = s3_key.split('/')[-1]
                            local_path = processed_data_path / file_name
                            
                            if s3_storage.download_file(s3_key, str(local_path)):
                                available_csvs.append(file_name)
                                st.success(f"✓ Downloaded: {file_name}")
                            else:
                                st.warning(f"⚠️ Failed to download: {file_name}")
                        
                        if available_csvs:
                            st.success(f"✅ Successfully retrieved {len(available_csvs)} file(s) from S3")
                    else:
                        st.info("No CSV files found in S3 processed_data folder")
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve from S3: {str(e)}")
        
        if available_csvs:
            # Sort files by modification time (newest first)
            available_csvs.sort(
                key=lambda x: (processed_data_path / x).stat().st_mtime,
                reverse=True
            )
            
            selected_csv_name = st.selectbox(
                "Select CSV File from processed_data/",
                options=available_csvs,
                help="Choose a CSV file from the processed_data directory",
                key="summarization_csv_select"
            )
            selected_csv_path = processed_data_path / selected_csv_name
            
        else:
            st.warning("⚠️ No CSV files found in `processed_data/` folder. Please process some files in LLM Extraction first.")
            st.info("💡 You can also manually place CSV files in the `processed_data/` folder.")
            st.stop()

        if selected_csv_path and selected_csv_path.exists():
            try:
                # Read the CSV to preview
                df_preview = pd.read_csv(selected_csv_path)
                
                st.success(f"File loaded: {selected_csv_path.name}")
                
                # Show file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df_preview))
                with col2:
                    st.metric("Total Columns", len(df_preview.columns))
                with col3:
                    # Check for either 'content' or 'text_content' column
                    content_col = 'content' if 'content' in df_preview.columns else ('text_content' if 'text_content' in df_preview.columns else None)
                    has_content = content_col is not None
                    st.metric(f"Has content column", "✓" if has_content else "✗")

                # Show column names
                with st.expander("View Columns"):
                    st.write(df_preview.columns.tolist())

                # Check if content column exists
                content_col = 'content' if 'content' in df_preview.columns else ('text_content' if 'text_content' in df_preview.columns else None)
                if content_col is None:
                    st.error("❌ CSV must contain a 'content' or 'text_content' column")
                    st.info(f"Available columns: {', '.join(df_preview.columns)}")
                    st.stop()

                # Preview data
                st.subheader("Data Preview")
                st.dataframe(df_preview.head(10), use_container_width=True)

                # Process button
                if st.button("Start Summarization", type="primary", use_container_width=True):
                    # Use the selected CSV file path directly (no need for temp file)
                    try:
                        # Set processing flag
                        st.session_state.csv_processing = True
                        st.session_state.csv_progress = {
                            'current': 0,
                            'total': len(df_preview),
                            'elapsed': 0,
                            'remaining': 0
                        }
                        
                        # Show warning about navigation
                        st.warning("⚠️ **Important:** Processing will continue only while you stay on this page. Navigating to another section will interrupt the task. Please wait for completion.")
                        
                        # Process the CSV
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        time_text = st.empty()
                        
                        # Create a container for progress updates
                        progress_info = {
                            'current': 0,
                            'total': len(df_preview),
                            'elapsed': 0,
                            'remaining': 0
                        }
                        
                        def format_time(seconds):
                            """Format seconds into human-readable time"""
                            if seconds < 60:
                                return f"{seconds:.0f}s"
                            elif seconds < 3600:
                                mins = int(seconds // 60)
                                secs = int(seconds % 60)
                                return f"{mins}m {secs}s"
                            else:
                                hours = int(seconds // 3600)
                                mins = int((seconds % 3600) // 60)
                                return f"{hours}h {mins}m"
                        
                        def update_progress(current, total, elapsed, est_remaining):
                            """Update progress display with time estimates"""
                            progress_info['current'] = current
                            progress_info['elapsed'] = elapsed
                            progress_info['remaining'] = est_remaining
                            
                            # Update session state for sidebar display
                            st.session_state.csv_progress = {
                                'current': current,
                                'total': total,
                                'elapsed': elapsed,
                                'remaining': est_remaining
                            }
                            
                            # Update progress bar
                            progress = current / total if total > 0 else 0
                            progress_bar.progress(progress)
                            
                            # Update status text
                            status_text.text(f"Processing row {current}/{total} (summarizing & classifying)...")
                            
                            # Update time information with dynamic estimates
                            elapsed_str = format_time(elapsed)
                            remaining_str = format_time(est_remaining)
                            
                            if current < total:
                                time_text.markdown(
                                    f"**Time Elapsed:** {elapsed_str} | "
                                    f"**Estimated Remaining:** {remaining_str} | "
                                    f"**Progress:** {current}/{total} rows ({progress*100:.1f}%)"
                                )
                            else:
                                time_text.markdown(
                                    f"**Total Duration:** {elapsed_str} | "
                                    f"**Completed:** {total} rows (100%)"
                                )

                        async def process_with_progress():
                            # Get selected model
                            model_config = st.session_state.get('selected_model_config', {'provider': 'azure', 'model_name': 'pmo-gpt-4.1-nano'})
                            selected_model = get_model(**model_config)
                            
                            # Type guard to ensure selected_csv_path and content_col are not None
                            if selected_csv_path is None:
                                raise ValueError("No CSV file selected")
                            if content_col is None:
                                raise ValueError("No content column selected")
                            
                            # Use the detected content column
                            df_result, duration, metadata = await summarize_csv_file(
                                selected_csv_path, 
                                content_col,
                                progress_callback=update_progress,
                                custom_model=selected_model
                            )
                            return df_result, duration, metadata

                        df_result, duration, metadata = run_async(process_with_progress())

                        progress_bar.progress(1.0)
                        status_text.text("✓ Processing complete!")
                        time_text.markdown(
                            f"✅ **Total Duration:** {format_time(duration)} | "
                            f"📊 **Completed:** {len(df_result)} rows (100%)"
                        )

                        # Store in session state
                        st.session_state.csv_processed_df = df_result
                        metadata['source_file'] = selected_csv_path.name
                        st.session_state.csv_metadata = CSVSummarizationMetadata(**metadata)
                        
                        # Auto-save to summarised_content folder and S3
                        try:
                            csv_path, json_path, log_path = save_summarized_csv(
                                df_result,
                                metadata
                            )
                            
                            # Update metadata with paths
                            st.session_state.csv_metadata.output_csv_path = str(csv_path)
                            st.session_state.csv_metadata.output_json_path = str(json_path)
                            st.session_state.csv_metadata.output_log_path = str(log_path)
                            
                            # Save to history
                            history_path = Path("summarised_content") / "history.json"
                            history = CSVSummarizationHistory.from_file(history_path)
                            history.add_file(st.session_state.csv_metadata)
                            history.to_file(history_path)
                            
                            logging.info(f"Auto-saved files: CSV={csv_path.name}, JSON={json_path.name}, Log={log_path.name}")
                            
                        except Exception as e:
                            logging.error(f"Auto-save failed: {e}")
                            st.warning(f"⚠️ Auto-save failed: {e}")
                        
                        # Clear processing flag
                        st.session_state.csv_processing = False

                        st.success(f"✓ Summarization complete in {duration:.2f} seconds ({format_time(duration)})!")
                        st.info("📁 Files automatically saved to `summarised_content/` folder and uploaded to S3")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing CSV: {e}")
                        logging.error(f"CSV processing error: {e}")
                        # Clear processing flag on error
                        st.session_state.csv_processing = False

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

        # Show processed results
        if st.session_state.csv_processed_df is not None and st.session_state.csv_metadata is not None:
            st.divider()
            st.subheader("✓ Processing Complete")

            metadata = st.session_state.csv_metadata
            df_result = st.session_state.csv_processed_df

            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", metadata.total_files if hasattr(metadata, 'total_files') else len(df_result))
            with col2:
                st.metric("Successful", metadata.processed if hasattr(metadata, 'processed') else len(df_result))
            with col3:
                st.metric("Failed", metadata.skipped_error if hasattr(metadata, 'skipped_error') else 0)
            with col4:
                total = metadata.total_files if hasattr(metadata, 'total_files') else len(df_result)
                success = metadata.processed if hasattr(metadata, 'processed') else len(df_result)
                success_rate = (success / total * 100) if total > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")

            col1, col2 = st.columns(2)
            with col1:
                duration = metadata.duration_seconds if hasattr(metadata, 'duration_seconds') else 0
                st.metric("Total Duration", f"{duration:.2f}s")
            with col2:
                avg_time = metadata.avg_time_per_row if hasattr(metadata, 'avg_time_per_row') else 0
                st.metric("Avg per Row", f"{avg_time:.2f}s")

            # Preview results
            st.subheader("Preview Summarized Content")
            
            # Fixed to show 5 rows
            preview_count = min(5, len(df_result))
            
            # Show results in expandable sections
            for idx in range(preview_count):
                row = df_result.iloc[idx]
                with st.expander(f"Row {idx + 1}", expanded=(idx == 0)):
                    # Show tech intelligence fields at the top
                    tech_fields = []
                    if row.get('Dimension'):
                        tech_fields.append(f"Dimension: {row['Dimension']}")
                    if row.get('Tech'):
                        tech_fields.append(f"Tech: {row['Tech']}")
                    if row.get('URL to start-up(s)') and str(row['URL to start-up(s)']) != 'N/A':
                        tech_fields.append(f"URL to start-up(s): {row['URL to start-up(s)']}")
                    
                    if tech_fields:
                        st.markdown(f"**Tech Intelligence:** :blue[{', '.join(tech_fields)}]")
                        st.divider()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Content:**")
                        # Use .get() to safely access content column (could be 'content' or 'text_content')
                        content = str(row.get('content', row.get('text_content', 'No content available')))
                        st.text_area(
                            "Original",
                            value=content[:500] + "..." if len(content) > 500 else content,
                            height=150,
                            key=f"orig_{idx}",
                            label_visibility="collapsed"
                        )
                    with col2:
                        st.markdown("**Tech Intelligence Analysis:**")
                        # Display tech intel fields
                        indicator = str(row.get('Indicator', 'No analysis available'))
                        dimension = str(row.get('Dimension', ''))
                        tech = str(row.get('Tech', ''))
                        trl = str(row.get('TRL', ''))
                        startup = str(row.get('URL to start-up(s)', ''))
                        
                        tech_intel_text = f"Indicator: {indicator}\n\n"
                        if dimension:
                            tech_intel_text += f"Dimension: {dimension}\n"
                        if tech:
                            tech_intel_text += f"Tech: {tech}\n"
                        if trl:
                            tech_intel_text += f"TRL: {trl}\n"
                        if startup and startup != 'N/A':
                            tech_intel_text += f"URL to start-up(s): {startup}\n"
                        
                        st.text_area(
                            "Tech Intelligence",
                            value=tech_intel_text,
                            height=150,
                            key=f"tech_intel_{idx}",
                            label_visibility="collapsed"
                        )

            # Full data preview
            st.subheader("Full Dataset Preview")
            
            # Exclude text_content column from display
            display_columns = [col for col in df_result.columns if col != 'text_content']
            df_display = df_result[display_columns]
            
            # Use AgGrid for better text selection
            from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
            
            gb_preview = GridOptionsBuilder.from_dataframe(df_display)
            gb_preview.configure_default_column(
                resizable=True,
                filterable=True,
                sortable=True,
                wrapText=True,
                autoHeight=True,
                enableCellTextSelection=True
            )
            gb_preview.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
            gb_preview.configure_grid_options(enableCellTextSelection=True, ensureDomOrder=True)
            
            grid_options_preview = gb_preview.build()
            
            AgGrid(
                df_display,
                gridOptions=grid_options_preview,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                height=400,
                theme='streamlit',
                allow_unsafe_jscode=True
            )

            # Download options
            st.subheader("Download")
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df_result.to_csv(index=False).encode('utf-8')
                source_file = getattr(metadata, 'source_file', '') if hasattr(metadata, 'source_file') else metadata.get('source_file', 'output')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{Path(source_file).stem}_summarized.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # Create log content for download
                source_file = getattr(metadata, 'source_file', '') if hasattr(metadata, 'source_file') else metadata.get('source_file', 'Unknown')
                timestamp = getattr(metadata, 'timestamp', datetime.now()) if hasattr(metadata, 'timestamp') else metadata.get('timestamp', datetime.now())
                content_column = getattr(metadata, 'content_column', '') if hasattr(metadata, 'content_column') else metadata.get('content_column', 'content')
                total_rows = getattr(metadata, 'total_rows', len(df_result)) if hasattr(metadata, 'total_rows') else metadata.get('total_rows', len(df_result))
                successful = getattr(metadata, 'successful', len(df_result)) if hasattr(metadata, 'successful') else metadata.get('successful', len(df_result))
                failed = getattr(metadata, 'failed', 0) if hasattr(metadata, 'failed') else metadata.get('failed', 0)
                success_rate = getattr(metadata, 'success_rate', 0) if hasattr(metadata, 'success_rate') else metadata.get('success_rate', 0)
                duration = getattr(metadata, 'duration_seconds', 0) if hasattr(metadata, 'duration_seconds') else metadata.get('duration_seconds', 0)
                avg_time = getattr(metadata, 'avg_time_per_row', 0) if hasattr(metadata, 'avg_time_per_row') else metadata.get('avg_time_per_row', 0)
                
                log_content = f"""{'='*60}
SUMMARIZATION LOG
{'='*60}

Source File: {source_file}
Date: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Content Column: {content_column}

{'-'*60}
PROCESSING STATISTICS
{'-'*60}
Total Rows: {total_rows}
Successfully Processed: {successful}
Failed: {failed}
Success Rate: {success_rate:.2f}%

{'-'*60}
DURATION
{'-'*60}
Total Duration: {duration:.2f} seconds
Average per Row: {avg_time:.2f} seconds

{'='*60}
"""
                st.download_button(
                    label="📥 Download Log",
                    data=log_content,
                    file_name=f"{Path(source_file).stem}_log.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            # Reset button
            if st.button("Process Another File", use_container_width=True):
                st.session_state.csv_processed_df = None
                st.session_state.csv_metadata = None
                st.rerun()

    with tab2:
        st.subheader("Processing History")
        
        history_path = Path("summarised_content") / "history.json"
        
        if history_path.exists():
            try:
                history = CSVSummarizationHistory.from_file(history_path)
                
                if history.files:
                    st.info(f"Found {len(history.files)} processed file(s)")
                    
                    # Display each file in history
                    for idx, file_meta in enumerate(reversed(history.files), 1):
                        with st.expander(
                            f"**{idx}. {file_meta.source_file}** - {file_meta.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                            expanded=(idx == 1)
                        ):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Rows", file_meta.total_rows)
                                st.metric("Duration", f"{file_meta.duration_seconds:.2f}s")
                            with col2:
                                st.metric("Successful", file_meta.successful)
                                st.metric("Failed", file_meta.failed)
                            with col3:
                                st.metric("Success Rate", f"{file_meta.success_rate:.1f}%")
                                st.metric("Avg per Row", f"{file_meta.avg_time_per_row:.2f}s")
                            
                            # Show file paths if available
                            if file_meta.output_csv_path:
                                st.text(f"📁 CSV: {Path(file_meta.output_csv_path).name}")
                            if file_meta.output_log_path:
                                st.text(f"📄 Log: {Path(file_meta.output_log_path).name}")
                            
                            # Load and preview if files exist
                            if file_meta.output_csv_path and Path(file_meta.output_csv_path).exists():
                                if st.button(f"Preview File", key=f"preview_{idx}"):
                                    try:
                                        preview_df = pd.read_csv(file_meta.output_csv_path)
                                        st.dataframe(preview_df.head(5), use_container_width=True)
                                        
                                        # Download button for historical file
                                        csv_data = preview_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label="📥 Download This File",
                                            data=csv_data,
                                            file_name=Path(file_meta.output_csv_path).name,
                                            mime="text/csv",
                                            key=f"download_{idx}"
                                        )
                                    except Exception as e:
                                        st.error(f"Error loading file: {e}")
                else:
                    st.info("No processing history yet. Process a CSV file to see it here.")
            
            except Exception as e:
                st.error(f"Error loading history: {e}")
                logging.error(f"Error loading CSV history: {e}")
        else:
            st.info("No processing history yet. Process a CSV file to see it here.")
            
        # Option to clear history
        if history_path.exists():
            st.divider()
            if st.button("🗑️ Clear History", type="secondary"):
                try:
                    history_path.unlink()
                    st.success("History cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing history: {e}")

# Database View Page
elif page == "Database":
    st.header("📊 Summarization Database")
    st.markdown("Consolidated view of all summarized CSV files with advanced search and filtering")
    
    # Add instructions
    with st.expander("ℹ️ How to use this page", expanded=False):
        st.markdown("""
        **Features:**
        - **Multi-file Selection**: Select multiple source files and dates to display simultaneously
        - **Row Selection**: Use checkboxes to select multiple rows to view details
        - **Search & Filter**: Use the search box and column filters to find specific entries
        - **Date Range Filter**: Filter articles by publication date
        - **Export**: Download filtered or complete database as CSV, JSON, or Excel
        
        **Tips:**
        - Use the date range filter to find articles from specific time periods
        - Combine filters for more precise results
        - Export your filtered results in multiple formats
        """)
    
    # Import AgGrid
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
    
    # Load all CSV files
    summarised_dir = Path("summarised_content")
    summarised_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files (excluding history.json)
    csv_files = list(summarised_dir.glob("*.csv"))
    
    # If local folder is empty, try to retrieve from S3
    if not csv_files:
        try:
            from aws_storage import get_storage
            s3_storage = get_storage()
            
            with st.spinner("📥 Checking S3 for summarized data..."):
                # List all CSV files in S3 summarised_content prefix
                s3_csv_files = s3_storage.list_files(prefix="summarised_content/", suffix=".csv")
                
                if s3_csv_files:
                    st.info(f"Found {len(s3_csv_files)} CSV file(s) in S3. Downloading...")
                    
                    # Download each file
                    for s3_key in s3_csv_files:
                        file_name = s3_key.split('/')[-1]
                        local_path = summarised_dir / file_name
                        
                        if s3_storage.download_file(s3_key, str(local_path)):
                            st.success(f"✓ Downloaded: {file_name}")
                        else:
                            st.warning(f"⚠️ Failed to download: {file_name}")
                    
                    # Refresh the csv_files list
                    csv_files = list(summarised_dir.glob("*.csv"))
                    
                    if csv_files:
                        st.success(f"✅ Successfully retrieved {len(csv_files)} file(s) from S3")
                else:
                    st.info("No CSV files found in S3 summarised_content folder")
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve from S3: {str(e)}")
    
    if not csv_files:
        st.info("No CSV files found in summarised_content folder. Process some files in Summarization first!")
        st.stop()
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files", len(csv_files))
    
    # Load and combine all CSVs
    @st.cache_data
    def load_all_csvs(file_list):
        """Load and combine all CSV files"""
        all_data = []
        total_rows = 0
        
        for csv_file in file_list:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.stem  # Add source file name
                # Extract date from filename (format: name_summarized_YYYYMMDD_HHMMSS)
                parts = csv_file.stem.split('_')
                if len(parts) >= 3:
                    df['processed_date'] = parts[-2] + '_' + parts[-1]
                else:
                    df['processed_date'] = 'unknown'
                all_data.append(df)
                total_rows += len(df)
            except Exception as e:
                st.warning(f"Could not load {csv_file.name}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Merge date, pubDate, and publication_date columns into a single 'date' column
            if 'publication_date' in combined_df.columns:
                # Start with publication_date
                combined_df['date'] = combined_df['publication_date']
                # Fill in missing values from pubDate if it exists
                if 'pubDate' in combined_df.columns:
                    combined_df['date'] = combined_df['date'].fillna(combined_df['pubDate'])
                    combined_df = combined_df.drop(columns=['pubDate'])
                combined_df = combined_df.drop(columns=['publication_date'])
            elif 'pubDate' in combined_df.columns and 'date' in combined_df.columns:
                # Prefer pubDate, fallback to date
                combined_df['date'] = combined_df['pubDate'].fillna(combined_df['date'])
                combined_df = combined_df.drop(columns=['pubDate'])
            elif 'pubDate' in combined_df.columns:
                # Rename pubDate to date
                combined_df = combined_df.rename(columns={'pubDate': 'date'})
            
            # Standardize date format to DD MMM YYYY
            if 'date' in combined_df.columns:
                def format_date(date_val):
                    if pd.isna(date_val):
                        return ''
                    try:
                        # Try to parse the date
                        parsed_date = pd.to_datetime(date_val, errors='coerce')
                        if pd.notna(parsed_date):
                            # Format as DD MMM YYYY (e.g., 13 Oct 2025)
                            return parsed_date.strftime('%d %b %Y')
                        return str(date_val)  # Return original if parsing fails
                    except:
                        return str(date_val)
                
                combined_df['date'] = combined_df['date'].apply(format_date)
            
            # Merge tags and classification columns into a single 'categories' column
            if 'tags' in combined_df.columns and 'classification' in combined_df.columns:
                # Combine tags and classification, removing duplicates
                def merge_categories(row):
                    tags = str(row.get('tags', '')).strip() if pd.notna(row.get('tags')) else ''
                    classification = str(row.get('classification', '')).strip() if pd.notna(row.get('classification')) else ''
                    
                    # Split by semicolon and clean
                    all_cats = []
                    if tags:
                        all_cats.extend([t.strip() for t in tags.split(';') if t.strip()])
                    if classification:
                        all_cats.extend([c.strip() for c in classification.split(';') if c.strip()])
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_cats = []
                    for cat in all_cats:
                        if cat.lower() not in seen:
                            seen.add(cat.lower())
                            unique_cats.append(cat)
                    
                    return '; '.join(unique_cats) if unique_cats else ''
                
                combined_df['categories'] = combined_df.apply(merge_categories, axis=1)
                # Drop original columns
                combined_df = combined_df.drop(columns=['tags', 'classification'])
            elif 'tags' in combined_df.columns:
                # Rename tags to categories if only tags exist
                combined_df = combined_df.rename(columns={'tags': 'categories'})
            elif 'classification' in combined_df.columns:
                # Rename classification to categories if only classification exists
                combined_df = combined_df.rename(columns={'classification': 'categories'})
            
            # Sort categories alphabetically within each cell
            if 'categories' in combined_df.columns:
                def sort_categories(cat_string):
                    # Handle NaN, None, or empty values
                    if cat_string is None or (isinstance(cat_string, float) and pd.isna(cat_string)):
                        return ''
                    cat_str = str(cat_string).strip()
                    if not cat_str:
                        return ''
                    # Split by semicolon, strip whitespace, sort alphabetically, rejoin
                    cats = [c.strip() for c in cat_str.split(';') if c.strip()]
                    sorted_cats = sorted(cats, key=lambda x: x.lower())
                    return '; '.join(sorted_cats)
                
                combined_df['categories'] = combined_df['categories'].apply(sort_categories)
            
            # Merge url and link columns into a single 'url' column
            if 'url' in combined_df.columns and 'link' in combined_df.columns:
                # Prefer url, fallback to link
                combined_df['url'] = combined_df['url'].fillna(combined_df['link'])
                combined_df = combined_df.drop(columns=['link'])
            elif 'link' in combined_df.columns:
                # Rename link to url if only link exists
                combined_df = combined_df.rename(columns={'link': 'url'})
            
            # Fill empty 'source' column with filename-based source
            if 'source' in combined_df.columns and 'source_file' in combined_df.columns:
                def extract_source_from_filename(row):
                    # If source already has a value, keep it
                    if pd.notna(row.get('source')) and str(row.get('source')).strip():
                        return row.get('source')
                    
                    # Extract source from filename
                    source_file = str(row.get('source_file', ''))
                    if source_file:
                        # Remove '_summarized_YYYYMMDD_HHMMSS' part
                        parts = source_file.split('_summarized_')
                        if len(parts) > 0:
                            source_name = parts[0]
                            # Convert underscores to spaces and title case
                            # e.g., "canary_media" -> "Canary Media"
                            formatted_source = source_name.replace('_', ' ').title()
                            return formatted_source
                    
                    return ''
                
                combined_df['source'] = combined_df.apply(extract_source_from_filename, axis=1)
            elif 'source_file' in combined_df.columns:
                # Create source column from filename if it doesn't exist
                def create_source_from_filename(source_file):
                    if pd.isna(source_file):
                        return ''
                    source_file = str(source_file)
                    # Remove '_summarized_YYYYMMDD_HHMMSS' part
                    parts = source_file.split('_summarized_')
                    if len(parts) > 0:
                        source_name = parts[0]
                        # Convert underscores to spaces and title case
                        formatted_source = source_name.replace('_', ' ').title()
                        return formatted_source
                    return ''
                
                combined_df['source'] = combined_df['source_file'].apply(create_source_from_filename)
            
            # Rename old 'Start-up' column to new 'URL to start-up(s)' for backward compatibility
            if 'Start-up' in combined_df.columns:
                combined_df = combined_df.rename(columns={'Start-up': 'URL to start-up(s)'})
            
            return combined_df, total_rows
        return None, 0
    
    with st.spinner("Loading all CSV files..."):
        combined_df, total_rows = load_all_csvs(csv_files)
    
    if combined_df is None:
        st.error("Could not load any CSV files!")
        st.stop()
    
    # Deduplicate based on URL
    rows_before_dedup = len(combined_df)
    if 'url' in combined_df.columns:
        # Keep first occurrence, drop duplicates based on URL
        combined_df = combined_df.drop_duplicates(subset=['url'], keep='first')
        rows_after_dedup = len(combined_df)
        duplicates_removed = rows_before_dedup - rows_after_dedup
        
        if duplicates_removed > 0:
            st.info(f"ℹ️ Removed {duplicates_removed} duplicate entries based on URL")
    
    # Reindex starting from 1
    combined_df.index = range(1, len(combined_df) + 1)
    
    # Update metrics
    with col2:
        st.metric("Total Entries", len(combined_df))
    with col3:
        if 'categories' in combined_df.columns:
            # Ensure we're working with a Series, not a DataFrame
            categories_series = combined_df['categories']
            if isinstance(categories_series, pd.DataFrame):
                categories_series = categories_series.iloc[:, 0]  # Take first column if DataFrame
            
            # Count unique categories
            unique_categories = categories_series.astype(str).str.split(';').explode().str.strip().nunique()
            st.metric("Unique Categories", unique_categories)
    
    st.divider()
    
    # Filters and Search
    st.subheader("🔍 Filters & Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Source file filter - changed to multiselect
        source_files_options = sorted(combined_df['source_file'].unique().tolist())
        selected_sources = st.multiselect(
            "Source Files (select one or more)",
            options=source_files_options,
            default=source_files_options,  # All selected by default
            help="Select one or more source files to display"
        )
    
    with col2:
        # Date range
        if 'processed_date' in combined_df.columns:
            unique_dates = sorted(combined_df['processed_date'].unique())
            if len(unique_dates) > 1:
                selected_dates = st.multiselect(
                    "Processed Dates (select one or more)",
                    options=unique_dates,
                    default=unique_dates,  # All selected by default
                    help="Select one or more dates to display"
                )
            else:
                selected_dates = unique_dates
        else:
            selected_dates = []
    
    # Publication Date Range Filter
    if 'date' in combined_df.columns:
        st.markdown("**📅 Publication Date Range**")
        
        # Parse dates for filtering (use a temporary Series, don't modify combined_df)
        parsed_dates = pd.to_datetime(combined_df['date'], format='%d %b %Y', errors='coerce')
        
        # Get min and max dates (exclude NaT values)
        valid_dates = parsed_dates.dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date = st.date_input(
                    "From Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    help="Select the start date for filtering articles"
                )
            with col_date2:
                end_date = st.date_input(
                    "To Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    help="Select the end date for filtering articles"
                )
            
            # Validate date range
            if start_date > end_date:
                st.warning("⚠️ Start date must be before or equal to end date")
        else:
            start_date = None
            end_date = None
            st.info("ℹ️ No valid publication dates found in the data")
    else:
        start_date = None
        end_date = None
    
    # Text search with multiple keywords support
    search_query = st.text_input(
        "🔎 Search in database", 
        placeholder="Enter keywords (separate with commas)...",
        help="Enter one or more keywords. Separate multiple keywords with commas. Any keyword will match (OR logic)."
    )
    
    # Apply filters
    filtered_df = combined_df.copy()
    
    # Filter by selected source files
    if selected_sources:
        filtered_df = filtered_df[filtered_df['source_file'].isin(selected_sources)]
    else:
        # If nothing selected, show nothing
        filtered_df = filtered_df.iloc[0:0]
    
    # Filter by selected dates
    if 'processed_date' in combined_df.columns and selected_dates:
        filtered_df = filtered_df[filtered_df['processed_date'].isin(selected_dates)]
    elif 'processed_date' in combined_df.columns and not selected_dates:
        # If nothing selected, show nothing
        filtered_df = filtered_df.iloc[0:0]
    
    # Filter by publication date range
    if 'date' in combined_df.columns and start_date and end_date:
        # Parse dates from the filtered dataframe for comparison
        filtered_parsed_dates = pd.to_datetime(filtered_df['date'], format='%d %b %Y', errors='coerce')
        
        # Convert start_date and end_date to datetime for comparison
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Include entire end date
        
        # Filter rows where parsed date is within range
        date_mask = (filtered_parsed_dates >= start_datetime) & (filtered_parsed_dates <= end_datetime)
        filtered_df = filtered_df[date_mask]
    
    # Determine which columns will be displayed (exclude hidden columns)
    exclude_cols = ['source_file', 'processed_date', 'content', 'file', 'file_location', 
                   'filename', 'file_name', 'folder', 'filepath', 'Source', 'source', 'author']
    display_columns = [col for col in filtered_df.columns if col not in exclude_cols]
    
    # Multi-keyword search (OR logic)
    if search_query:
        # Parse multiple keywords - split by comma only
        keywords = [k.strip() for k in search_query.split(',') if k.strip()]
        
        if not keywords:
            keywords = [search_query.strip()]
        
        # Apply OR logic - any keyword may match
        final_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)

        for keyword in keywords:
            # Create a mask for this keyword across all display columns
            keyword_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            for col in display_columns:
                keyword_mask |= filtered_df[col].astype(str).str.contains(keyword, case=False, na=False, regex=False)
            # OR with the final mask (any keyword matches)
            final_mask |= keyword_mask

        filtered_df = filtered_df[final_mask]

        # Show which keywords are being searched
        if len(keywords) > 1:
            st.caption(f"🔍 Searching for any of: {', '.join([f'**{k}**' for k in keywords])} (any match)")
        else:
            st.caption(f"🔍 Searching for: **{keywords[0]}**")
    
    st.info(f"Showing {len(filtered_df)} of {len(combined_df)} entries")
    
    st.divider()
    
    # Display results with AgGrid
    st.subheader("📋 Results")
    
    if len(filtered_df) == 0:
        st.warning("No entries match your filters.")
    else:
        # Prepare dataframe for display - exclude specified columns
        exclude_cols = ['source_file', 'processed_date', 'content', 'file', 'file_location', 
                       'filename', 'file_name', 'folder', 'filepath', 'Source', 'source', 'author']
        display_df = filtered_df.drop(columns=[col for col in exclude_cols if col in filtered_df.columns])
        
        # Define desired column order
        desired_order = [
            'title',
            'Indicator',  # This is the summary/indicator column
            'publication_date',  # publication date (matches CSV column name)
            'categories',
            'Dimension',  # matches CSV column name
            'Tech',  # matches CSV column name
            'TRL',  # matches CSV column name
            'URL to start-up(s)',  # matches CSV column name
            'url'  # URL as the last column
        ]
        
        # Reorder columns according to desired order - only include columns that exist
        ordered_cols = [col for col in desired_order if col in display_df.columns]
        
        # Reorder the dataframe (don't include remaining columns - strict ordering)
        display_df = display_df[ordered_cols]
        
        # Reset index to show row numbers starting from 1
        display_df = display_df.reset_index(drop=True)
        
        # Convert categories from semicolon-separated to comma-separated for better display
        if 'categories' in display_df.columns:
            display_df['categories'] = display_df['categories'].apply(
                lambda x: str(x).replace(';', ',') if pd.notna(x) else ''
            )
        
        st.info(f"📊 Showing {len(display_df)} entries | Use search and filters in the table below")
        
        # Add custom CSS for darker table borders
        st.markdown("""
        <style>
        /* Black border around each cell */
        .ag-theme-streamlit-dark .ag-cell,
        .ag-theme-streamlit .ag-cell {
            border-right: 1px solid black !important;
            border-bottom: 1px solid black !important;
        }
        
        /* Black border around header cells */
        .ag-theme-streamlit-dark .ag-header-cell,
        .ag-theme-streamlit .ag-header-cell {
            border-right: 1px solid black !important;
            border-bottom: 1px solid black !important;
        }
        
        /* Black border around the entire grid */
        .ag-theme-streamlit-dark .ag-root-wrapper,
        .ag-theme-streamlit .ag-root-wrapper {
            border: 2px solid black !important;
            border-radius: 4px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Configure AgGrid options
        gb = GridOptionsBuilder.from_dataframe(display_df)
        
        # Enable features
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        gb.configure_side_bar(filters_panel=True, columns_panel=True)
        gb.configure_default_column(
            filterable=True,
            sortable=True,
            resizable=True,
            wrapText=True,
            autoHeight=True,
            enableCellTextSelection=True,
            editable=False  # Disable editing for all columns
        )
        # Configure specific columns
        if 'url' in display_df.columns:
            gb.configure_column(
                'url',
                headerName='URL',
                width=100,
                wrapText=True,
                autoHeight=True,
                cellStyle={'word-break': 'break-all', 'white-space': 'normal'},
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'Indicator' in display_df.columns:
            gb.configure_column(
                'Indicator', 
                headerName='Summary/Indicator', 
                width=150, 
                wrapText=True, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'title' in display_df.columns:
            gb.configure_column(
                'title', 
                headerName='Title', 
                width=100, 
                wrapText=True, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'publication_date' in display_df.columns:
            gb.configure_column(
                'publication_date', 
                headerName='Date', 
                width=100, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'categories' in display_df.columns:
            gb.configure_column(
                'categories', 
                headerName='Categories', 
                width=120, 
                wrapText=True, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'Dimension' in display_df.columns:
            gb.configure_column(
                'Dimension', 
                headerName='Dimension', 
                width=100, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'Tech' in display_df.columns:
            gb.configure_column(
                'Tech', 
                headerName='Technology', 
                width=100, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'TRL' in display_df.columns:
            gb.configure_column(
                'TRL', 
                headerName='TRL Level', 
                width=80, 
                enableCellTextSelection=True,
                editable=False
            )
        
        if 'Start-up' in display_df.columns:
            gb.configure_column(
                'Start-up', 
                headerName='URL to Start-up', 
                width=120, 
                wrapText=True, 
                enableCellTextSelection=True,
                editable=False
            )
        
        # Configure selection
        gb.configure_selection(selection_mode='multiple', use_checkbox=True)
        
        # Enable text selection (read-only mode)
        gb.configure_grid_options(
            enableCellTextSelection=True, 
            ensureDomOrder=True,
            suppressRowClickSelection=False  # Allow row selection on click
        )
        
        gridOptions = gb.build()
        
        # Configure default column options (read-only)
        gridOptions['defaultColDef']['wrapText'] = True
        gridOptions['defaultColDef']['autoHeight'] = True
        gridOptions['defaultColDef']['enableCellTextSelection'] = True
        gridOptions['defaultColDef']['editable'] = False
        
        # Display AgGrid
        grid_response = AgGrid(
            display_df,
            gridOptions=gridOptions,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.NO_UPDATE,  # Read-only mode
            fit_columns_on_grid_load=False,
            theme='streamlit',
            width=800,
            height=800,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            reload_data=False
        )
        
        # Show selection info if any rows are selected
        selected_rows = grid_response['selected_rows']
        if selected_rows is not None and len(selected_rows) > 0:
            st.success(f"✅ Selected {len(selected_rows)} row(s)")
            with st.expander("View Selected Rows"):
                st.dataframe(pd.DataFrame(selected_rows), use_container_width=True)
    
    st.divider()
    
    # Export options
    st.subheader("📥 Export Database")
    
    # Define column order for exports (same as display)
    desired_order = [
        'title',
        'Indicator',  # summary/indicator column
        'publication_date',  # publication date (matches CSV column name)
        'categories',
        'Dimension',  # matches CSV column name
        'Tech',  # matches CSV column name
        'TRL',  # matches CSV column name
        'URL to start-up(s)',  # matches CSV column name
        'url'  # URL as the last column
    ]
    
    # Helper function to reorder columns
    def reorder_export_columns(df):
        """Reorder dataframe columns according to desired order, excluding unwanted columns"""
        # Exclude columns that shouldn't be in exports
        exclude_cols = ['source_file', 'processed_date', 'content', 'file', 'file_location', 
                       'filename', 'file_name', 'folder', 'filepath', 'Source', 'source', 'author']
        df_clean = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        
        # Get columns that exist in desired order - strict ordering, no remaining columns
        ordered_cols = [col for col in desired_order if col in df_clean.columns]
        return df_clean[ordered_cols]
    
    # CSV & JSON exports
    st.markdown("**CSV & JSON Formats**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Export filtered results as CSV
        if len(filtered_df) > 0:
            filtered_export = reorder_export_columns(filtered_df.copy())
            csv_export = filtered_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Filtered CSV",
                data=csv_export,
                file_name=f"filtered_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        # Export filtered results as JSON
        if len(filtered_df) > 0:
            filtered_export = reorder_export_columns(filtered_df.copy())
            # Convert DataFrame to JSON (orient='records' creates array of objects)
            json_export = filtered_export.to_json(orient='records', indent=2, force_ascii=False).encode('utf-8')
            st.download_button(
                label="📋 Filtered JSON",
                data=json_export,
                file_name=f"filtered_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col3:
        # Export all data as CSV
        all_export = reorder_export_columns(combined_df.copy())
        all_csv = all_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Complete CSV",
            data=all_csv,
            file_name=f"complete_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col4:
        # Export all data as JSON
        all_export = reorder_export_columns(combined_df.copy())
        all_json = all_export.to_json(orient='records', indent=2, force_ascii=False).encode('utf-8')
        st.download_button(
            label="📋 Complete JSON",
            data=all_json,
            file_name=f"complete_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Excel export (separate row)
    st.markdown("**Excel Format**")
    col_excel = st.columns(1)[0]
    
    # Export to Excel (if openpyxl is installed)
    try:
        from io import BytesIO
        output = BytesIO()
        
        # Prepare dataframes for export with reordered columns
        export_all_df = reorder_export_columns(combined_df.copy())
        if 'categories' in export_all_df.columns:
            export_all_df['categories'] = export_all_df['categories'].apply(
                lambda x: '; '.join(x) if isinstance(x, list) else x
            )
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_all_df.to_excel(writer, index=False, sheet_name='All Data')
            if len(filtered_df) > 0 and len(filtered_df) < len(combined_df):
                export_filtered_df = reorder_export_columns(filtered_df.copy())
                if 'categories' in export_filtered_df.columns:
                    export_filtered_df['categories'] = export_filtered_df['categories'].apply(
                        lambda x: '; '.join(x) if isinstance(x, list) else x
                    )
                export_filtered_df.to_excel(writer, index=False, sheet_name='Filtered')
        
        st.download_button(
            label="📊 Download as Excel (All Sheets)",
            data=output.getvalue(),
            file_name=f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    except ImportError:
        st.caption("Install openpyxl for Excel export")


elif page == "Chatbot":
    st.header("🤖 AI Research Assistant")
    st.markdown("Chat with your technology intelligence database using AI-powered keyword search and metadata filtering")
    
    # Initialize session state for chat
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_processor' not in st.session_state:
        st.session_state.chat_processor = None
    if 'embedding_indexes' not in st.session_state:
        st.session_state.embedding_indexes = {}
    if 'embeddings_built' not in st.session_state:
        st.session_state.embeddings_built = False
    
    # Check for available data
    summarised_dir = Path("summarised_content")
    json_files = list(summarised_dir.glob("*.json")) if summarised_dir.exists() else []
    
    # Filter out history.json
    json_files = [f for f in json_files if f.name != "history.json"]
    
    if not json_files:
        st.warning("⚠️ No JSON files found in `summarised_content/` folder. Please process some files in the Summarization page first.")
        st.info("💡 The chatbot needs processed JSON data to answer your questions.")
        st.stop()
    
    # Helper function for source name extraction
    def extract_source_name_from_filename(filename: str) -> str:
        """
        Extract and format source name from filename.
        Uses same logic as JSONEmbeddingProcessor.extract_source_name()
        
        Args:
            filename: Filename like "techcrunch_com_20251127.json"
            
        Returns:
            Formatted source name like "Techcrunch"
        """
        from pathlib import Path
        
        # Remove .json extension and split by underscores
        stem = Path(filename).stem
        parts = stem.split('_')
        
        # Take first part and capitalize it
        if parts:
            source = parts[0].replace('com', '').replace('org', '').replace('net', '')
            # Capitalize first letter
            return source.capitalize()
        
        return stem
    
    def extract_date_from_filename(filename: str) -> str:
        """
        Extract and format date from filename.
        
        Args:
            filename: Filename like "techcrunch_com_20251127.json"
            
        Returns:
            Formatted date like "2025-11-27" or empty string if not found
        """
        from pathlib import Path
        import re
        
        # Remove .json extension
        stem = Path(filename).stem
        
        # Look for date pattern YYYYMMDD
        date_match = re.search(r'_(\d{8})$', stem)
        if date_match:
            date_str = date_match.group(1)
            # Format as YYYY-MM-DD
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        return ""
    
    # Get available JSON files with metadata
    available_files = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data:  # Check if file has content
                    # Extract source name and date from filename
                    filename = json_file.name
                    source_name = extract_source_name_from_filename(filename)
                    date_str = extract_date_from_filename(filename)
                    
                    available_files.append({
                        'filename': filename,
                        'source_name': source_name,
                        'date': date_str,
                        'path': json_file,
                        'count': len(data)
                    })
        except Exception as e:
            st.warning(f"⚠️ Could not read {json_file.name}: {e}")
    
    if not available_files:
        st.error("No valid JSON files found with content.")
        st.stop()
    
    # Create options for multiselect with compact format for sidebar display
    # Format: <Source> <date>
    file_options = []
    file_options_dict = {}
    
    for f in available_files:
        # Compact format: "Techcrunch 11/28"
        date_parts = f['date'].split('-') if f['date'] else ['', '', '']
        if len(date_parts) == 3:
            short_date = f"{date_parts[1]}/{date_parts[2]}"  # MM/DD
        else:
            short_date = ""
        
        compact_label = f"{f['source_name']} {short_date}"
        file_options.append(compact_label)
        file_options_dict[compact_label] = f
    
    # Default to all files if none selected, or filter out invalid cached selections
    if 'selected_files' not in st.session_state:
        st.session_state.selected_files = file_options
    else:
        # Filter out any cached selections that are no longer valid (e.g., history.json was removed)
        st.session_state.selected_files = [opt for opt in st.session_state.selected_files if opt in file_options]
        # If all selections were invalid, default to all files
        if not st.session_state.selected_files:
            st.session_state.selected_files = file_options
    
    # Helper functions for S3 embeddings management
    def upload_embeddings_to_s3(processor, embedding_indexes):
        """Upload all embedding indexes to S3"""
        success_count = 0
        for source_name, embedding_index in embedding_indexes.items():
            try:
                if processor.upload_index_to_s3(embedding_index):
                    success_count += 1
            except Exception as e:
                st.error(f"Failed to upload {source_name}: {e}")
        return success_count
    
    def download_embeddings_from_s3(processor):
        """Download all embedding indexes from S3"""
        try:
            s3_indexes = processor.list_s3_indexes()
            embedding_indexes = {}
            
            for index_name in s3_indexes:
                try:
                    embedding_index = processor.download_index_from_s3(index_name)
                    if embedding_index:
                        # Extract source name from index name
                        source_name = index_name.replace('_embeddings', '')
                        embedding_indexes[source_name] = embedding_index
                except Exception as e:
                    st.warning(f"Failed to download {index_name}: {e}")
                    continue
            
            return embedding_indexes
        except Exception as e:
            st.error(f"Failed to download from S3: {e}")
            return {}
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### � Select Data Sources")
        
        # File selection widget in sidebar
        selected_options = st.multiselect(
            "Choose data sources:",
            options=file_options,
            default=st.session_state.selected_files,
            help="Format: Source MM/DD (article count). Example: Techcrunch 11/28 (50)",
            label_visibility="collapsed"
        )
        
        # Update session state
        st.session_state.selected_files = selected_options
        
        # Get selected file objects
        selected_files = [file_options_dict[opt] for opt in selected_options] if selected_options else []
        
        if not selected_files:
            st.warning("Please select at least one data source.")
            st.stop()
        
        st.divider()
        st.markdown("### �🗄️ Database Status")
        
        # Check if embeddings exist in S3 for selected files
        s3_available_indexes = []
        s3_index_mapping = {}  # Map source_name to full index name with date
        try:
            from aws_storage import get_storage
            s3_storage = get_storage()
            s3_files = s3_storage.list_files(prefix="rag_embeddings/", suffix=".pkl")
            
            # Parse S3 filenames to extract source name and create mapping
            for s3_file in s3_files:
                # Extract index name from path: "rag_embeddings/Techcrunch_20251128_embeddings.pkl"
                index_name = s3_file.replace("rag_embeddings/", "").replace(".pkl", "")
                
                # Extract source name (first part before date)
                import re
                match = re.match(r'^([A-Za-z\-]+)', index_name)
                if match:
                    source_name = match.group(1)
                    s3_available_indexes.append(source_name)
                    s3_index_mapping[source_name] = index_name
        except:
            pass
        
        if s3_available_indexes:
            available_selected = [f for f in selected_files if f['source_name'] in s3_available_indexes]
            if available_selected:
                st.success(f"☁️ {len(available_selected)}/{len(selected_files)} selected sources available in S3")
        
        if st.session_state.embeddings_built:
            st.success(f"✅ Data indexed")
            
            # Show document count for loaded sources
            if st.session_state.embedding_indexes:
                loaded_sources = list(st.session_state.embedding_indexes.keys())
                total_docs = sum(idx.num_documents for idx in st.session_state.embedding_indexes.values())
                total_chunks = sum(idx.num_chunks for idx in st.session_state.embedding_indexes.values())
                st.metric("Loaded Sources", len(loaded_sources))
                st.metric("Documents", total_docs)
                st.metric("Chunks", total_chunks)
                
                # Show which sources are loaded
                with st.expander("📋 Loaded Sources", expanded=False):
                    for source in loaded_sources:
                        st.write(f"• {source}")
        else:
            st.info(f"📂 Selected {len(selected_files)} source(s)")
        
        st.divider()
        
        # Load from S3 button for selected files
        available_in_s3 = [f for f in selected_files if f['source_name'] in s3_available_indexes]
        if available_in_s3:
            if st.button("☁️ Load Selected from S3", use_container_width=True, type="secondary"):
                with st.spinner("Downloading selected embeddings from S3..."):
                    try:
                        from embeddings_processor import JSONEmbeddingProcessor
                        processor = JSONEmbeddingProcessor()
                        
                        embedding_indexes = {}
                        loaded_count = 0
                        
                        for file_info in selected_files:
                            source_name = file_info['source_name']
                            if source_name in s3_index_mapping:
                                try:
                                    # Use the full index name with date from the mapping
                                    full_index_name = s3_index_mapping[source_name]
                                    embedding_index = processor.download_index_from_s3(full_index_name)
                                    if embedding_index:
                                        # Store with a friendly key (source_name for backward compatibility)
                                        embedding_indexes[source_name] = embedding_index
                                        loaded_count += 1
                                except Exception as e:
                                    st.warning(f"Failed to download {source_name}: {e}")
                        
                        if embedding_indexes:
                            st.session_state.chat_processor = processor
                            st.session_state.embedding_indexes = embedding_indexes
                            st.session_state.embeddings_built = True
                            total_docs = sum(idx.num_documents for idx in embedding_indexes.values())
                            total_chunks = sum(idx.num_chunks for idx in embedding_indexes.values())
                            st.success(f"✅ Loaded {loaded_count}/{len(selected_files)} sources from S3! {total_docs} documents ({total_chunks} chunks)")
                            st.rerun()
                        else:
                            st.error("Failed to load any embeddings from S3")
                    except Exception as e:
                        st.error(f"Error loading from S3: {e}")
        
        # Build/Rebuild embeddings for selected files
        if st.button("🔄 Build Selected Embeddings", use_container_width=True, type="primary"):
            with st.spinner("Building embeddings for selected files..."):
                try:
                    from embeddings_processor import JSONEmbeddingProcessor
                    
                    # Initialize processor with same parameters as rebuild_embeddings.py
                    processor = JSONEmbeddingProcessor(
                        enable_s3_sync=True,
                        local_storage_dir="./embeddings_storage"
                    )
                    
                    embedding_indexes = {}
                    
                    for file_info in selected_files:
                        try:
                            st.info(f"Processing {file_info['filename']}...")
                            
                            # Process the entire JSON file using the processor
                            embedding_index = processor.process_json_file(
                                file_info['path'],
                                progress_callback=lambda msg, current, total: st.info(f"{msg} [{current}/{total}]")
                            )
                            
                            st.info(f"✓ Created embedding index: {embedding_index.num_documents} docs, {embedding_index.num_chunks} chunks")
                            
                            # Save locally
                            local_path = processor.save_index_locally(embedding_index)
                            st.info(f"✓ Saved locally: {local_path}")
                            
                            # Upload to S3
                            if processor.upload_index_to_s3(embedding_index):
                                st.info(f"✓ Uploaded to S3")
                            else:
                                st.warning(f"⚠️ S3 upload failed for {file_info['filename']}")
                            
                            # Store in session state
                            embedding_indexes[file_info['source_name']] = embedding_index
                            
                        except Exception as e:
                            st.warning(f"Error processing {file_info['filename']}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    if not embedding_indexes:
                        st.error("❌ No embeddings were built successfully")
                        st.stop()
                    
                    # Store in session state
                    st.session_state.chat_processor = processor
                    st.session_state.embedding_indexes = embedding_indexes
                    st.session_state.embeddings_built = True
                    
                    total_docs = sum(idx.num_documents for idx in embedding_indexes.values())
                    total_chunks = sum(idx.num_chunks for idx in embedding_indexes.values())
                    
                    st.success(f"✅ Embeddings built successfully! {total_docs} documents processed into {total_chunks} chunks across {len(embedding_indexes)} sources.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error building embeddings: {e}")
                    import traceback
                    traceback.print_exc()
                    st.stop()
        
        if st.session_state.embeddings_built:
            st.caption("💡 Rebuild if you've added new data")
        
        st.divider()
        
        # Retrieval settings
        st.markdown("### ⚙️ Retrieval Settings")
        num_results = st.slider(
            "Number of documents to retrieve",
            min_value=5,
            max_value=30,
            value=10,
            step=1,
            help="Adjust how many relevant documents are retrieved and used to generate the response"
        )
        
        st.divider()
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    
    # Main chat interface
    if not st.session_state.embeddings_built:
        st.info("👆 Click **Load from S3** or **Build/Rebuild Embeddings** in the sidebar to start chatting")
        st.stop()
    
    # Load processor if not in session state
    if st.session_state.chat_processor is None or not st.session_state.embedding_indexes:
        try:
            processor = JSONEmbeddingProcessor()
            # Load all indexes from local storage
            local_indexes = processor.list_local_indexes()
            embedding_indexes = {}
            for index_name in local_indexes:
                try:
                    embedding_index = processor.load_index_locally(index_name)
                    source_name = index_name.replace('_embeddings', '')
                    embedding_indexes[source_name] = embedding_index
                except Exception as e:
                    st.warning(f"Failed to load local index {index_name}: {e}")
            
            if embedding_indexes:
                st.session_state.chat_processor = processor
                st.session_state.embedding_indexes = embedding_indexes
            else:
                st.error("No embedding indexes found. Please build embeddings first.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        # Use URL as title if title is None or empty
                        title = source.get('title', 'Untitled')
                        if not title or title.lower() in ['none', 'untitled', 'n/a']:
                            title = source.get('url', 'Untitled')
                        
                        st.markdown(f"**{i}. [{title}]({source['url']})**")
                        st.caption(f"📅 {source.get('publication_date', 'N/A')} | 🏷️ {source.get('categories', 'N/A')}")
                        if source.get('source'):
                            st.caption(f"📺 Source: {source['source']}")
                        if source.get('tech'):
                            st.caption(f"💡 Tech: {source['tech']} | TRL: {source.get('trl', 'N/A')} | Dim: {source.get('dimension', 'N/A')}")
                        
                        # Display Indicator/Summary
                        indicator = source.get('indicator', '')
                        if indicator and indicator != 'N/A':
                            st.markdown(f"**📝 Summary:** {indicator}")
                        
                        # Display URL to start-up(s)
                        startup = source.get('startup', '')
                        if startup and startup != 'N/A' and startup:
                            st.caption(f"🚀 URL to start-up(s): {startup}")
                        
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Chat with the data (try: 'latest AI articles', 'TechCrunch hydrogen news', 'TRL 8 technologies')"):
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    processor = st.session_state.chat_processor
                    
                    # Parse query for field-based filtering
                    def parse_query_for_filters(query):
                        """Parse natural language query to extract metadata filters and date ranges"""
                        filters = {}
                        date_filters = {}
                        query_lower = query.lower()
                        
                        # Import datetime for date parsing
                        from datetime import datetime, timedelta
                        import re
                        
                        # Source filters (e.g., "from TechCrunch", "TechCrunch articles")
                        sources = ["techcrunch", "carbonherald", "hydrogen-central", "interestingengineering", "canarymedia", "bioenergy-news", "pv-magazine"]
                        for source in sources:
                            if source in query_lower or f"from {source}" in query_lower:
                                # Keep the original source name (preserve hyphens) for matching
                                filters["source"] = source
                                break
                        
                        # TRL filters (Technology Readiness Level)
                        # First check for research stage keywords that map to TRL ranges
                        trl_stage_patterns = {
                            r'\b(basic|fundamental)\s+research\b': '1-3',
                            r'\bproof\s+of\s+concept\b': '1-3',
                            r'\blaboratory\s+(research|validation|testing|demonstration)\b': '1-5',
                            r'\blab\s+(testing|validation|scale)\b': '4-5',
                            r'\bprototype\s+(testing|development)\b': '6-7',
                            r'\bpilot\s+(project|testing|scale)\b': '6-7',
                            r'\b(commercial|deployed|operational)\s+(system|deployment|scale)\b': '8-9',
                            r'\bmarket\s+ready\b': '8-9'
                        }
                        
                        for pattern, trl_range in trl_stage_patterns.items():
                            if re.search(pattern, query_lower):
                                filters["trl_range"] = trl_range
                                break
                        
                        # Then check for explicit TRL numbers (only if no stage pattern matched)
                        if "trl_range" not in filters and "trl" in query_lower:
                            for i in range(1, 10):
                                if f"trl {i}" in query_lower or f"trl{i}" in query_lower:
                                    filters["trl"] = str(i)
                                    break
                        
                        # Enhanced date parsing
                        current_year = datetime.now().year
                        current_month = datetime.now().month
                        
                        # Specific month/year patterns (e.g., "Nov 2025", "November 2025", "in 2025")
                        month_patterns = {
                            r'(january|jan)\s+(\d{4})': 1,
                            r'(february|feb)\s+(\d{4})': 2,
                            r'(march|mar)\s+(\d{4})': 3,
                            r'(april|apr)\s+(\d{4})': 4,
                            r'(may)\s+(\d{4})': 5,
                            r'(june|jun)\s+(\d{4})': 6,
                            r'(july|jul)\s+(\d{4})': 7,
                            r'(august|aug)\s+(\d{4})': 8,
                            r'(september|sept|sep)\s+(\d{4})': 9,
                            r'(october|oct)\s+(\d{4})': 10,
                            r'(november|nov)\s+(\d{4})': 11,
                            r'(december|dec)\s+(\d{4})': 12
                        }
                        
                        for pattern, month_num in month_patterns.items():
                            match = re.search(pattern, query_lower)
                            if match:
                                year = int(match.group(2))
                                # Date range for that month
                                date_filters['year'] = year
                                date_filters['month'] = month_num
                                date_filters['date_after'] = f"{year}-{month_num:02d}-01"
                                # Calculate last day of month
                                if month_num == 12:
                                    next_month = 1
                                    next_year = year + 1
                                else:
                                    next_month = month_num + 1
                                    next_year = year
                                date_filters['date_before'] = f"{next_year}-{next_month:02d}-01"
                                break
                        
                        # Year only patterns (e.g., "in 2025", "from 2025")
                        if not date_filters:
                            year_match = re.search(r'\b(in|from|during)\s+(\d{4})\b', query_lower)
                            if year_match:
                                year = int(year_match.group(2))
                                date_filters['year'] = year
                                date_filters['date_after'] = f"{year}-01-01"
                                date_filters['date_before'] = f"{year + 1}-01-01"
                        
                        # Relative date filters
                        if not date_filters:
                            # Pattern for "last X months/weeks/days"
                            last_period_match = re.search(r'last\s+(\d+)\s+(month|week|day)s?', query_lower)
                            if last_period_match:
                                num = int(last_period_match.group(1))
                                period = last_period_match.group(2)
                                
                                if period == "month":
                                    days_ago = num * 30
                                elif period == "week":
                                    days_ago = num * 7
                                else:  # day
                                    days_ago = num
                                
                                date_after = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
                                date_filters['date_after'] = date_after
                            elif "latest" in query_lower or "recent" in query_lower or "new" in query_lower:
                                # Get recent articles (last 2 months / 60 days)
                                two_months_ago = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
                                date_filters['date_after'] = two_months_ago
                            elif "this month" in query_lower:
                                date_filters['date_after'] = f"{current_year}-{current_month:02d}-01"
                            elif "last month" in query_lower:
                                last_month = current_month - 1 if current_month > 1 else 12
                                last_year = current_year if current_month > 1 else current_year - 1
                                date_filters['date_after'] = f"{last_year}-{last_month:02d}-01"
                                date_filters['date_before'] = f"{current_year}-{current_month:02d}-01"
                        
                        return filters, date_filters
                    
                    # Parse query for metadata filters
                    metadata_filters, date_filters = parse_query_for_filters(prompt)
                    
                    # Extract source filter separately (since it's not in embeddings metadata)
                    source_filter = metadata_filters.pop("source", None) if metadata_filters else None
                    
                    # Show debug info about detected filters
                    if metadata_filters or date_filters or source_filter:
                        filter_info = []
                        if source_filter:
                            filter_info.append(f"source={source_filter}")
                        if metadata_filters:
                            # Handle TRL range separately for better display
                            display_filters = {}
                            for k, v in metadata_filters.items():
                                if k == 'trl_range':
                                    filter_info.append(f"TRL range: {v}")
                                else:
                                    display_filters[k] = v
                            
                            if display_filters:
                                filter_info.append(f"{', '.join([f'{k}={v}' for k, v in display_filters.items()])}")
                        if date_filters:
                            if 'year' in date_filters and 'month' in date_filters:
                                month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                                filter_info.append(f"Date filter: {month_names[date_filters['month']]} {date_filters['year']}")
                            elif 'year' in date_filters:
                                filter_info.append(f"Date filter: Year {date_filters['year']}")
                            elif 'date_after' in date_filters:
                                filter_info.append(f"Date filter: After {date_filters['date_after']}")
                        
                        st.info("🔍 Metadata filters: " + " | ".join(filter_info))
                    
                    # Query across all embedding indexes
                    all_results = []
                    processor = st.session_state.chat_processor
                    embedding_indexes = st.session_state.embedding_indexes
                    
                    # Get more results initially if we need to filter by date or source
                    # Use num_results * 2 for filtering scenarios to ensure we have enough after filtering
                    initial_top_k = num_results * 2 if (date_filters or source_filter) else num_results
                    
                    # Query all sources (we'll filter by URL domain later)
                    sources_to_query = list(embedding_indexes.items())
                    
                    for source_name, embedding_index in sources_to_query:
                        try:
                            results = processor.query_similar(
                                embedding_index,
                                query_text=prompt,
                                top_k=initial_top_k,
                                filter_metadata=metadata_filters if metadata_filters else None
                            )
                            
                            # Add source information to results
                            for result in results:
                                result['metadata']['source'] = source_name
                            
                            all_results.extend(results)
                        except Exception as e:
                            st.warning(f"Error querying {source_name}: {e}")
                            continue
                    
                    # Apply source filtering by URL domain if specified
                    if source_filter and all_results:
                        filtered_results = []
                        debug_info = {"before_filter": len(all_results), "after_filter": 0, "filtered_out": 0}
                        
                        for result in all_results:
                            url = result['metadata'].get('url', '').lower()
                            # Check if the source filter appears in the URL domain
                            if source_filter in url:
                                filtered_results.append(result)
                            else:
                                debug_info["filtered_out"] += 1
                        
                        debug_info["after_filter"] = len(filtered_results)
                        all_results = filtered_results
                        
                        # Show debug info
                        if debug_info["after_filter"] > 0:
                            st.info(f"🔍 Source filter applied: {debug_info['before_filter']} results → {debug_info['after_filter']} results from '{source_filter}' domain (removed {debug_info['filtered_out']} from other sources)")
                        else:
                            st.warning(f"⚠️ No results found with '{source_filter}' in the URL domain.")
                    
                    # Apply date filtering on results if specified
                    if date_filters and all_results:
                        from datetime import datetime
                        
                        filtered_results = []
                        debug_info = {"before_filter": len(all_results), "after_filter": 0, "filtered_out": 0}
                        
                        for result in all_results:
                            pub_date = result['metadata'].get('publication_date', '')
                            if not pub_date:
                                continue
                            
                            try:
                                # Parse publication date (handle various formats)
                                if 'T' in pub_date:
                                    pub_date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                else:
                                    pub_date_obj = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                                
                                # Check date filters
                                if 'date_after' in date_filters:
                                    after_date = datetime.strptime(date_filters['date_after'], '%Y-%m-%d')
                                    if pub_date_obj < after_date:
                                        debug_info["filtered_out"] += 1
                                        continue
                                
                                if 'date_before' in date_filters:
                                    before_date = datetime.strptime(date_filters['date_before'], '%Y-%m-%d')
                                    if pub_date_obj >= before_date:
                                        debug_info["filtered_out"] += 1
                                        continue
                                
                                filtered_results.append(result)
                                
                            except (ValueError, AttributeError) as e:
                                # Skip results with invalid dates
                                continue
                        
                        debug_info["after_filter"] = len(filtered_results)
                        all_results = filtered_results
                        
                        # Show debug info
                        if 'date_after' in date_filters:
                            st.info(f"📅 Date filter applied: {debug_info['before_filter']} results → {debug_info['after_filter']} results after filtering (removed {debug_info['filtered_out']} old articles)")
                    
                    # Apply TRL range filtering if specified
                    if metadata_filters and 'trl_range' in metadata_filters and all_results:
                        trl_range = metadata_filters.pop('trl_range')
                        filtered_results = []
                        debug_info = {"before_filter": len(all_results), "after_filter": 0, "filtered_out": 0}
                        
                        # Parse TRL range (e.g., "1-3", "4-5", "6-7", "8-9")
                        trl_min, trl_max = map(int, trl_range.split('-'))
                        
                        for result in all_results:
                            trl_str = result['metadata'].get('trl', '')
                            if not trl_str:
                                continue
                            
                            try:
                                # Parse TRL value (handle both int and float)
                                trl_value = int(float(str(trl_str).strip()))
                                
                                # Check if TRL is within range
                                if trl_min <= trl_value <= trl_max:
                                    filtered_results.append(result)
                                else:
                                    debug_info["filtered_out"] += 1
                                    
                            except (ValueError, AttributeError):
                                # Skip results with invalid TRL values
                                continue
                        
                        debug_info["after_filter"] = len(filtered_results)
                        all_results = filtered_results
                        
                        # Show debug info
                        st.info(f"🔬 TRL filter applied (TRL {trl_range}): {debug_info['before_filter']} results → {debug_info['after_filter']} results after filtering (removed {debug_info['filtered_out']} articles)")
                    
                    # Sort all results by score and take top N (based on slider)
                    all_results.sort(key=lambda x: x['score'], reverse=True)
                    results = all_results[:num_results]
                    
                    # Debug: Show how many results were found
                    if source_filter or date_filters:
                        st.caption(f"📊 Found {len(all_results)} matching articles, showing top {len(results)}")
                    
                    # If no results found, try relaxing filters automatically
                    if not results and (metadata_filters.get('trl_range') or date_filters):
                        st.warning("⚠️ No results found with current filters. Attempting to relax filters...")
                        
                        # Strategy: Remove TRL range filter (most restrictive for research queries)
                        relaxed = False
                        
                        # Try removing TRL range first
                        if 'trl_range' in metadata_filters:
                            st.info("🔄 Retrying without TRL range filter...")
                            metadata_filters.pop('trl_range')
                            # Re-run the query without TRL filtering
                            all_results_backup = []
                            for source_name, embedding_index in sources_to_query:
                                try:
                                    results_retry = processor.query_similar(
                                        embedding_index,
                                        query_text=prompt,
                                        top_k=num_results * 2,  # Get more results for filtering
                                        filter_metadata=metadata_filters if metadata_filters else None
                                    )
                                    for result in results_retry:
                                        result['metadata']['source'] = source_name
                                    all_results_backup.extend(results_retry)
                                except Exception:
                                    continue
                            
                            # Re-apply date filters only (no tech filter)
                            if date_filters:
                                from datetime import datetime
                                filtered = []
                                for result in all_results_backup:
                                    pub_date = result['metadata'].get('publication_date', '')
                                    if pub_date:
                                        try:
                                            if 'T' in pub_date:
                                                pub_date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                            else:
                                                pub_date_obj = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                                            
                                            if 'date_after' in date_filters:
                                                after_date = datetime.strptime(date_filters['date_after'], '%Y-%m-%d')
                                                if pub_date_obj < after_date:
                                                    continue
                                            if 'date_before' in date_filters:
                                                before_date = datetime.strptime(date_filters['date_before'], '%Y-%m-%d')
                                                if pub_date_obj >= before_date:
                                                    continue
                                            filtered.append(result)
                                        except:
                                            continue
                                all_results_backup = filtered
                            
                            if all_results_backup:
                                all_results_backup.sort(key=lambda x: x['score'], reverse=True)
                                results = all_results_backup[:num_results]
                                st.success(f"✅ Found {len(results)} results after relaxing TRL filter")
                                relaxed = True
                    
                    # Extract relevant information
                    sources = []
                    context_texts = []
                    
                    if not results:
                        # Semantic fallback: retrieve based on pure relevance without metadata filters
                        st.warning("⚠️ No articles found with metadata filters. Falling back to semantic search...")
                        st.info("🔍 Searching for semantically relevant content in the knowledge base...")
                        
                        # Perform semantic search without any filters
                        fallback_results = []
                        for source_name, embedding_index in sources_to_query:
                            try:
                                semantic_results = processor.query_similar(
                                    embedding_index,
                                    query_text=prompt,
                                    top_k=num_results * 2,  # Get more results for better coverage
                                    filter_metadata=None  # No metadata filtering
                                )
                                for result in semantic_results:
                                    result['metadata']['source'] = source_name
                                fallback_results.extend(semantic_results)
                            except Exception as e:
                                logging.warning(f"Error in semantic fallback for {source_name}: {e}")
                                continue
                        
                        if fallback_results:
                            # Sort by relevance score and take top N (based on slider)
                            fallback_results.sort(key=lambda x: x['score'], reverse=True)
                            results = fallback_results[:num_results]
                            
                            st.success(f"✅ Found {len(results)} semantically relevant articles (metadata filters removed)")
                            st.caption("💡 Results are based on semantic relevance to your query, not metadata filters")
                        else:
                            # Absolute fallback - no results at all
                            error_msg = "No articles found in the knowledge base."
                            if source_filter:
                                error_msg += f" (Source filter: {source_filter})"
                            st.error(error_msg)
                            st.info("💡 Try:\n- Checking if data from that source is loaded\n- Using different keywords\n- Rebuilding embeddings if the knowledge base was recently updated")
                            st.stop()
                    
                    if results:
                        for result in results:
                            metadata = result['metadata']
                            
                            # Add to sources with ALL available fields
                            sources.append({
                                "url": metadata.get("url", ""),
                                "title": metadata.get("title", "Untitled"),
                                "publication_date": metadata.get("publication_date", ""),
                                "categories": metadata.get("categories", ""),
                                "indicator": metadata.get("full_indicator", ""),
                                "dimension": metadata.get("dimension", ""),
                                "tech": metadata.get("tech", ""),
                                "trl": metadata.get("trl", ""),
                                "startup": metadata.get("startup", ""),
                                "source": metadata.get("source", "")
                            })
                            
                            # Add to context (use the chunk text)
                            context_texts.append(result['text'])
                    
                    # Generate response using OpenAI
                    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
                    
                    # Create system prompt with context
                    current_date = datetime.now().strftime("%B %d, %Y")  # e.g., "December 01, 2025"
                    
                    # Add date filter info to help LLM understand the query context
                    date_context = ""
                    if date_filters:
                        if 'year' in date_filters and 'month' in date_filters:
                            month_names = ["", "January", "February", "March", "April", "May", "June", 
                                         "July", "August", "September", "October", "November", "December"]
                            month_name = month_names[date_filters['month']]
                            date_context = f"\n\nIMPORTANT: The user is asking specifically about articles from {month_name} {date_filters['year']}. Only consider the provided sources which have been filtered to this date range."
                        elif 'year' in date_filters:
                            date_context = f"\n\nIMPORTANT: The user is asking specifically about articles from {date_filters['year']}. Only consider the provided sources which have been filtered to this date range."
                        elif 'date_after' in date_filters:
                            date_context = f"\n\nIMPORTANT: The user is asking about recent/latest articles. Only consider the provided sources which have been filtered to recent publications."
                    
                    system_prompt = f"""You are an AI research assistant specialized in technology intelligence and innovation analysis. 
You have access to a knowledge base of articles about emerging technologies, startups, carbon markets, and clean energy from various sources.

**CURRENT DATE: {current_date}**

When interpreting dates in the provided sources, always reference them relative to the current date above.{date_context}

Use the provided context to answer questions accurately and comprehensively. 
When answering analytical questions (e.g., "what technologies are mentioned most", "trending topics", "most discussed"):
1. COUNT and ANALYZE the information from the provided sources
2. Look for PATTERNS and FREQUENCIES in the data
3. Provide SPECIFIC NUMBERS and STATISTICS when possible
4. List technologies/topics in order of frequency or importance
5. Always verify dates from the source metadata to ensure accuracy

If the context doesn't contain relevant information, say so clearly.

CRITICAL CITATION RULES:
You MUST accurately cite all your outputs with your sources using markdown hyperlink format when referencing information.

IMPORTANT: 
1. The user message contains a "CITATION GUIDE" section with exact URLs you MUST use
2. ONLY use URLs that are explicitly provided in that Citation Guide
3. NEVER create, modify, shorten, or guess URLs - use them EXACTLY as provided
4. Format: [Source Title](EXACT_URL_FROM_CITATION_GUIDE)
5. If a source has no valid URL in the Citation Guide, reference it by title only without a hyperlink

Example of correct citation: 'According to [TechCrunch Article](https://techcrunch.com/2024/11/15/full-article-url), the technology...'
The URL must be copied EXACTLY from the Citation Guide - do not modify it in any way.

The knowledge base includes articles from sources like TechCrunch, Carbon Herald, Hydrogen Central, etc., with metadata about:
- Technology areas (AI, renewable energy, carbon tech, etc.)
- Technology Readiness Levels (TRL 1-9)
- Dimensions (energy, environment, technology, climate, etc.)
- Publication dates and sources"""
                    
                    # Create context from search results
                    context = "\n\n---\n\n".join(context_texts) if context_texts else "No relevant information found in the knowledge base."
                    
                    # Create source information for citations
                    source_info = []
                    metadata_summary = []
                    valid_sources = []  # Track sources with valid URLs
                    
                    for i, src in enumerate(sources, 1):
                        title = src.get('title', f'Source {i}')
                        url = src.get('url', '')
                        pub_date = src.get('publication_date', 'Unknown date')
                        tech = src.get('tech', 'N/A')
                        dimension = src.get('dimension', 'N/A')
                        
                        # Validate URL - must start with http:// or https://
                        is_valid_url = url and (url.startswith('http://') or url.startswith('https://'))
                        
                        # Format publication date
                        if pub_date and pub_date != 'Unknown date':
                            try:
                                if 'T' in pub_date:
                                    date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                else:
                                    date_obj = datetime.strptime(pub_date[:10], '%Y-%m-%d')
                                pub_date = date_obj.strftime('%B %d, %Y')
                            except:
                                pass
                        
                        if is_valid_url:
                            # Create a clearer format for the model
                            source_info.append(f"**Source {i}**\n  - Title: {title}\n  - URL: {url}\n  - Published: {pub_date}")
                            metadata_summary.append(f"Source {i}: Tech: {tech}, Dimension: {dimension}")
                            valid_sources.append((i, title, url))
                        else:
                            # Skip sources without valid URLs
                            source_info.append(f"**Source {i}**\n  - Title: {title}\n  - URL: Not available\n  - Published: {pub_date}")
                            metadata_summary.append(f"Source {i}: Tech: {tech}, Dimension: {dimension}")
                    
                    sources_text = "\n\n".join(source_info)
                    metadata_text = "\n".join(metadata_summary)
                    
                    # Create a reference list of valid citations for the model
                    citation_guide = "\n".join([f"Source {i}: Use the format [{title}]({url})" for i, title, url in valid_sources]) if valid_sources else "No sources with valid URLs available."
                    
                    user_prompt = f"""Context information is below.
---------------------
{context}
---------------------

AVAILABLE SOURCES WITH URLS:
{sources_text}

CITATION GUIDE - Use EXACTLY these URLs when citing sources:
{citation_guide}

Source Metadata (for analysis):
{metadata_text}

CRITICAL CITATION INSTRUCTIONS:
- You MUST cite information using the EXACT URLs provided above
- Use the format: [Source Title](URL) where URL is taken EXACTLY from the "Citation Guide" above
- NEVER modify, shorten, or create your own URLs
- ONLY use the URLs explicitly listed in the Citation Guide
- If a source doesn't have a valid URL in the guide, reference it by title only without a link

For analytical questions about trends, frequencies, or patterns:
- Analyze ALL provided sources
- Count mentions across different articles
- Note publication dates to verify they match the requested timeframe
- Provide specific examples with citations using the EXACT URLs above

Question: {prompt}
Answer: """
                    
                    # Stream response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    stream = openai_client.chat.completions.create(
                        model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1500,
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 Sources (Database View)"):
                            # Validate URLs and show warning if needed
                            invalid_urls = []
                            for i, source in enumerate(sources, 1):
                                url = source.get('url', '')
                                if not url or not (url.startswith('http://') or url.startswith('https://')):
                                    invalid_urls.append(f"Source {i}: {url or 'No URL'}")
                            
                            if invalid_urls:
                                st.warning(f"⚠️ **URL Validation Warning:** {len(invalid_urls)} source(s) have invalid or missing URLs. These sources cannot be cited with hyperlinks.\n\n" + "\n".join(invalid_urls))
                            
                            for i, source in enumerate(sources, 1):
                                # Use URL as title if title is None or empty
                                title = source.get('title', 'Untitled')
                                if not title or title.lower() in ['none', 'untitled', 'n/a']:
                                    title = source.get('url', 'Untitled')
                                
                                st.markdown(f"**{i}. [{title}]({source['url']})**")
                                
                                # Display all fields in a structured database-like format
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.markdown("**📊 Database Fields:**")
                                    st.markdown(f"**Publication Date:** {source.get('publication_date', 'N/A')}")
                                    st.markdown(f"**Categories:** {source.get('categories', 'N/A')}")
                                    st.markdown(f"**Dimension:** {source.get('dimension', 'N/A')}")
                                    st.markdown(f"**Technology:** {source.get('tech', 'N/A')}")
                                    st.markdown(f"**TRL Level:** {source.get('trl', 'N/A')}")
                                    st.markdown(f"**Start-up:** {source.get('startup', 'N/A')}")
                                    st.markdown(f"**Source:** {source.get('source', 'N/A')}")
                                
                                with col2:
                                    st.markdown("**📝 Content Summary (Indicator):**")
                                    indicator_text = source.get('indicator', 'N/A')
                                    # Display full indicator text without truncation
                                    st.markdown(indicator_text)
                                
                                st.divider()
                    
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.stop()


elif page == "About":
    st.header("About Technology Intelligence Tool")

    st.markdown("""
    ### Overview

    The Technology Intelligence (TI) Tool is a comprehensive AI-powered platform for automated research, web crawling, 
    content extraction, analysis, and knowledge management. It streamlines the entire pipeline from discovery to insights 
    with cloud-native AWS S3 integration.

    ### Core Components

    **1. Web Search**
    - **Clarification** - AI asks targeted questions to refine research scope
    - **SERP Generation** - Creates optimized search queries from clarified intent
    - **Web Search** - Executes searches via SearxNG search engine
    - **Learning Extraction** - AI-powered analysis extracting key learnings, entities, and metrics
    - **Auto-save** - Results automatically saved locally and uploaded to S3

    **2. Web Crawler**
    - **Intelligent Crawling** - Respects robots.txt and configurable depth/delay settings
    - **Content Extraction** - Extracts clean text content from web pages
    - **URL Filtering** - Remove non-article pages (about, contact, author profiles, etc.)
    - **Customizable Patterns** - Preview and customize filter patterns before applying
    - **Auto-cleanup** - Original unfiltered files automatically deleted after S3 upload
    - **S3 Sync** - All crawled and filtered data uploaded to AWS S3

    **3. LLM Extraction**
    - **Structured Metadata** - AI extracts title, summary, author, publication date
    - **Tech Intelligence** - Dimension, Tech, TRL, URL to start-up(s) classification
    - **Smart Filtering** - Auto-removes empty content and dates >2 years old
    - **Dual Format** - Outputs both CSV and JSON to processed_data folder
    - **S3 Upload** - Both formats automatically uploaded to cloud storage

    **4. Summarization**
    - **Tech-Intel Analysis** - AI generates Indicator, Dimension, Tech, TRL, URL to start-up(s) fields
    - **Auto-save** - Files automatically saved and uploaded to S3 after completion
    - **Processing History** - Track all processed files with metadata
    - **Preview Mode** - View first 5 entries with original vs analyzed content
    - **Export Options** - Download CSV and detailed processing logs

    **5. Database**
    - **Consolidated View** - All summarized content from multiple sources
    - **Advanced Filtering** - By category, source, date range, and keywords
    - **Full-text Search** - Search across summaries, titles, and content
    - **Multiple Views** - Cards, Table, and Detailed view modes
    - **Bulk Export** - Export filtered or complete database
    - **Text Selection** - Copy text from any cell in the table

    **6. LinkedIn Home Feed Monitor**
    - **Automated Scraping** - Collects posts from LinkedIn home feed
    - **Smart Filtering** - Excludes promoted/suggested/reposted content
    - **Content Processing** - Extracts author, date, content, and URLs
    - **Auto-translation** - Non-English posts translated to English
    - **Deduplication** - Removes duplicate posts based on content
    - **URL Filtering** - Removes profile/company/hashtag links, keeps articles
    - **Date Limiting** - Configurable days back (1-90 days)
    - **S3 Management** - Upload, download, delete files via UI
    - **Targeted Network** - Monitors VCs and Companies (SOSV, Seed Capital, ADB Ventures)

    ### Key Features

    ✅ **End-to-End Pipeline** - From web search to structured insights  
    ✅ **AI-Powered** - GPT-4 for extraction, summarization, and analysis  
    ✅ **Cloud-Native** - Automatic AWS S3 backup for all outputs  
    ✅ **Multi-Format** - CSV, JSON, Markdown outputs  
    ✅ **Real-time Progress** - Live tracking with time estimates  
    ✅ **Smart Filtering** - Automatic cleanup of irrelevant content  
    ✅ **Social Intelligence** - LinkedIn network monitoring  
    ✅ **Batch Processing** - Handle large datasets efficiently  
    ✅ **History Tracking** - Complete audit trail of all operations  

    ### Technology Stack

    - **Frontend:** Streamlit with AgGrid
    - **AI Models:** OpenAI (gpt-4o, gpt-4o-mini, text-embedding-3-large)
    - **Alternative LLMs:** OpenAI, LM Studio (local)
    - **Search Engine:** SearxNG
    - **Web Automation:** Selenium WebDriver
    - **Vector Store:** Simple in-memory keyword search
    - **Cloud Storage:** AWS S3 (boto3)
    - **Validation:** Pydantic
    - **Agent Framework:** Pydantic AI
    - **Data Processing:** Pandas, NumPy

    ### AWS S3 Integration

    All pipeline outputs are automatically backed up to AWS S3:

    - **research_results/** - Web search outputs (JSON, Markdown)
    - **crawled_data/** - Web crawler outputs (CSV, JSON)
    - **processed_data/** - LLM extraction outputs (CSV, JSON)
    - **summarised_content/** - Summarization outputs (CSV, JSON)
    - **linkedin_data/** - LinkedIn posts (CSV, JSON)

    **Features:**
    - ✅ Automatic upload after processing
    - ✅ Download and delete via UI (LinkedIn)
    - ✅ Graceful fallback if S3 unavailable

    ### Usage Workflow

    **Complete Research Pipeline:**
    ```
    Web Search → Web Crawler → URL Filter → LLM Extraction → Summarization → Database
    ```

    **LinkedIn Intelligence:**
    ```
    LinkedIn Monitor → Scrape Posts → Deduplicate → Filter URLs → S3 Upload
    ```

    ### Quick Start Guide

    **Web Search:**
    1. Enter research topic
    2. Answer clarification questions
    3. Execute search and save results
    4. Files auto-uploaded to S3

    **Web Crawler & URL Filtering:**
    1. Navigate to "Web Crawler" page
    2. **Crawl Websites** tab:
       - Enter website URLs to crawl
       - Configure crawl settings (depth, delay)
       - Start crawling
       - Results saved and uploaded to S3
    3. **Filter URLs** tab:
       - Select crawled CSV file
       - Preview URLs to be removed
       - Apply filter
       - Filtered file uploaded to S3, original deleted

    **LLM Extraction:**
    1. Navigate to "LLM Extraction" page
    2. Select filtered CSV from crawled_data
    3. Choose output folder
    4. Start extraction
    5. CSV and JSON auto-uploaded to S3

    **Summarization:**
    1. Navigate to "Summarization" tab
    2. Select CSV from processed_data folder
    3. Click "Start Summarization"
    4. Files automatically saved to summarised_content/
    5. CSV and JSON auto-uploaded to S3 (logs kept local)
    6. View preview of 5 entries
    7. Browse full dataset in table

    **Database:**
    1. Navigate to "Database" tab
    2. View consolidated data from all sources
    3. Use filters to narrow down results
    4. Search for specific keywords
    5. Switch between view modes
    6. Export filtered or complete data

    **LinkedIn Monitor:**
    1. Navigate to "LinkedIn Home Feed Monitor"
    2. Configure scraping settings (scrolls, delay, days back)
    3. Click "Start Collection"
    4. Watch live browser automation
    5. Files auto-uploaded to S3
    6. Manage files via S3 Storage UI

    ### File Structure

    **Local Storage:**
    ```
    data/                          # Web search results
    crawled_data/                  # Raw and filtered crawls
    processed_data/                # LLM extracted metadata
    summarised_content/            # Summarized content + history.json
    rag_storage/                   # Vector indexes (local cache)
    linkedin_posts_monitor/
      └── linkedin_data/           # LinkedIn posts
    ```

    **AWS S3 Structure:**
    ```
    s3://bucket/
    ├── research_results/          # Search results, learnings
    ├── crawled_data/              # Crawled and filtered CSVs
    ├── processed_data/            # LLM extraction outputs
    ├── summarised_content/        # Summarized CSVs and JSONs
    └── linkedin_data/             # LinkedIn posts CSVs/JSONs
    ```

    ### Support & Troubleshooting

    - **Logs:** Check `research.log` for detailed operation logs
    - **S3 Status:** Run `python3 check_s3_status.py` to verify bucket contents
    - **Upload Missing Files:** Use `python3 upload_linkedin_to_s3.py` for backfill
    - **LinkedIn:** Monitor uses standard Selenium, ensure ChromeDriver installed. A Linkedin account was created solely for this tool.

    ---
    
    **Developed by Sharifah, with some guidance from AISG** | Technology Intelligence Tool © 2025
    """)

elif page == "LinkedIn Home Feed Monitor":
    st.header("🔗 LinkedIn Home Feed Monitor")
    
    st.markdown("""
    ### Overview
    
    This tool automates the collection of posts from your LinkedIn home feed using browser automation. 
    It filters out promoted/suggested/reposted content and collects authentic posts from your network.
    
    **Features:**
    - Automated LinkedIn login and feed scrolling
    - Filters out ads, promoted posts, and reposts
    - Extracts author, date, content, and URLs
    - Automatic translation of non-English posts to English
    - Live browser preview during scraping
    - Saves results in CSV and JSON formats
    - Automatic S3 backup and restore
    """)
    
    st.divider()
    
    # S3 Management Section
    with st.expander("☁️ S3 Storage Management", expanded=False):
        st.markdown("**Download or delete LinkedIn data from S3**")
        
        try:
            from aws_storage import get_storage
            s3_storage = get_storage()
            
            # List all LinkedIn files in S3
            linkedin_files = s3_storage.list_files(prefix="linkedin_data/", suffix=".csv")
            linkedin_files.extend(s3_storage.list_files(prefix="linkedin_data/", suffix=".json"))
            
            if linkedin_files:
                st.success(f"Found {len(linkedin_files)} files in S3")
                
                # Display files in a table
                file_data = []
                for file_key in linkedin_files:
                    file_name = file_key.split('/')[-1]
                    file_type = file_name.split('.')[-1].upper()
                    file_data.append({
                        "File Name": file_name,
                        "Type": file_type,
                        "S3 Key": file_key
                    })
                
                files_df = pd.DataFrame(file_data)
                st.dataframe(files_df, use_container_width=True)
                
                # Download and Delete options
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_file = st.selectbox(
                        "Select file to download",
                        options=[f["File Name"] for f in file_data]
                    )
                    
                    if st.button("⬇️ Download from S3", use_container_width=True):
                        # Find the S3 key
                        s3_key = next((f["S3 Key"] for f in file_data if f["File Name"] == selected_file), None)
                        if s3_key:
                            temp_path = Path(f"/tmp/{selected_file}")
                            if s3_storage.download_file(s3_key, str(temp_path)):
                                with open(temp_path, 'rb') as f:
                                    st.download_button(
                                        f"📥 Save {selected_file}",
                                        data=f.read(),
                                        file_name=selected_file,
                                        mime="text/csv" if selected_file.endswith('.csv') else "application/json"
                                    )
                                temp_path.unlink()
                                st.success(f"✅ Downloaded {selected_file}")
                            else:
                                st.error("Failed to download file")
                
                with col2:
                    selected_delete = st.selectbox(
                        "Select file to delete",
                        options=[f["File Name"] for f in file_data],
                        key="delete_select"
                    )
                    
                    if st.button("🗑️ Delete from S3", use_container_width=True, type="secondary"):
                        s3_key = next((f["S3 Key"] for f in file_data if f["File Name"] == selected_delete), None)
                        if s3_key:
                            if s3_storage.delete_file(s3_key):
                                st.success(f"✅ Deleted {selected_delete} from S3")
                                st.rerun()
                            else:
                                st.error("Failed to delete file")
            else:
                st.info("No LinkedIn files found in S3 yet. Scrape some posts to upload!")
                
        except Exception as e:
            st.warning(f"S3 not configured: {str(e)}")
    
    st.divider()
    
    # Fixed credentials from environment
    linkedin_username = os.getenv("LINKEDIN_USERNAME")
    linkedin_password = os.getenv("LINKEDIN_PASSWORD")
    
    # Configuration section
    st.subheader("⚙️ Configuration")
    
    # Show tracking info (without exposing email)
    st.info("🎯 **Tracking:** Posts from VCs and Companies including SOSV, Seed Capital, ADB Ventures, and portfolio network")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scroll_method = st.selectbox(
            "Scroll Method",
            ["smooth", "to_bottom", "fixed_pixels", "by_viewport"],
            index=0,
            help="How to scroll the page: smooth (human-like), to_bottom (jump), fixed_pixels (fixed distance), by_viewport (screen height)"
        )
        
        scroll_pause = st.slider(
            "Scroll Pause (seconds)",
            min_value=5,
            max_value=20,
            value=10,
            help="Time to wait between scrolls for content to load"
        )
        
        days_limit = st.slider(
            "Days to Look Back",
            min_value=1,
            max_value=90,
            value=30,
            help="Stop scrolling when posts older than this many days are found"
        )
    
    with col2:
        enable_translation = st.checkbox(
            "Enable Translation",
            value=True,
            help="Automatically translate non-English posts to English using OpenAI"
        )
        
        output_dir = st.text_input(
            "Output Directory",
            value="linkedin_posts_monitor/linkedin_data",
            help="Directory to save collected posts"
        )
    
    st.divider()
    
    # Status and controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if 'linkedin_scraping' not in st.session_state:
            st.session_state.linkedin_scraping = False
        
        if st.session_state.linkedin_scraping:
            st.warning("⏳ Scraping in progress... Please wait.")
        else:
            st.info("👉 Click 'Start Scraping' to begin collecting LinkedIn posts.")
    
    with col2:
        start_button = st.button(
            "🚀 Start Scraping",
            disabled=st.session_state.get('linkedin_scraping', False),
            use_container_width=True
        )
    
    with col3:
        if st.session_state.get('linkedin_scraping', False):
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.linkedin_scraping = False
                st.warning("Scraping stopped by user.")
                st.rerun()
    
    st.divider()
    
    # Display area
    status_container = st.container()
    screenshot_container = st.container()
    results_container = st.container()
    
    if start_button:
        st.session_state.linkedin_scraping = True
        
        # Import required modules for LinkedIn scraping
        import time
        import json
        import csv
        import re
        from datetime import datetime, timedelta
        from pathlib import Path
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.common.exceptions import ElementClickInterceptedException
        from openai import OpenAI
        
        # Status display
        status_placeholder = status_container.empty()
        screenshot_placeholder = screenshot_container.empty()
        
        # Helper functions from linkedin_homefeed.py
        def is_english(text):
            """Detect if text is in English using multiple heuristics"""
            if not text or len(text.strip()) < 10:
                return True
            
            text_lower = text.lower()
            common_english_words = [
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
            ]
            
            word_count = sum(1 for word in common_english_words 
                           if f' {word} ' in f' {text_lower} ' or 
                           text_lower.startswith(f'{word} ') or 
                           text_lower.endswith(f' {word}'))
            
            words_in_text = len(text_lower.split())
            if words_in_text > 0:
                english_ratio = word_count / min(words_in_text, len(common_english_words))
                if english_ratio >= 0.3:
                    return True
            
            non_ascii_count = sum(1 for char in text if ord(char) > 127)
            if non_ascii_count > len(text) * 0.2:
                return False
            
            return word_count >= 5
        
        def translate_to_english(text, openai_client):
            """Translate text to English using OpenAI API"""
            if not text or not text.strip() or not openai_client:
                return text
            
            if is_english(text):
                return text
            
            try:
                response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a professional translator. Translate the following text to English. Only return the translated text, nothing else."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                translated_text = response.choices[0].message.content.strip()
                status_placeholder.info(f"🌐 Translated non-English post")
                return translated_text
                
            except Exception as e:
                status_placeholder.warning(f"⚠️ Translation failed: {str(e)[:50]}... Keeping original text.")
                return text
        
        def parse_relative_date(relative_date_text):
            """Convert LinkedIn's relative date format to actual datetime string"""
            if not relative_date_text:
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            text = relative_date_text.lower().strip()
            now = datetime.now()
            
            patterns = [
                (r'(\d+)\s*s(?:ec|econd)?s?\s*(?:ago)?', 'seconds'),
                (r'(\d+)\s*m(?:in|inute)?s?\s*(?:ago)?', 'minutes'),
                (r'(\d+)\s*h(?:r|our)?s?\s*(?:ago)?', 'hours'),
                (r'(\d+)\s*d(?:ay)?s?\s*(?:ago)?', 'days'),
                (r'(\d+)\s*w(?:eek)?s?\s*(?:ago)?', 'weeks'),
                (r'(\d+)\s*mo(?:nth)?s?\s*(?:ago)?', 'months'),
                (r'(\d+)\s*y(?:ear)?s?\s*(?:ago)?', 'years'),
            ]
            
            post_time = now  # Initialize with current time
            
            for pattern, unit in patterns:
                match = re.search(pattern, text)
                if match:
                    value = int(match.group(1))
                    
                    if unit == 'seconds':
                        post_time = now - timedelta(seconds=value)
                    elif unit == 'minutes':
                        post_time = now - timedelta(minutes=value)
                    elif unit == 'hours':
                        post_time = now - timedelta(hours=value)
                    elif unit == 'days':
                        post_time = now - timedelta(days=value)
                    elif unit == 'weeks':
                        post_time = now - timedelta(weeks=value)
                    elif unit == 'months':
                        post_time = now - timedelta(days=value*30)
                    elif unit == 'years':
                        post_time = now - timedelta(days=value*365)
                    
                    return post_time.strftime("%Y-%m-%d %H:%M:%S")
            
            return now.strftime("%Y-%m-%d %H:%M:%S")
        
        def scroll_page(driver, method="smooth", pixels=800, speed="slow"):
            """Scroll the page using different methods"""
            if method == "to_bottom":
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            elif method == "fixed_pixels":
                driver.execute_script(f"window.scrollBy(0, {pixels});")
            elif method == "by_viewport":
                driver.execute_script("window.scrollBy(0, window.innerHeight);")
            elif method == "smooth":
                speed_map = {"slow": 50, "medium": 100, "fast": 200}
                step = speed_map.get(speed, 100)
                viewport_height = driver.execute_script("return window.innerHeight;")
                
                for i in range(0, viewport_height, step):
                    driver.execute_script(f"window.scrollBy(0, {step});")
                    time.sleep(0.05)
            
            return driver.execute_script("return window.pageYOffset + window.innerHeight;")
        
        def parse_post(post_element, openai_client, enable_translation):
            """Parse a LinkedIn post element with comprehensive author extraction"""
            try:
                # Skip promoted/suggested/reposted/liked posts
                ad_badges = post_element.find_elements(By.XPATH, ".//*[contains(text(),'Promoted') or contains(text(),'Suggested') or contains(text(),'Reposted') or contains(text(),'Liked')]")
                if ad_badges:
                    return None

                # Try multiple strategies to find author/company name
                author = ""
                
                # Strategy 1: Look for aria-label with person/company name
                try:
                    author_links = post_element.find_elements(By.XPATH, ".//a[contains(@href, '/in/') or contains(@href, '/company/')]")
                    for author_link in author_links:
                        aria_label = author_link.get_attribute("aria-label")
                        if aria_label and len(aria_label.strip()) > 0:
                            # Filter out non-name labels
                            if not any(x in aria_label.lower() for x in ['hashtag', 'like', 'comment', 'share', 'repost']):
                                # Clean up the aria-label to extract just the name
                                cleaned_name = aria_label.strip()
                                cleaned_name = re.sub(r'^View\s+', '', cleaned_name, flags=re.IGNORECASE)
                                cleaned_name = re.sub(r"'s?\s+(profile|page|link)\s*$", '', cleaned_name, flags=re.IGNORECASE)
                                cleaned_name = re.sub(r'\s+profile\s*$', '', cleaned_name, flags=re.IGNORECASE)
                                cleaned_name = re.sub(r',?\s*graphic\.?$', '', cleaned_name, flags=re.IGNORECASE)
                                cleaned_name = re.sub(r'\s+graphic\s+(link|icon)?\s*$', '', cleaned_name, flags=re.IGNORECASE)
                                cleaned_name = cleaned_name.strip()
                                
                                if cleaned_name and len(cleaned_name) > 1:
                                    author = cleaned_name
                                    break
                except:
                    pass
                
                # Strategy 2: Enhanced span selectors with more variations
                if not author:
                    author_selectors = [
                        ".//span[contains(@class, 'feed-shared-actor__name')]//span[@aria-hidden='true']",
                        ".//span[contains(@class, 'update-components-actor__name')]//span[@aria-hidden='true']",
                        ".//div[contains(@class, 'update-components-actor__name')]//span[@aria-hidden='true']",
                        ".//div[contains(@class, 'update-components-actor')]//span[@dir='ltr']",
                        ".//a[contains(@class, 'app-aware-link')]//span[@dir='ltr'][1]",
                        ".//span[contains(@class, 'feed-shared-actor__name')]",
                        ".//div[contains(@class, 'feed-shared-actor__container-link')]//span[1]",
                        ".//a[contains(@class, 'feed-shared-actor__container-link')]//span[not(@aria-hidden='true')]",
                        ".//div[contains(@class, 'feed-shared-actor')]//a//span[1]",
                        ".//span[contains(@class, 'update-components-actor__title')]//span[1]"
                    ]
                    for selector in author_selectors:
                        try:
                            elem = post_element.find_element(By.XPATH, selector)
                            author = elem.text.strip()
                            if author and len(author) > 0 and not author.startswith('•'):
                                break
                        except:
                            continue
                
                # Strategy 3: Look for links with profile/company URLs and extract visible text
                if not author:
                    try:
                        profile_links = post_element.find_elements(By.XPATH, ".//a[contains(@href, '/in/') or contains(@href, '/company/')]")
                        for link in profile_links:
                            text = link.text.strip()
                            if text and len(text) > 2 and len(text) < 100:
                                if not any(x in text.lower() for x in ['ago', 'edited', '•', 'follow', 'like', 'comment']):
                                    cleaned_name = re.sub(r'^View\s+', '', text, flags=re.IGNORECASE)
                                    cleaned_name = re.sub(r"'s?\s+(profile|page|link)\s*$", '', cleaned_name, flags=re.IGNORECASE)
                                    cleaned_name = re.sub(r'\s+profile\s*$', '', cleaned_name, flags=re.IGNORECASE)
                                    cleaned_name = cleaned_name.strip()
                                    
                                    if cleaned_name and len(cleaned_name) > 1:
                                        author = cleaned_name
                                        break
                    except:
                        pass
                
                # Final validation and cleanup
                if author:
                    author = author.split('•')[0].strip()
                    author = author.split('\n')[0].strip()
                    author = re.sub(r"'s?\s*$", '', author, flags=re.IGNORECASE)
                    author = author.rstrip('.,')
                    if len(author) > 150:
                        author = ""
                
                if not author or len(author) < 2:
                    return None

                # Extract date with multiple selectors
                relative_date = ""
                date_selectors = [
                    ".//span[contains(@class, 'feed-shared-actor__sub-description')]",
                    ".//span[contains(@class, 'update-components-actor__sub-description')]",
                    ".//time",
                    ".//*[contains(text(), 'ago') or contains(text(), 'h') or contains(text(), 'd') or contains(text(), 'w')]"
                ]
                
                for selector in date_selectors:
                    try:
                        elem = post_element.find_element(By.XPATH, selector)
                        text = elem.text.strip()
                        if any(indicator in text.lower() for indicator in ['ago', 'h', 'd', 'w', 'mo', 'yr', 'sec', 'min', 'hour', 'day', 'week', 'month', 'year']):
                            relative_date = text
                            break
                    except:
                        continue
                
                actual_datetime = parse_relative_date(relative_date) if relative_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Try to expand "see more" with multiple selectors
                try:
                    see_more_selectors = [
                        ".//button[contains(@aria-label, 'more')]",
                        ".//button[contains(text(), '…more')]",  # Ellipsis character
                        ".//button[contains(text(), '...more')]",  # Three dots
                        ".//button[contains(text(), 'see more')]",
                        ".//button[contains(@class, 'see-more')]",
                        ".//span[contains(@class, 'see-more')]//button",
                        ".//button[contains(@aria-label, 'See more')]",
                        ".//div[contains(@class, 'feed-shared-inline-show-more-text')]//button"
                    ]
                    
                    for selector in see_more_selectors:
                        try:
                            see_more_button = post_element.find_element(By.XPATH, selector)
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", see_more_button)
                            time.sleep(0.3)
                            see_more_button.click()
                            time.sleep(0.5)
                            break
                        except:
                            continue
                except:
                    pass

                # Extract content with multiple selectors
                content = ""
                content_selectors = [
                    ".//div[contains(@class, 'feed-shared-update-v2__description')]",
                    ".//div[contains(@class, 'update-components-text')]",
                    ".//div[contains(@class, 'feed-shared-text')]",
                    ".//span[contains(@class, 'break-words')]"
                ]
                for selector in content_selectors:
                    try:
                        content_elem = post_element.find_element(By.XPATH, selector)
                        content = content_elem.text.strip()
                        if content:
                            break
                    except:
                        continue
                
                if enable_translation and content and openai_client:
                    content = translate_to_english(content, openai_client)
                
                # Extract URLs
                urls = " | ".join([a.get_attribute("href") for a in post_element.find_elements(By.TAG_NAME, "a") 
                                  if a.get_attribute("href") and ("http" in a.get_attribute("href"))])

                return {
                    "Person/Company name": author,
                    "Date of post": actual_datetime,
                    "Content of post": content if content else "No content",
                    "URLs": urls
                }
                
            except Exception as e:
                return None
        
        try:
            # Initialize OpenAI if translation is enabled
            openai_client = None
            if enable_translation:
                try:
                    openai_client = OpenAI(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        base_url="https://api.openai.com/v1"
                    )
                except Exception as e:
                    status_placeholder.warning(f"⚠️ Could not initialize OpenAI: {e}\nTranslation disabled.")
            
            # Setup Chrome driver
            chrome_options = Options()
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--start-maximized")
            
            driver = webdriver.Chrome(options=chrome_options)
            
            # Login
            status_placeholder.info("🌐 Opening LinkedIn login page...")
            driver.get("https://www.linkedin.com/login")
            driver.find_element(By.ID, "username").send_keys(linkedin_username)
            driver.find_element(By.ID, "password").send_keys(linkedin_password)
            driver.find_element(By.XPATH, "//button[@type='submit']").click()
            time.sleep(5)
            
            # Show screenshot
            screenshot = driver.get_screenshot_as_png()
            screenshot_placeholder.image(screenshot, caption="Browser View", use_container_width=True)
            
            status_placeholder.success("✅ Login successful!")
            
            # Navigate to feed
            status_placeholder.info("📰 Navigating to LinkedIn feed...")
            driver.get("https://www.linkedin.com/feed/")
            time.sleep(3)
            
            screenshot = driver.get_screenshot_as_png()
            screenshot_placeholder.image(screenshot, caption="Browser View", use_container_width=True)
            
            # Collect posts
            posts = set()
            results = []
            last_height = driver.execute_script("return document.body.scrollHeight")
            start_time = time.time()
            scroll_count = 0
            cutoff_date = datetime.now() - timedelta(days=days_limit)
            oldest_post_date = datetime.now()
            posts_beyond_limit = 0  # Counter for consecutive posts beyond date limit
            
            status_placeholder.info(f"🔄 Starting to collect posts (looking back {days_limit} days, max 10 minutes)...")
            
            while st.session_state.linkedin_scraping:
                # Find post elements
                post_elements = driver.find_elements(By.XPATH, "//div[contains(@class,'feed-shared-update-v2')]")
                
                status_placeholder.info(f"Scroll #{scroll_count + 1}: Found {len(post_elements)} post elements")
                
                new_posts = 0
                for post_element in post_elements:
                    if post_element in posts:
                        continue
                    
                    data = parse_post(post_element, openai_client, enable_translation)
                    if data and data not in results:
                        # Parse the post date to check if it's within the limit
                        try:
                            post_date = datetime.strptime(data["Date of post"], "%Y-%m-%d %H:%M:%S")
                            
                            # Track the oldest post we've seen
                            if post_date < oldest_post_date:
                                oldest_post_date = post_date
                            
                            # Check if post is within the date range
                            if post_date >= cutoff_date:
                                results.append(data)
                                new_posts += 1
                                posts_beyond_limit = 0  # Reset counter
                            else:
                                # Post is too old, increment counter
                                posts_beyond_limit += 1
                                
                        except Exception as e:
                            # If date parsing fails, still add the post
                            results.append(data)
                            new_posts += 1
                    
                    posts.add(post_element)
                
                scroll_count += 1
                days_back = (datetime.now() - oldest_post_date).days
                status_placeholder.info(f"✅ New posts: {new_posts} | Total collected: {len(results)} | Oldest: {days_back} days ago")
                
                # Update screenshot
                screenshot = driver.get_screenshot_as_png()
                screenshot_placeholder.image(screenshot, caption="Browser View", use_container_width=True)
                
                # Check if we've hit too many posts beyond the limit
                if posts_beyond_limit >= 10:
                    status_placeholder.success(f"✅ Reached {days_limit}-day limit! Found posts from {days_back} days ago.")
                    break
                
                # Scroll
                scroll_page(driver, method=scroll_method, speed="slow")
                time.sleep(scroll_pause)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                
                if new_height == last_height:
                    status_placeholder.info("⚠️ Reached end of feed")
                    break
                last_height = new_height
                
                # Time limit
                if (time.time() - start_time) > 3600:  # 60 minutes
                    status_placeholder.info("⏱️ Time limit reached (60 minutes)")
                    break
            
            driver.quit()
            
            # Post-processing: Deduplication and URL filtering
            if results:
                status_placeholder.info("🔄 Post-processing: Removing duplicates and filtering URLs...")
                
                original_count = len(results)
                
                # Step 1: Deduplicate based on "Content of post"
                seen_content = set()
                deduplicated_results = []
                
                for post in results:
                    content = post.get("Content of post", "").strip()
                    
                    # Skip if we've seen this exact content before
                    if content and content != "No content" and content not in seen_content:
                        seen_content.add(content)
                        deduplicated_results.append(post)
                    elif not content or content == "No content":
                        # Keep posts with no content (they might have unique URLs)
                        deduplicated_results.append(post)
                
                duplicates_removed = original_count - len(deduplicated_results)
                status_placeholder.info(f"✓ Removed {duplicates_removed} duplicate posts")
                
                # Step 2: Filter URLs - Remove individual/company/hashtag links
                def filter_urls(url_string):
                    """Filter out LinkedIn profile, company, and hashtag URLs"""
                    if not url_string or url_string.strip() == "":
                        return ""
                    
                    urls = url_string.split(" | ")
                    filtered_urls = []
                    
                    for url in urls:
                        url_lower = url.lower()
                        
                        # Skip URLs that are profiles, companies, hashtags, or LinkedIn internal pages
                        skip_patterns = [
                            '/in/',           # Individual profiles
                            '/company/',      # Company pages
                            '/school/',       # School pages
                            '/feed/',         # Feed links
                            '/hashtag/',      # Hashtag pages
                            '/groups/',       # Group pages
                            '/showcase/',     # Showcase pages
                            'linkedin.com/posts/',  # Direct post links
                            'linkedin.com/pulse/',  # Pulse articles (keep these as they're content)
                        ]
                        
                        # Keep pulse articles as they are actual content
                        if 'linkedin.com/pulse/' in url_lower:
                            filtered_urls.append(url)
                            continue
                        
                        # Skip if URL matches any skip pattern
                        should_skip = any(pattern in url_lower for pattern in skip_patterns[:-1])  # Exclude pulse from skip
                        
                        if not should_skip:
                            # Keep external URLs and content URLs
                            filtered_urls.append(url)
                    
                    return " | ".join(filtered_urls)
                
                # Apply URL filtering to all posts
                urls_filtered_count = 0
                for post in deduplicated_results:
                    original_urls = post.get("URLs", "")
                    filtered_urls = filter_urls(original_urls)
                    
                    if original_urls != filtered_urls:
                        urls_filtered_count += 1
                    
                    post["URLs"] = filtered_urls
                
                status_placeholder.info(f"✓ Filtered URLs in {urls_filtered_count} posts (removed profile/company/hashtag links)")
                
                # Update results with processed data
                results = deduplicated_results
                
                status_placeholder.success(f"✅ Post-processing complete! {len(results)} unique posts (removed {duplicates_removed} duplicates)")
            
            # Save results
            if results:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                csv_filename = f"{output_dir}/linkedin_posts_{timestamp}.csv"
                json_filename = f"{output_dir}/linkedin_posts_{timestamp}.json"
                
                with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ["Person/Company name", "Date of post", "Content of post", "URLs"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
                
                with open(json_filename, 'w', encoding='utf-8') as jsonfile:
                    json.dump(results, jsonfile, indent=2, ensure_ascii=False)
                
                # Upload to S3 if configured
                try:
                    from aws_storage import get_storage
                    s3_storage = get_storage()
                    
                    # Upload both CSV and JSON to S3
                    csv_s3_key = f"linkedin_data/linkedin_posts_{timestamp}.csv"
                    json_s3_key = f"linkedin_data/linkedin_posts_{timestamp}.json"
                    
                    csv_uploaded = s3_storage.upload_file(csv_filename, csv_s3_key)
                    json_uploaded = s3_storage.upload_file(json_filename, json_s3_key)
                    
                    if csv_uploaded and json_uploaded:
                        status_placeholder.success(f"✅ Files uploaded to S3: s3://{s3_storage.bucket_name}/linkedin_data/")
                    else:
                        status_placeholder.warning("⚠️ Some files failed to upload to S3")
                        
                except Exception as e:
                    status_placeholder.warning(f"⚠️ S3 upload skipped: {str(e)}")
                
                status_placeholder.success(f"✅ Collection complete! {len(results)} unique posts")
                
                # Display results with statistics
                results_container.subheader("📊 Collected Posts")
                
                # Show statistics
                stats_col1, stats_col2, stats_col3 = results_container.columns(3)
                with stats_col1:
                    st.metric("Total Posts", len(results))
                with stats_col2:
                    st.metric("Duplicates Removed", duplicates_removed)
                with stats_col3:
                    st.metric("URLs Filtered", urls_filtered_count)
                
                results_container.info("""
                **Post-Processing Applied:**
                - ✅ Deduplicated based on post content
                - ✅ Removed profile/company/hashtag links
                - ✅ Kept external article/video/document links
                """)
                
                results_df = pd.DataFrame(results)
                results_container.dataframe(results_df, use_container_width=True)
                
                results_container.download_button(
                    "📥 Download CSV",
                    data=open(csv_filename, 'rb').read(),
                    file_name=f"linkedin_posts_{timestamp}.csv",
                    mime="text/csv"
                )
                
                results_container.download_button(
                    "📥 Download JSON",
                    data=open(json_filename, 'rb').read(),
                    file_name=f"linkedin_posts_{timestamp}.json",
                    mime="application/json"
                )
            else:
                status_placeholder.warning("⚠️ No posts collected")
            
        except Exception as e:
            status_placeholder.error(f"❌ Error during scraping: {str(e)}")
            if 'driver' in locals():
                driver.quit()
        
        finally:
            st.session_state.linkedin_scraping = False

# Footer
st.sidebar.divider()

# Show processing status in sidebar
if st.session_state.csv_processing:
    st.sidebar.markdown("### 🔄 Summarization Status")
    progress = st.session_state.csv_progress
    if progress['total'] > 0:
        progress_pct = progress['current'] / progress['total']
        st.sidebar.progress(progress_pct)
        st.sidebar.caption(f"Summarizing: {progress['current']}/{progress['total']} rows")
        if progress['remaining'] > 0:
            mins = int(progress['remaining'] // 60)
            secs = int(progress['remaining'] % 60)
            st.sidebar.caption(f"⏳ Est. remaining: {mins}m {secs}s" if mins > 0 else f"⏳ Est. remaining: {secs}s")
        st.sidebar.error("⚠️ Stay on Summarization page!")
    st.sidebar.divider()
    
    # If not on the summarization page, show warning and option to stop
    if page != "Summarization":
        st.sidebar.warning("Processing was interrupted by navigation.")
        if st.sidebar.button("Clear Interrupted Task"):
            st.session_state.csv_processing = False
            st.session_state.csv_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
            st.rerun()

# Show crawling status in sidebar
if st.session_state.crawling_in_progress:
    st.sidebar.markdown("### 🕷️ Crawling Status")
    progress = st.session_state.crawl_progress
    if progress['total'] > 0 and progress['current'] > 0:
        st.sidebar.progress(progress['current'] / progress['total'])
        st.sidebar.caption(f"{progress['current']}/{progress['total']} pages crawled")
        if progress['remaining'] > 0:
            mins = int(progress['remaining'] // 60)
            secs = int(progress['remaining'] % 60)
            st.sidebar.caption(f"⏳ Est. remaining: {mins}m {secs}s" if mins > 0 else f"⏳ Est. remaining: {secs}s")
        st.sidebar.error("⚠️ Stay on Web Crawler page!")
    st.sidebar.divider()
    
    # If not on the web crawler page, show warning and option to cancel
    if page != "Web Crawler":
        st.sidebar.warning("Crawling was interrupted by navigation.")
        if st.sidebar.button("Cancel Crawl"):
            st.session_state.crawl_cancel_requested = True
            st.session_state.crawling_in_progress = False
            st.session_state.crawl_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0, 'start_time': None}
            st.rerun()

st.sidebar.markdown("### ⚙️ Settings")

# LLM Provider Configuration
st.sidebar.markdown("#### 🤖 LLM Provider")
from config.model_config import MODEL_OPTIONS, get_available_providers

# Get current provider from environment
current_provider = os.getenv("LLM_PROVIDER", "openai").lower()

# Map provider codes to display names
provider_display_map = {
    "openai": "OpenAI",
    "lm_studio": "LM Studio (Local)"
}

# Get available providers
available_providers = get_available_providers()

# Create selectbox options - just use the available providers directly
provider_options = []
for code in available_providers:
    display_name = provider_display_map.get(code, code)
    provider_options.append(display_name)

if provider_options:
    # Find current selection
    current_display = provider_display_map.get(current_provider, "LM Studio (Local)")
    current_index = provider_options.index(current_display) if current_display in provider_options else 0
    
    selected_provider_display = st.sidebar.selectbox(
        "Select LLM Provider",
        provider_options,
        index=current_index,
        help="Choose which LLM provider to use for AI operations"
    )
    
    # Reverse map display name to code
    reverse_map = {v: k for k, v in provider_display_map.items()}
    selected_provider = reverse_map.get(selected_provider_display, "lm_studio")
    
    # Update environment variable if changed
    if selected_provider != current_provider:
        os.environ["LLM_PROVIDER"] = selected_provider
        st.sidebar.success(f"✅ Switched to {selected_provider_display}")
    
    # Show provider-specific info
    if selected_provider == "openai":
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        st.sidebar.caption(f"Model: {model_name}")
    elif selected_provider == "lm_studio":
        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        st.sidebar.caption(f"Server: {base_url}")
        
        # Try to get the actual loaded model name
        try:
            import requests
            models_url = base_url.replace("/v1", "") + "/v1/models"
            
            response = requests.get(models_url, timeout=2)
            if response.status_code == 200:
                models_data = response.json()
                if models_data.get("data") and len(models_data["data"]) > 0:
                    loaded_model = models_data["data"][0].get("id", "Unknown")
                    st.sidebar.success(f"✅ Model: `{loaded_model}`")
                else:
                    st.sidebar.warning("⚠️ No model loaded")
            else:
                st.sidebar.caption("💡 Ensure LM Studio is running with a model loaded")
        except requests.exceptions.RequestException:
            st.sidebar.warning("⚠️ Cannot connect to LM Studio")
        except Exception:
            st.sidebar.caption("💡 Ensure LM Studio is running with a model loaded")
else:
    st.sidebar.warning("⚠️ No LLM provider configured. Please set up your .env file.")

st.sidebar.divider()
st.sidebar.info(f"Session ID: {id(st.session_state)}")

if st.sidebar.button("Clear Session"):
    reset_session_state()
    st.rerun()
