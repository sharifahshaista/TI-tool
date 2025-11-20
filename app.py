""" # type: ignore
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
from pathlib import Path
from datetime import datetime
import logging
from streamlit_agraph import agraph, Node, Edge, Config
from dotenv import load_dotenv

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
    ["Web Search", "Web Crawler", "LLM Extraction", "Summarization", "Database", "RAG", "About", "LinkedIn Home Feed Monitor"],
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
                    progress_callback=processing_progress_callback
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

    # RAG demo moved to its own page ("RAG") — open the RAG page from the sidebar to use it.

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
                    if row.get('Start-up') and str(row['Start-up']) != 'N/A':
                        tech_fields.append(f"Start-up: {row['Start-up']}")
                    
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
                        startup = str(row.get('Start-up', ''))
                        
                        tech_intel_text = f"Indicator: {indicator}\n\n"
                        if dimension:
                            tech_intel_text += f"Dimension: {dimension}\n"
                        if tech:
                            tech_intel_text += f"Tech: {tech}\n"
                        if trl:
                            tech_intel_text += f"TRL: {trl}\n"
                        if startup and startup != 'N/A':
                            tech_intel_text += f"Start-up: {startup}\n"
                        
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
        - **Inline Editing**: Click any cell to edit its content directly in the table
        - **Row Selection**: Use checkboxes to select multiple rows to view details
        - **Save Changes**: After editing cells, click 'Save Changes' to persist modifications
        - **Search & Filter**: Use the search box and column filters to find specific entries
        - **Export**: Download filtered or complete database as CSV or Excel
        
        **Tips:**
        - Single-click a cell to start editing
        - Press Enter or Tab to move to the next cell
        - All changes are saved back to the original CSV and JSON files
        - Changes are automatically synced to S3 if configured
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
            
            # Merge date and pubDate columns into a single 'date' column
            if 'pubDate' in combined_df.columns and 'date' in combined_df.columns:
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
    
    # Text search
    search_query = st.text_input("🔎 Search in database", placeholder="Enter keywords...")
    
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
    
    # Determine which columns will be displayed (exclude hidden columns)
    exclude_cols = ['source_file', 'processed_date', 'content', 'file', 'file_location', 
                   'filename', 'file_name', 'folder', 'filepath', 'Source', 'source', 'author']
    display_columns = [col for col in filtered_df.columns if col not in exclude_cols]
    
    if search_query:
        # Search only across columns that will be displayed to the user
        mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        for col in display_columns:
            mask |= filtered_df[col].astype(str).str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    st.info(f"Showing {len(filtered_df)} of {len(combined_df)} entries")
    
    st.divider()
    
    # Display results with AgGrid
    st.subheader("📋 Results")
    
    if len(filtered_df) == 0:
        st.warning("No entries match your filters.")
    else:
        # Prepare dataframe for display - exclude specified columns
        exclude_cols = ['source_file', 'processed_date', 'content', 'file', 'file_location', 
                       'filename', 'file_name', 'folder', 'filepath', 'Source', 'source', 'author', 'categories']
        display_df = filtered_df.drop(columns=[col for col in exclude_cols if col in filtered_df.columns])
        
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
            editable=True  # Enable editing for all columns by default
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
                editable=True
            )
        
        if 'Indicator' in display_df.columns:
            gb.configure_column(
                'Indicator', 
                headerName='Summary/Indicator', 
                width=150, 
                wrapText=True, 
                enableCellTextSelection=True,
                editable=True
            )
        
        if 'title' in display_df.columns:
            gb.configure_column(
                'title', 
                headerName='Title', 
                width=100, 
                wrapText=True, 
                enableCellTextSelection=True,
                editable=True
            )
        
        if 'date' in display_df.columns:
            gb.configure_column(
                'date', 
                headerName='Date', 
                width=100, 
                enableCellTextSelection=True,
                editable=True
            )
        
        # Configure selection
        gb.configure_selection(selection_mode='multiple', use_checkbox=True)
        
        # Enable text selection and editing
        gb.configure_grid_options(
            enableCellTextSelection=True, 
            ensureDomOrder=True,
            suppressRowClickSelection=True,  # Prevent row selection on cell click (for editing)
            singleClickEdit=True  # Enable single click to edit
        )
        
        gridOptions = gb.build()
        
        # Add custom CSS for URL wrapping
        gridOptions['defaultColDef']['wrapText'] = True
        gridOptions['defaultColDef']['autoHeight'] = True
        gridOptions['defaultColDef']['enableCellTextSelection'] = True
        gridOptions['defaultColDef']['editable'] = True
        
        # Display AgGrid
        grid_response = AgGrid(
            display_df,
            gridOptions=gridOptions,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.VALUE_CHANGED,  # Changed to capture value changes
            fit_columns_on_grid_load=False,
            theme='streamlit',  # can be 'streamlit' or 'streamlit-dark'
            width=800,
            height=800,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False,
            reload_data=False  # Don't reload data on every interaction
        )
        
        # Get edited data
        edited_df = pd.DataFrame(grid_response['data'])
        
        # Check if data was modified
        data_modified = False
        if not display_df.equals(edited_df):
            data_modified = True
            st.info("⚠️ Data has been modified. Click 'Save Changes' to persist changes to files.")
        
        # Save changes button
        if data_modified:
            col_save1, col_save2, col_save3 = st.columns([1, 1, 2])
            with col_save1:
                if st.button("💾 Save Changes", type="primary", use_container_width=True):
                    try:
                        # Map edited data back to original dataframe using URL as key
                        if 'url' not in edited_df.columns:
                            st.error("Cannot save: URL column is required for tracking changes")
                        else:
                            files_modified = set()
                            rows_updated = 0
                            
                            # Restore the excluded columns to edited_df from filtered_df
                            # First, create a mapping from display index back to filtered_df
                            edited_df_with_metadata = edited_df.copy()
                            
                            # For each CSV file, update matching rows
                            for csv_file in csv_files:
                                try:
                                    df = pd.read_csv(csv_file)
                                    original_len = len(df)
                                    file_modified = False
                                    
                                    if 'url' not in df.columns:
                                        continue
                                    
                                    # Update rows that match URLs in edited data
                                    for idx, edited_row in edited_df.iterrows():
                                        url = edited_row.get('url')
                                        if pd.isna(url):
                                            continue
                                        
                                        # Find matching row in CSV
                                        mask = df['url'] == url
                                        if mask.any():
                                            # Update columns that exist in both dataframes
                                            for col in edited_df.columns:
                                                if col in df.columns:
                                                    df.loc[mask, col] = edited_row[col]
                                            
                                            file_modified = True
                                            rows_updated += mask.sum()
                                    
                                    if file_modified:
                                        # Save modified CSV
                                        df.to_csv(csv_file, index=False)
                                        files_modified.add(csv_file.name)
                                        
                                        # Also update JSON if it exists
                                        json_file = csv_file.with_suffix('.json')
                                        if json_file.exists():
                                            try:
                                                df.to_json(json_file, orient='records', indent=2)
                                            except Exception as e:
                                                st.warning(f"Could not update {json_file.name}: {e}")
                                
                                except Exception as e:
                                    st.warning(f"Error processing {csv_file.name}: {e}")
                            
                            # Upload modified files to S3
                            if files_modified:
                                try:
                                    from aws_storage import get_storage
                                    s3_storage = get_storage()
                                    
                                    for file_name in files_modified:
                                        csv_path = summarised_dir / file_name
                                        s3_key = f"summarised_content/{file_name}"
                                        s3_storage.upload_file(str(csv_path), s3_key)
                                        
                                        # Also upload JSON if it exists
                                        json_name = file_name.replace('.csv', '.json')
                                        json_path = summarised_dir / json_name
                                        if json_path.exists():
                                            s3_json_key = f"summarised_content/{json_name}"
                                            s3_storage.upload_file(str(json_path), s3_json_key)
                                except Exception as e:
                                    st.warning(f"⚠️ Could not sync to S3: {e}")
                            
                            # Clear cache and refresh
                            load_all_csvs.clear()
                            st.success(f"✅ Successfully saved changes to {rows_updated} row(s) across {len(files_modified)} file(s)")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error saving changes: {e}")
            
            with col_save2:
                if st.button("🔄 Discard Changes", use_container_width=True):
                    st.rerun()
        
        # Show selection info if any rows are selected
        selected_rows = grid_response['selected_rows']
        if selected_rows is not None and len(selected_rows) > 0:
            st.success(f"✅ Selected {len(selected_rows)} row(s)")
            with st.expander("View Selected Rows"):
                st.dataframe(pd.DataFrame(selected_rows), use_container_width=True)
    
    st.divider()
    
    # Export options
    st.subheader("📥 Export Database")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export filtered results
        if len(filtered_df) > 0:
            csv_export = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Filtered Results",
                data=csv_export,
                file_name=f"filtered_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        # Export all data
        all_csv = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Complete Database",
            data=all_csv,
            file_name=f"complete_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Export to Excel (if openpyxl is installed)
        try:
            from io import BytesIO
            output = BytesIO()
            
            # Prepare dataframes for export
            export_all_df = combined_df.copy()
            if 'categories' in export_all_df.columns:
                export_all_df['categories'] = export_all_df['categories'].apply(
                    lambda x: '; '.join(x) if isinstance(x, list) else x
                )
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_all_df.to_excel(writer, index=False, sheet_name='All Data')
                if len(filtered_df) > 0 and len(filtered_df) < len(combined_df):
                    export_filtered_df = filtered_df.copy()
                    if 'categories' in export_filtered_df.columns:
                        export_filtered_df['categories'] = export_filtered_df['categories'].apply(
                            lambda x: '; '.join(x) if isinstance(x, list) else x
                        )
                    export_filtered_df.to_excel(writer, index=False, sheet_name='Filtered')
            
            st.download_button(
                label="Download as Excel",
                data=output.getvalue(),
                file_name=f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.caption("Install openpyxl for Excel export")

# RAG Page - Chat Interface with Citations
elif page == "RAG":
    st.caption("Ask questions about your documents. Responses include [Document N] citations.")
    
    # Initialize RAG-specific session state
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "rag_show_sources" not in st.session_state:
        st.session_state.rag_show_sources = True
    if "rag_show_scores" not in st.session_state:
        st.session_state.rag_show_scores = True
    if "rag_top_k" not in st.session_state:
        st.session_state.rag_top_k = 3
    if "rag_index_built" not in st.session_state:
        st.session_state.rag_index_built = False
    if "rag_llm_provider" not in st.session_state:
        st.session_state.rag_llm_provider = "azure_openai"  # Default to Azure OpenAI
    if "rag_lm_studio_url" not in st.session_state:
        st.session_state.rag_lm_studio_url = "http://127.0.0.1:1234/v1"
    if "rag_chunk_size" not in st.session_state:
        st.session_state.rag_chunk_size = 1024
    if "rag_chunk_overlap" not in st.session_state:
        st.session_state.rag_chunk_overlap = 200
    if "rag_system" not in st.session_state:
        from embeddings_rag import LlamaIndexRAG
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        # Build parameters based on provider
        rag_params = {
            "persist_dir": "rag_storage",
            "llm_provider": st.session_state.rag_llm_provider,
            "chunk_size": st.session_state.rag_chunk_size,
            "chunk_overlap": st.session_state.rag_chunk_overlap
        }
        
        if st.session_state.rag_llm_provider == "azure_openai":
            azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
            if azure_deployment:
                rag_params["azure_deployment"] = azure_deployment
            rag_params["embedding_model"] = embedding_deployment
        elif st.session_state.rag_llm_provider == "lm_studio":
            rag_params["lm_studio_base_url"] = st.session_state.rag_lm_studio_url
        
        st.session_state.rag_system = LlamaIndexRAG(**rag_params)
    
    rag_system = st.session_state.rag_system
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ RAG Configuration")
        
        # LLM Provider Selection
        st.subheader("🤖 LLM Provider")
        
        llm_provider = st.selectbox(
            "Select LLM for Response Generation",
            options=["azure_openai", "openai", "lm_studio"],
            format_func=lambda x: "Azure OpenAI" if x == "azure_openai" else ("OpenAI" if x == "openai" else "LM Studio (Local)"),
            index=0 if st.session_state.rag_llm_provider == "azure_openai" else (1 if st.session_state.rag_llm_provider == "openai" else 2),
            help="Choose which LLM to use. Azure OpenAI is used for both embeddings and generation."
        )
        
        # Show Azure OpenAI config if selected
        if llm_provider == "azure_openai":
            # Use AZURE_OPENAI_CHAT_DEPLOYMENT_NAME first, then fallback to AZURE_OPENAI_DEPLOYMENT_NAME
            default_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
            azure_deployment = st.text_input(
                "Azure Deployment Name",
                value=default_deployment,
                help="Azure OpenAI deployment name (LLM)"
            )
            
            # Check Azure OpenAI configuration
            if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
                st.success(f"✅ Azure OpenAI configured - Deployment: `{azure_deployment}`")
            else:
                st.error("❌ Azure OpenAI not configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env")
        
        # Show LM Studio URL input if selected
        elif llm_provider == "lm_studio":
            azure_deployment = None
            lm_studio_url = st.text_input(
                "LM Studio Base URL",
                value=st.session_state.rag_lm_studio_url,
                help="URL where LM Studio is running"
            )
            
            # Try to get loaded model info
            try:
                import requests
                models_url = lm_studio_url.replace("/v1", "") + "/v1/models"
                response = requests.get(models_url, timeout=2)
                if response.status_code == 200:
                    models_data = response.json()
                    if models_data.get("data") and len(models_data["data"]) > 0:
                        loaded_model = models_data["data"][0].get("id", "Unknown")
                        st.success(f"✅ Connected - Model: `{loaded_model}`")
                    else:
                        st.warning("⚠️ No model loaded in LM Studio")
                else:
                    st.warning(f"⚠️ Could not connect (Status: {response.status_code})")
            except requests.exceptions.RequestException:
                st.error("❌ Cannot connect to LM Studio. Make sure it's running.")
            except Exception:
                st.info("💡 Make sure LM Studio is running with a model loaded")
        else:
            # OpenAI
            azure_deployment = None
            lm_studio_url = st.session_state.rag_lm_studio_url
            if os.getenv("OPENAI_API_KEY"):
                st.success("✅ OpenAI API key configured")
            else:
                st.error("❌ OpenAI API key not found in .env")
        
        # Update RAG system if provider changed
        if llm_provider != st.session_state.rag_llm_provider or (llm_provider == "lm_studio" and lm_studio_url != st.session_state.rag_lm_studio_url):
            if st.button("🔄 Apply LLM Settings", type="primary", use_container_width=True):
                st.session_state.rag_llm_provider = llm_provider
                if llm_provider == "lm_studio":
                    st.session_state.rag_lm_studio_url = lm_studio_url
                
                # Reinitialize RAG system with new settings
                from embeddings_rag import LlamaIndexRAG
                
                embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
                
                if llm_provider == "azure_openai":
                    st.session_state.rag_system = LlamaIndexRAG(
                        persist_dir="rag_storage",
                        llm_provider=llm_provider,
                        azure_deployment=azure_deployment,
                        embedding_model=embedding_deployment,
                        chunk_size=st.session_state.rag_chunk_size,
                        chunk_overlap=st.session_state.rag_chunk_overlap
                    )
                elif llm_provider == "lm_studio":
                    st.session_state.rag_system = LlamaIndexRAG(
                        persist_dir="rag_storage",
                        llm_provider=llm_provider,
                        lm_studio_base_url=lm_studio_url,
                        embedding_model=embedding_deployment,
                        chunk_size=st.session_state.rag_chunk_size,
                        chunk_overlap=st.session_state.rag_chunk_overlap
                    )
                else:  # openai
                    st.session_state.rag_system = LlamaIndexRAG(
                        persist_dir="rag_storage",
                        llm_provider=llm_provider,
                        embedding_model=embedding_deployment,
                        chunk_size=st.session_state.rag_chunk_size,
                        chunk_overlap=st.session_state.rag_chunk_overlap
                    )
                
                # Show which provider is being used for embeddings
                embedding_provider = os.getenv("EMBEDDING_PROVIDER", llm_provider)
                st.success(f"✓ Switched to {llm_provider.upper()} for response generation")
                st.info(f"ℹ️ Embeddings use {embedding_provider.upper()}")
                st.rerun()
        
        st.caption("💡 **Note:** Embeddings use Azure OpenAI (text-embedding-3-large)")
        
        # Show current configuration
        config_status = "🤖 **Current Setup:**\n"
        config_status += f"- Embeddings: Azure OpenAI ({os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-large')})\n"
        config_status += f"- LLM Generation: {st.session_state.rag_llm_provider.upper()}"
        if st.session_state.rag_llm_provider == "lm_studio":
            config_status += f"\n- LM Studio URL: {st.session_state.rag_lm_studio_url}"
        st.info(config_status)
        
        st.divider()
        
        # Chunk size configuration
        st.subheader("📝 Text Chunking")
        
        if "rag_chunk_size" not in st.session_state:
            st.session_state.rag_chunk_size = 2048
        if "rag_chunk_overlap" not in st.session_state:
            st.session_state.rag_chunk_overlap = 400
        
        chunk_size = st.number_input(
            "Chunk Size (tokens)",
            min_value=256,
            max_value=4096,
            value=st.session_state.rag_chunk_size,
            step=128,
            help="Size of text chunks for embedding. Smaller chunks = more precise but more embeddings. Larger chunks = more context but less precise."
        )
        
        chunk_overlap = st.number_input(
            "Chunk Overlap (tokens)",
            min_value=0,
            max_value=512,
            value=st.session_state.rag_chunk_overlap,
            step=50,
            help="Overlap between consecutive chunks. Helps maintain context across chunk boundaries."
        )
        
        # Apply chunk size changes
        if chunk_size != st.session_state.rag_chunk_size or chunk_overlap != st.session_state.rag_chunk_overlap:
            if st.button("🔄 Apply Chunking Settings", type="secondary", use_container_width=True):
                st.session_state.rag_chunk_size = chunk_size
                st.session_state.rag_chunk_overlap = chunk_overlap
                
                # Reinitialize RAG system with new chunk settings
                from embeddings_rag import LlamaIndexRAG
                
                embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
                
                rag_params = {
                    "persist_dir": "rag_storage",
                    "llm_provider": st.session_state.rag_llm_provider,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
                
                if st.session_state.rag_llm_provider == "azure_openai":
                    rag_params["azure_deployment"] = azure_deployment
                    rag_params["embedding_model"] = embedding_deployment
                elif st.session_state.rag_llm_provider == "lm_studio":
                    rag_params["lm_studio_base_url"] = st.session_state.rag_lm_studio_url
                    rag_params["embedding_model"] = embedding_deployment
                else:  # openai
                    rag_params["embedding_model"] = embedding_deployment
                
                st.session_state.rag_system = LlamaIndexRAG(**rag_params)
                
                st.success(f"✓ Updated chunking: {chunk_size} tokens with {chunk_overlap} overlap")
                st.warning("⚠️ Note: Existing indexes won't be affected. Rebuild indexes to use new chunk settings.")
                st.rerun()
        
        st.caption(f"Current: {st.session_state.rag_chunk_size} tokens, {st.session_state.rag_chunk_overlap} overlap")
        
        st.divider()
        
        # Add reload button
        if st.button("🔄 Reload RAG System", help="Click if you updated the RAG code", use_container_width=True):
            # Force reload of the RAG system
            from importlib import reload
            import embeddings_rag
            reload(embeddings_rag)
            from embeddings_rag import LlamaIndexRAG
            
            embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
            
            # Build parameters based on provider
            rag_params = {
                "persist_dir": "rag_storage",
                "llm_provider": st.session_state.rag_llm_provider,
                "embedding_model": embedding_deployment,
                "chunk_size": st.session_state.rag_chunk_size,
                "chunk_overlap": st.session_state.rag_chunk_overlap
            }
            
            if st.session_state.rag_llm_provider == "azure_openai":
                # Get Azure deployment from environment (prioritize AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
                import os
                from dotenv import load_dotenv
                load_dotenv()
                azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                if azure_deployment:
                    rag_params["azure_deployment"] = azure_deployment
            elif st.session_state.rag_llm_provider == "lm_studio":
                rag_params["lm_studio_base_url"] = st.session_state.rag_lm_studio_url
            
            st.session_state.rag_system = LlamaIndexRAG(**rag_params)
            st.success("✓ RAG system reloaded!")
            st.rerun()
        
        st.divider()
        
        # Show available persisted indexes
        available_indexes = rag_system.get_available_indexes()
        
        # If no local indexes found, try to retrieve from S3
        if not available_indexes:
            try:
                from aws_storage import get_storage
                s3_storage = get_storage()
                
                with st.spinner("📥 Checking S3 for RAG indexes..."):
                    # List all ZIP files in S3 rag_embeddings prefix
                    s3_zip_files = s3_storage.list_files(prefix="rag_embeddings/", suffix=".zip")
                    
                    if s3_zip_files:
                        st.info(f"Found {len(s3_zip_files)} index(es) in S3. Downloading and extracting...")
                        
                        import zipfile
                        rag_storage_dir = Path(rag_system.persist_dir)
                        rag_storage_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Download and extract each ZIP file
                        for s3_key in s3_zip_files:
                            file_name = s3_key.split('/')[-1]
                            index_name = file_name.replace('.zip', '')
                            
                            # Download to temp location
                            temp_zip = rag_storage_dir / file_name
                            
                            if s3_storage.download_file(s3_key, str(temp_zip)):
                                # Extract ZIP file
                                try:
                                    with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                                        zip_ref.extractall(rag_storage_dir)
                                    st.success(f"✓ Downloaded and extracted: {index_name}")
                                    # Remove the ZIP file after extraction
                                    temp_zip.unlink()
                                except Exception as e:
                                    st.warning(f"⚠️ Failed to extract {file_name}: {str(e)}")
                            else:
                                st.warning(f"⚠️ Failed to download: {file_name}")
                        
                        # Refresh available indexes
                        available_indexes = rag_system.get_available_indexes()
                        
                        if available_indexes:
                            st.success(f"✅ Successfully retrieved {len(available_indexes)} index(es) from S3")
                    else:
                        st.info("No index files found in S3 rag_embeddings folder")
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve from S3: {str(e)}")
        
        if available_indexes:
            st.success(f"💾 Found {len(available_indexes)} persisted index(es)")
            
            # Multi-select for loading indexes
            st.subheader("� Active Indexes")
            
            # Initialize selected indexes in session state
            if "rag_selected_indexes" not in st.session_state:
                st.session_state.rag_selected_indexes = []
            
            index_names = [idx['index_name'] for idx in available_indexes]
            
            selected_indexes = st.multiselect(
                "Select indexes to query",
                options=index_names,
                default=st.session_state.rag_selected_indexes,
                help="Choose one or more indexes to search across",
                key="rag_multiselect"
            )
            
            if st.button("🔄 Load Selected", use_container_width=True, disabled=not selected_indexes):
                with st.spinner(f"Loading {len(selected_indexes)} index(es)..."):
                    try:
                        results = rag_system.load_multiple_indexes(selected_indexes)
                        total_docs = sum(v for v in results.values() if isinstance(v, int))
                        st.session_state.rag_selected_indexes = selected_indexes
                        st.session_state.rag_index_built = True
                        st.success(f"✓ Loaded {len(selected_indexes)} index(es) with {total_docs} total docs!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            # Show details of available indexes
            with st.expander("📦 Index Details", expanded=False):
                for idx_meta in available_indexes:
                    is_selected = idx_meta['index_name'] in st.session_state.rag_selected_indexes
                    status_icon = "✅" if is_selected else "⬜"
                    st.caption(f"{status_icon} **{idx_meta['index_name']}**")
                    st.caption(f"📄 {idx_meta['num_documents']} docs • 🕐 {idx_meta['created_at'][:10]}")
                    st.divider()
        
        # Index building section
        st.subheader("📊 Build New Index")
        
        # Get list of JSON files in summarised_content folder
        summarised_dir = Path("summarised_content")
        summarised_dir.mkdir(parents=True, exist_ok=True)
        json_files = []
        if summarised_dir.exists():
            json_files = sorted([f.name for f in summarised_dir.glob("*.json") if f.name != "history.json"])
        
        # If local folder is empty, try to retrieve from S3
        if not json_files:
            try:
                from aws_storage import get_storage
                s3_storage = get_storage()
                
                with st.spinner("📥 Checking S3 for summarized JSON files..."):
                    # List all JSON files in S3 summarised_content prefix
                    s3_json_files = s3_storage.list_files(prefix="summarised_content/", suffix=".json")
                    
                    if s3_json_files:
                        st.info(f"Found {len(s3_json_files)} JSON file(s) in S3. Downloading...")
                        
                        # Download each file
                        for s3_key in s3_json_files:
                            file_name = s3_key.split('/')[-1]
                            # Skip history.json
                            if file_name == "history.json":
                                continue
                            
                            local_path = summarised_dir / file_name
                            
                            if s3_storage.download_file(s3_key, str(local_path)):
                                json_files.append(file_name)
                                st.success(f"✓ Downloaded: {file_name}")
                            else:
                                st.warning(f"⚠️ Failed to download: {file_name}")
                        
                        if json_files:
                            json_files = sorted(json_files)
                            st.success(f"✅ Successfully retrieved {len(json_files)} JSON file(s) from S3")
                    else:
                        st.info("No JSON files found in S3 summarised_content folder")
            except Exception as e:
                st.warning(f"⚠️ Could not retrieve from S3: {str(e)}")
        
        if json_files:
            # Dropdown for selecting JSON file
            selected_file = st.selectbox(
                "Select JSON file",
                options=json_files,
                help="Choose a summarized JSON file from the folder",
                key="rag_selected_file"
            )
            data_source = str(summarised_dir / selected_file)
            st.caption(f"📁 Path: `{data_source}`")
            
            # Check if index already exists
            index_name = Path(selected_file).stem
            index_exists = (rag_system.persist_dir / index_name).exists()
            
            col1, col2 = st.columns(2)
            with col1:
                build_button_label = "🔄 Rebuild" if index_exists else "🔨 Build"
                if st.button(build_button_label, use_container_width=True):
                    with st.spinner("Building index..."):
                        try:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def progress_callback(message, current, total):
                                if total > 0:
                                    progress_bar.progress(min(current / total, 1.0))
                                status_text.text(message)
                            
                            num_docs = rag_system.build_index_from_json(
                                data_source,
                                force_rebuild=True,
                                progress_callback=progress_callback
                            )
                            
                            st.session_state.rag_index_built = True
                            progress_bar.empty()
                            status_text.empty()
                            st.success(f"✓ Built & saved {num_docs} docs!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Build failed: {e}")
            
            with col2:
                if index_exists:
                    if st.button("📂 Load", use_container_width=True):
                        with st.spinner("Loading..."):
                            try:
                                num_docs = rag_system.load_index(index_name)
                                st.session_state.rag_index_built = True
                                st.success(f"✓ Loaded {num_docs} docs!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Load failed: {e}")
        else:
            # Fallback to text input if no files found
            st.warning("No JSON files found in summarised_content folder")
            data_source = st.text_input(
                "Enter JSON file path manually",
                value="summarised_content/thinkgeoenergy_20251028.json",
                help="Path to your summarized JSON file",
                key="rag_data_source"
            )
            
            if st.button("🔨 Build Index", use_container_width=True):
                with st.spinner("Building index..."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def progress_callback(message, current, total):
                            if total > 0:
                                progress_bar.progress(min(current / total, 1.0))
                            status_text.text(message)
                        
                        # Show initial info
                        status_text.info("⏳ Initializing... This may take a few minutes depending on the number of documents.")
                        
                        num_docs = rag_system.build_index_from_json(
                            data_source,
                            force_rebuild=True,
                            progress_callback=progress_callback
                        )
                        
                        st.session_state.rag_index_built = True
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✓ Built & saved {num_docs} docs!")
                        st.rerun()
                        
                    except Exception as e:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"Build failed: {e}")
                        st.exception(e)  # Show full traceback
        
        # Index status
        if st.session_state.rag_index_built and st.session_state.rag_selected_indexes:
            st.success(f"✓ {len(st.session_state.rag_selected_indexes)} index(es) loaded")
            st.caption(f"Active: {', '.join(st.session_state.rag_selected_indexes)}")
        elif st.session_state.rag_index_built:
            st.success("✓ Index is ready")
        else:
            st.warning("⚠️ Please build or load index(es) first")
        
        st.divider()
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        st.session_state.rag_top_k = st.slider(
            "Top-K documents",
            min_value=1,
            max_value=10,
            value=st.session_state.rag_top_k,
            help="Number of documents to retrieve for each query"
        )
        
        st.divider()
        
        # Display options
        st.subheader("👁️ Display Options")
        st.session_state.rag_show_sources = st.checkbox(
            "Show retrieved sources",
            value=st.session_state.rag_show_sources,
            help="Display metadata of retrieved documents"
        )
        st.session_state.rag_show_scores = st.checkbox(
            "Show similarity scores",
            value=st.session_state.rag_show_scores,
            help="Display cosine similarity scores"
        )
        
        st.divider()
        
        # Statistics
        st.subheader("📈 Statistics")
        st.metric("Total Messages", len(st.session_state.rag_messages))
        st.metric("User Queries", len([m for m in st.session_state.rag_messages if m["role"] == "user"]))
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.rag_messages = []
            st.rerun()
        
        st.divider()
        
        # Information
        st.info("""
        **RAG Pipeline:**
        
        1. **Index** - Build embeddings using LlamaIndex
        2. **Retrieve** - Find relevant docs via vector similarity
        3. **Generate** - LLM answers with [Document N] citations
        
        💾 **Persistent Storage:** Indexes saved to `rag_storage/` - load instantly without rebuilding!
        
        📚 Citations reference the sources shown below each response.
        """)
    
    # Display chat history
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources metadata only
            if message["role"] == "assistant" and "sources" in message:
                if st.session_state.rag_show_sources and message["sources"]:
                    with st.expander("📚 Retrieved Sources", expanded=False):
                        for idx, doc in enumerate(message["sources"], 1):
                            md = doc['metadata']
                            score = doc['score']
                            
                            # Extract website name from source_index or use document number
                            if 'source_index' in md:
                                website_name = md['source_index'].rsplit('_', 1)[0]
                                header = f"[{website_name}] - Document {idx}"
                            else:
                                header = f"Document {idx}"
                            
                            st.markdown(f"### {header}")
                            if st.session_state.rag_show_scores:
                                st.markdown(f"**Score:** {score:.4f}")
                            
                            # Display metadata fields
                            metadata_display = []
                            for key in ['title', 'filename', 'date', 'dimension', 'tech', 'trl', 'startup', 'url', 'indicator']:
                                value = md.get(key)
                                if value not in [None, '', 'N/A']:
                                    if key == 'url':
                                        metadata_display.append(f"**{key.title()}:** [{value}]({value})")
                                    else:
                                        metadata_display.append(f"**{key.title()}:** {value}")
                            
                            if metadata_display:
                                st.markdown("\n\n".join(metadata_display))
                            else:
                                st.caption("(No metadata available)")
                            
                            if idx < len(message["sources"]):
                                st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if index is built
        if not st.session_state.rag_index_built:
            st.error("⚠️ Please build or load an index first using the sidebar.")
            st.stop()
        
        # Add user message to chat history
        st.session_state.rag_messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            try:
                # Query using LlamaIndex
                with st.status("🔍 Retrieving relevant documents...", expanded=True) as status:
                    st.write("Searching knowledge base...")
                    
                    # Detect query intent (if method exists)
                    if hasattr(rag_system, '_detect_query_intent'):
                        intent = rag_system._detect_query_intent(prompt)
                        if intent['sort_by_date']:
                            st.write("🗓️ Detected temporal query - sorting by date (most recent first)")
                    
                    # Check if multiple indexes are selected
                    if st.session_state.rag_selected_indexes and len(st.session_state.rag_selected_indexes) > 1:
                        # Multi-index query
                        st.write(f"Querying {len(st.session_state.rag_selected_indexes)} indexes...")
                        result = rag_system.query_multiple_indexes(
                            query=prompt,
                            index_names=st.session_state.rag_selected_indexes,
                            top_k=st.session_state.rag_top_k
                        )
                        
                        response = result['response']
                        retrieved_docs = [{
                            'metadata': node.metadata,
                            'score': node.score,
                            'text': node.text
                        } for node in result['source_nodes']]
                        
                        st.write(f"✓ Searched across: {', '.join(result['indexes_queried'])}")
                    else:
                        # Single index query
                        result = rag_system.query(
                            query_text=prompt,
                            top_k=st.session_state.rag_top_k
                        )
                        
                        response = result['response']
                        retrieved_docs = result['retrieved_docs']
                    
                    st.write(f"✓ Found {len(retrieved_docs)} relevant documents")
                    st.write("💬 Generated response with citations")
                    
                    status.update(label="✓ Complete!", state="complete", expanded=False)
                
                # Display response
                st.markdown(response)
                
                # Display sources metadata
                if st.session_state.rag_show_sources and retrieved_docs:
                    with st.expander("📚 Retrieved Sources", expanded=False):
                        for idx, doc in enumerate(retrieved_docs, 1):
                            md = doc['metadata']
                            score = doc['score']
                            
                            # Extract website name from source_index or use document number
                            if 'source_index' in md:
                                website_name = md['source_index'].rsplit('_', 1)[0]
                                header = f"[{website_name}] - Document {idx}"
                            else:
                                header = f"Document {idx}"
                            
                            st.markdown(f"### {header}")
                            if st.session_state.rag_show_scores:
                                st.markdown(f"**Score:** {score:.4f}")
                            
                            # Display metadata fields in the requested order
                            # Order: URL, Publication Date, Title, Indicator, Dimension, Tech, TRL, Start-up
                            metadata_display = []
                            
                            # URL (required field - always show first)
                            url_value = md.get('url') or md.get('URL')
                            if url_value and url_value not in [None, '', 'N/A']:
                                metadata_display.append(f"**URL:** [{url_value}]({url_value})")
                            
                            # Publication Date
                            pub_date = md.get('publication_date') or md.get('date') or md.get('Publication Date')
                            if pub_date and pub_date not in [None, '', 'N/A']:
                                metadata_display.append(f"**Publication Date:** {pub_date}")
                            
                            # Title
                            title_value = md.get('title') or md.get('Title')
                            if title_value and title_value not in [None, '', 'N/A']:
                                metadata_display.append(f"**Title:** {title_value}")
                            
                            # Indicator
                            indicator_value = md.get('Indicator') or md.get('indicator')
                            if indicator_value and indicator_value not in [None, '', 'N/A']:
                                # Truncate long indicators for readability
                                if len(indicator_value) > 300:
                                    indicator_value = indicator_value[:300] + "..."
                                metadata_display.append(f"**Indicator:** {indicator_value}")
                            
                            # Dimension
                            dimension_value = md.get('Dimension') or md.get('dimension')
                            if dimension_value and dimension_value not in [None, '', 'N/A']:
                                metadata_display.append(f"**Dimension:** {dimension_value}")
                            
                            # Tech
                            tech_value = md.get('Tech') or md.get('tech')
                            if tech_value and tech_value not in [None, '', 'N/A']:
                                metadata_display.append(f"**Tech:** {tech_value}")
                            
                            # TRL (Technology Readiness Level)
                            trl_value = md.get('TRL') or md.get('trl')
                            if trl_value and trl_value not in [None, '', 'N/A']:
                                metadata_display.append(f"**TRL:** {trl_value}")
                            
                            # Start-up
                            startup_value = md.get('Start-up') or md.get('startup') or md.get('Startup')
                            if startup_value and startup_value not in [None, '', 'N/A']:
                                metadata_display.append(f"**Start-up:** {startup_value}")
                            
                            if metadata_display:
                                st.markdown("\n\n".join(metadata_display))
                            else:
                                st.caption("(No metadata available)")
                            
                            if idx < len(retrieved_docs):
                                st.divider()
                
                # Add assistant message to chat history
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": retrieved_docs,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                import traceback
                st.code(traceback.format_exc())

## About Page
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
    - **Tech Intelligence** - Dimension, Tech, TRL, Start-up classification
    - **Smart Filtering** - Auto-removes empty content and dates >2 years old
    - **Dual Format** - Outputs both CSV and JSON to processed_data folder
    - **S3 Upload** - Both formats automatically uploaded to cloud storage

    **4. Summarization**
    - **Tech-Intel Analysis** - AI generates Indicator, Dimension, Tech, TRL, Start-up fields
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

    **6. RAG (Retrieval-Augmented Generation)**
    - **Multi-Source Indexing** - Create vector indexes from JSON files
    - **Smart Search** - Semantic search with date-based relevance sorting
    - **LLM Providers** - Support for Azure OpenAI, OpenAI, and LM Studio
    - **S3 Persistence** - Indexes automatically sync to/from AWS S3
    - **Index Management** - List, load, query, and delete indexes
    - **Cloud Restore** - Auto-download indexes from S3 if missing locally

    **7. LinkedIn Home Feed Monitor**
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
    ✅ **Vector Search** - RAG with semantic retrieval  
    ✅ **Social Intelligence** - LinkedIn network monitoring  
    ✅ **Batch Processing** - Handle large datasets efficiently  
    ✅ **History Tracking** - Complete audit trail of all operations  

    ### Technology Stack

    - **Frontend:** Streamlit with AgGrid
    - **AI Models:** Azure OpenAI (GPT-4.1-mini, text-embedding-3-large)
    - **Alternative LLMs:** OpenAI, LM Studio (local)
    - **Search Engine:** SearxNG
    - **Web Automation:** Selenium WebDriver
    - **Vector Store:** LlamaIndex with ChromaDB
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
    - **rag_embeddings/** - Vector indexes (ZIP)
    - **linkedin_data/** - LinkedIn posts (CSV, JSON)

    **Features:**
    - ✅ Automatic upload after processing
    - ✅ Download and delete via UI (LinkedIn, RAG)
    - ✅ Auto-sync for RAG indexes
    - ✅ Graceful fallback if S3 unavailable

    ### Usage Workflow

    **Complete Research Pipeline:**
    ```
    Web Search → Web Crawler → URL Filter → LLM Extraction → Summarization → Database/RAG
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

    **RAG:**
    1. Navigate to "RAG" tab
    2. Select JSON files to index
    3. Create vector index (auto-uploaded to S3)
    4. Query with natural language
    5. Get relevant documents with sources
    6. Indexes sync across machines via S3

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
    ├── rag_embeddings/            # Compressed vector indexes
    └── linkedin_data/             # LinkedIn posts CSVs/JSONs
    ```

    ### Support & Troubleshooting

    - **Logs:** Check `research.log` for detailed operation logs
    - **S3 Status:** Run `python3 check_s3_status.py` to verify bucket contents
    - **Upload Missing Files:** Use `python3 upload_linkedin_to_s3.py` for backfill
    - **RAG Issues:** Indexes auto-download from S3 if missing locally
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
            help="Automatically translate non-English posts to English using Azure OpenAI"
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
        from openai import AzureOpenAI
        
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
        
        def translate_to_english(text, azure_client):
            """Translate text to English using Azure OpenAI API"""
            if not text or not text.strip() or not azure_client:
                return text
            
            if is_english(text):
                return text
            
            try:
                response = azure_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4"),
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
        
        def parse_post(post_element, azure_client, enable_translation):
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
                
                if enable_translation and content and azure_client:
                    content = translate_to_english(content, azure_client)
                
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
            # Initialize Azure OpenAI if translation is enabled
            azure_client = None
            if enable_translation:
                try:
                    azure_client = AzureOpenAI(
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                    )
                except Exception as e:
                    status_placeholder.warning(f"⚠️ Could not initialize Azure OpenAI: {e}\nTranslation disabled.")
            
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
                    
                    data = parse_post(post_element, azure_client, enable_translation)
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
current_provider = os.getenv("LLM_PROVIDER", "azure").lower()

# Map provider codes to display names
provider_display_map = {
    "azure": "Azure OpenAI",
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
    if selected_provider == "azure":
        model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4")
        st.sidebar.caption(f"Model: {model_name}")
    elif selected_provider == "openai":
        model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
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
