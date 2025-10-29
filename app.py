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
from pathlib import Path
from datetime import datetime
import logging
from streamlit_agraph import agraph, Node, Edge, Config

# Import existing modules
from agents.clarification import get_clarifications
from agents.serp import get_serp_queries
from agents.learn import get_learning_structured
from agents.summarise_csv import summarize_csv_file, save_summarized_csv
from config.searxng_tools import searxng_web_tool
from config.model_config import get_model
from schemas.datamodel import (
    SearchResultsCollection,
    CSVSummarizationMetadata,
    CSVSummarizationHistory,
)
from agents import quick_start_post_processing

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

# Initialize session state
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
                results = await searxng_web_tool(None, query)
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
st.title("TI Tool")
st.markdown("AI-powered tool equipped with discovery of sources, web-crawling, post-processing into structured data for a combined database and chatbot")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Mode",
    ["Web Search", "Web Crawler", "Post-Processing", "Summarization", "Database", "RAG", "About"],
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
    st.header("🕷️ Web Crawler")
    st.markdown("Crawl websites with intelligent strategy detection and multiple crawl modes")
    
    # Combined crawling interface
    st.subheader("Web Crawler Configuration")
    st.markdown("Configure a custom crawl for any website")
    
    # URL input
    url_input = st.text_input(
        "Website URL",
        placeholder="https://www.example.com",
        help="Enter the website URL you want to crawl (protocol will be added automatically if missing)"
    )
    
    # Auto-correct URL: add https:// if missing
    if url_input:
        url_input = url_input.strip()
        if not url_input.startswith(('http://', 'https://')):
            url_input = 'https://' + url_input
            st.info(f"🔗 Auto-corrected URL: `{url_input}`")
    
        # Strategy selection
        st.subheader("Crawl Strategy")
        strategy_method = st.radio(
            "Choose strategy selection method",
            ["Auto-Detection (Recommended)", "Manual Selection"],
            help="Auto-detection analyzes the site and recommends the best approach"
        )
        
        # Auto-detection logic
        if strategy_method == "Auto-Detection (Recommended)":
            st.markdown("The system will analyze your URL and recommend the best crawling strategy.")
            
            if url_input:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button("🔍 Analyze URL & Recommend Strategy", type="primary"):
                        with st.spinner("🔍 Testing crawl strategies... This may take 15-30 seconds..."):
                            try:
                                from agents.crawl_strategy_detector import CrawlStrategyDetector
                                import asyncio
                                
                                detector = CrawlStrategyDetector(url_input)
                                results = asyncio.run(detector.detect_best_strategy())
                                st.session_state.detection_results = results
                                st.session_state.strategy_confirmed = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Detection failed: {str(e)}")
                
                # Show detection results if available
                if st.session_state.detection_results:
                    results = st.session_state.detection_results
                    st.success("✅ Strategy Analysis Complete!")
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### 📊 Detection Results")
                        sorted_strategies = sorted(results['details'].items(), key=lambda x: x[1]['score'], reverse=True)
                        
                        for strategy_name, strategy_data in sorted_strategies:
                            score = strategy_data['score']
                            if score >= 80:
                                color = "🟢"
                                status = "Recommended"
                            elif score >= 60:
                                color = "🟡"
                                status = "Viable"
                            elif score >= 1:
                                color = "🔴"
                                status = "Fallback"
                            else:
                                color = "⚫"
                                status = "Not Viable"
                            
                            st.markdown(f"**{color} {strategy_name}**: {score}/100 - {status}")
                    
                    with col2:
                        st.markdown("### 🎯 Recommended Strategy")
                        best_strategy = max(results['details'].items(), key=lambda x: x[1]['score'])
                        st.success(f"**{best_strategy[0]}**")
                        st.metric("Score", f"{best_strategy[1]['score']}/100")
                        
                        if st.button("✅ Use This Strategy", type="primary"):
                            st.session_state.strategy_confirmed = True
                            st.session_state.selected_strategy = best_strategy[0]
                            st.rerun()
        
        # Manual selection
        else:
            st.markdown("Choose a crawling strategy manually.")
            
            crawl_mode = st.selectbox(
                "Select Crawl Mode",
            ["Simple Discovery", "Sitemap Crawl", "Pagination Crawl", "Category Crawl", "Deep Crawl (BFS)", "Deep Crawl (DFS)"],
                help="Choose the crawling strategy that best fits your website"
            )
            
            strategy_descriptions = {
            "Simple Discovery": "Basic link following. Good for simple sites and quick crawling.",
                "Sitemap Crawl": "Best for sites with XML sitemaps. Fast and comprehensive.",
                "Pagination Crawl": "Ideal for news sites with numbered pagination (page 1, 2, 3...).",
                "Category Crawl": "Good for sites with category pages and topic-based navigation.",
                "Deep Crawl (BFS)": "Breadth-first search. Good for discovering all content at each level.",
                "Deep Crawl (DFS)": "Depth-first search. Good for following content chains."
            }
            st.info(strategy_descriptions[crawl_mode])
        
        # Crawl configuration
        st.subheader("Crawl Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_pages = st.number_input(
                "Max Pages",
                min_value=1,
                max_value=50000,
                value=500,
                step=10,
                help="Maximum number of pages to crawl"
            )
        
        with col2:
            max_depth = st.number_input(
                "Max Depth",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Maximum depth for deep crawl strategies (levels from start URL)"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                help="Number of pages to process in parallel"
            )
        
        with col4:
            delay = st.number_input(
                "Delay (seconds)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Delay between requests"
            )
        
        # Construct output directory path
        suggested_name = "saved_md"
        if url_input:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url_input)
                domain = parsed.netloc.replace('www.', '').replace('.com', '').replace('.org', '').replace('.net', '')
                # Clean domain name to be folder-safe
                suggested_name = ''.join(c for c in domain if c.isalnum() or c in '-_').strip('-_')
                if not suggested_name:
                    suggested_name = "saved_md"
            except:
                suggested_name = "saved_md"
        
        output_dir = f"crawled_data/{suggested_name}"
        
        # Display output location
        st.caption(f"📁 **Output Location:** `{output_dir}`")
        
        # Start crawl button and cancel option
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if url_input and st.button("🕷️ Start Crawling", type="primary", use_container_width=True):
                st.session_state.crawling_in_progress = True
                st.session_state.crawl_cancel_requested = False
                st.session_state.crawl_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0, 'start_time': time.time()}
                
                # Create progress containers
                progress_container = st.container()
                status_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    time_text = st.empty()
                
                with status_container:
                    status_text = st.empty()
                
                # Start crawling process with progress tracking
                try:
                    from agents.web_crawler import EnhancedWebCrawler, CrawlMode
                    
                    # Map strategy names to crawl modes
                    strategy_mapping = {
                        "Pagination Crawl": CrawlMode.PAGINATION,
                        "Sitemap Crawl": CrawlMode.SITEMAP,
                        "Category Crawl": CrawlMode.CATEGORY,
                        "Simple Discovery": CrawlMode.SIMPLE,
                        "Deep Crawl (BFS)": CrawlMode.DEEP_BFS,
                        "Deep Crawl (DFS)": CrawlMode.DEEP_DFS
                    }
                    
                    # Determine crawl mode based on strategy selection method
                    if strategy_method == "Auto-Detection (Recommended)":
                        # Use the recommended strategy from detection results
                        if st.session_state.detection_results:
                            recommended_strategy = st.session_state.detection_results['recommended_strategy']
                            crawl_mode = strategy_mapping.get(recommended_strategy, CrawlMode.SIMPLE)
                        else:
                            st.error("Please run strategy detection first!")
                            st.stop()
                    else:
                        # Use manually selected strategy
                        crawl_mode = strategy_mapping.get(crawl_mode, CrawlMode.SIMPLE)
                
                    def progress_callback(message, current, total):
                        """Progress callback for real-time updates"""
                        # Handle user cancellation
                        if st.session_state.crawl_cancel_requested:
                            try:
                                # Signal the crawler to stop gracefully
                                crawler.cancelled = True
                            except Exception:
                                pass
                            status_text.text("🛑 Cancelling crawl...")
                            return
                        if total > 0:
                            progress = current / total
                            progress_bar.progress(progress)
                            progress_text.text(f"Progress: {current}/{total} ({progress*100:.1f}%)")
                            
                            # Calculate time estimates
                            elapsed = time.time() - st.session_state.crawl_progress['start_time']
                            remaining = 0  # Initialize remaining time
                            if current > 0:
                                estimated_total = elapsed * total / current
                                remaining = max(0, estimated_total - elapsed)
                                
                                time_text.text(f"⏱️ Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining)}")
                            else:
                                time_text.text(f"⏱️ Elapsed: {format_time(elapsed)} | Remaining: Calculating...")
                            
                            status_text.text(f"🔄 {message}")
                            
                            # Update session state
                            st.session_state.crawl_progress = {
                                'current': current,
                                'total': total,
                                'elapsed': elapsed,
                                'remaining': remaining,
                                'start_time': st.session_state.crawl_progress['start_time']
                            }
                    
                    crawler = EnhancedWebCrawler(
                        start_url=url_input,
                        output_dir=output_dir
                    )
                    
                    # Show warning about Streamlit UI limitations and monitoring tips
                    monitoring_info = st.expander("📊 Real-Time Progress Monitoring", expanded=True)
                    with monitoring_info:
                        st.warning("⚠️ **UI Limitation:** Progress display may freeze during crawling. This is normal - the crawl IS running!")
                        st.info(f"**Monitor in real-time:**\n\n**Terminal command:**\n```bash\nwatch -n 1 'ls -lh \"{output_dir}\" | tail -20'\n```\n\n**Or open in Finder/Explorer:**\n`{output_dir}`")
                    
                    # Run the crawler with progress tracking
                    # Use run_async helper for better Streamlit compatibility
                    results = run_async(crawler.crawl(
                        max_pages=max_pages,
                        max_depth=max_depth,
                        mode=crawl_mode,
                        progress_callback=progress_callback,
                        cancel_callback=lambda: st.session_state.crawl_cancel_requested
                    ))
                    st.session_state.crawl_results = results
                    st.session_state.crawling_in_progress = False
                    
                    # Show completion message
                    progress_bar.progress(1.0)
                    progress_text.text("✅ Crawl Complete!")
                    total_time = time.time() - st.session_state.crawl_progress['start_time']
                    time_text.text(f"⏱️ Total time: {format_time(total_time)}")
                    status_text.text(f"🎉 Successfully crawled {results.pages_saved} pages")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Crawling failed: {str(e)}")
                    st.session_state.crawling_in_progress = False
    
        with col2:
            stop_clicked = st.button("🛑 Stop Crawl", type="secondary", use_container_width=True, disabled=not st.session_state.crawling_in_progress)
            if stop_clicked:
                st.session_state.crawl_cancel_requested = True
                st.warning("Cancelling crawl...")
    
    # Results section
    st.divider()
    st.subheader("📊 Crawl Results")
    
    # Show progress if crawling is in progress
    if st.session_state.crawling_in_progress:
        st.markdown("### 🔄 Crawling in Progress...")
        
        progress = st.session_state.crawl_progress
        if progress['total'] > 0:
            progress_percent = progress['current'] / progress['total']
            st.progress(progress_percent)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Progress", f"{progress['current']}/{progress['total']}")
            with col2:
                st.metric("Elapsed", format_time(progress['elapsed']))
            with col3:
                st.metric("Remaining", format_time(progress['remaining']))
        else:
            st.info("🔄 Initializing crawl...")
    elif st.session_state.crawl_results:
        results = st.session_state.crawl_results
        
        # Handle both CrawlResult objects and dictionary results from quick start crawlers
        if isinstance(results, dict):
            # Quick start crawler results (dictionary format)
            pages_found = results.get('total_pages_crawled', results.get('total_urls', 0))
            pages_saved = results.get('total_pages_saved', results.get('saved_count', 0))
            failed_count = results.get('failed_count', 0)
            duration = results.get('total_duration', 0)
            success = pages_saved > 0
            message = results.get('message', f"Crawled {pages_saved} pages successfully")
            mode = "quick_start"
            failed_urls = []
            structured_data = []
        else:
            # CrawlResult object format
            pages_found = results.pages_found
            pages_saved = results.pages_saved
            failed_count = len(results.failed_urls)
            duration = results.duration
            success = results.success
            message = results.message
            mode = results.mode
            failed_urls = results.failed_urls
            structured_data = getattr(results, 'structured_data', [])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pages Found", pages_found)
        with col2:
            st.metric("Pages Saved", pages_saved)
        with col3:
            st.metric("Failed", failed_count)
        with col4:
            success_rate = (pages_saved / pages_found * 100) if pages_found > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Show detailed results
        st.subheader("Crawl Summary")
        
        # Display basic information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{duration:.2f}s")
        with col2:
            st.metric("Mode", mode)
        with col3:
            st.metric("Success", "✅" if success else "❌")
        
        # Show message
        if message:
            st.info(f"📝 {message}")
        
        # Show multi-format export information
        if structured_data:
            st.success("✅ Multi-format export completed! Files saved in:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("📁 **Markdown Files**\nSaved in main output directory")
            with col2:
                st.info("📊 **CSV Files**\nSaved in `csv/` subdirectory")
            with col3:
                st.info("📋 **JSON Files**\nSaved in `json/` subdirectory")
        
        # Show failed URLs if any
        if failed_urls:
            with st.expander(f"Failed URLs ({len(failed_urls)})"):
                for failed in failed_urls:
                    if isinstance(failed, dict):
                        st.write(f"- {failed.get('url', 'Unknown')}: {failed.get('error', 'Unknown error')}")
                    else:
                        st.write(f"- {failed}")
                    
                    # Download options
        st.subheader("📥 Download Options")
        
        # Create tabs for different download types
        download_tab1, download_tab2, download_tab3 = st.tabs(["📊 Summary", "📋 Structured Data", "📁 All Files"])
        
        with download_tab1:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Summary CSV"):
                    summary_data = {
                        "Metric": ["Pages Found", "Pages Saved", "Failed URLs", "Duration (s)", "Mode", "Success"],
                        "Value": [pages_found, pages_saved, failed_count, duration, mode, success]
                    }
                    df = pd.DataFrame(summary_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Summary CSV",
                        data=csv,
                        file_name=f"crawl_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📥 Download Summary JSON"):
                    json_data = {
                        "pages_found": pages_found,
                        "pages_saved": pages_saved,
                        "failed_urls": failed_urls,
                        "duration": duration,
                        "mode": mode,
                        "success": success,
                        "message": message
                    }
                    json_str = json.dumps(json_data, indent=2)
                    st.download_button(
                        label="Download Summary JSON",
                        data=json_str,
                        file_name=f"crawl_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
        with download_tab2:
            if structured_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create CSV from structured data
                    df_structured = pd.DataFrame(structured_data)
                    csv_structured = df_structured.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Structured Data CSV",
                        data=csv_structured,
                        file_name=f"crawl_structured_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Create JSON from structured data
                    json_structured = json.dumps(structured_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📋 Download Structured Data JSON",
                        data=json_structured,
                        file_name=f"crawl_structured_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                # Show preview of structured data
                st.subheader("📋 Structured Data Preview")
                st.dataframe(df_structured.head(10), use_container_width=True)
            else:
                st.info("No structured data available for download.")
        
        with download_tab3:
            st.info("📁 **File Locations:**")
            st.markdown("""
            - **Markdown files**: `crawled_data/[domain_name]/`
            - **CSV files**: `crawled_data/[domain_name]/csv/`
            - **JSON files**: `crawled_data/[domain_name]/json/`
            """)
            
            if st.button("📁 Open Output Directory"):
                import subprocess
                import platform
                
                # Get the output directory from the crawler
                if structured_data:
                    # Extract domain from first URL
                    first_url = structured_data[0].get('url', '')
                    if first_url:
                        from urllib.parse import urlparse
                        parsed = urlparse(first_url)
                        domain = parsed.netloc.replace('www.', '').replace('.com', '').replace('.org', '').replace('.net', '')
                        output_dir = f"crawled_data/{domain}"
                        
                        try:
                            if platform.system() == "Windows":
                                subprocess.run(["explorer", output_dir], check=True)
                            elif platform.system() == "Darwin":  # macOS
                                subprocess.run(["open", output_dir], check=True)
                            else:  # Linux
                                subprocess.run(["xdg-open", output_dir], check=True)
                            st.success(f"Opened directory: {output_dir}")
                        except Exception as e:
                            st.error(f"Could not open directory: {e}")
                            st.info(f"Manual path: {output_dir}")
                elif isinstance(results, dict) and 'output_dir' in results:
                    # Quick start crawler results have output_dir
                    output_dir = results['output_dir']
                    try:
                        if platform.system() == "Windows":
                            subprocess.run(["explorer", output_dir], check=True)
                        elif platform.system() == "Darwin":  # macOS
                            subprocess.run(["open", output_dir], check=True)
                        else:  # Linux
                            subprocess.run(["xdg-open", output_dir], check=True)
                        st.success(f"Opened directory: {output_dir}")
                    except Exception as e:
                        st.error(f"Could not open directory: {e}")
                        st.info(f"Manual path: {output_dir}")
                else:
                    st.warning("No output directory information available.")
        
        # Option to clear results
        if st.button("Clear Results"):
            st.session_state.crawl_results = None
            st.rerun()
    else:
        st.info("No crawl results available. Run a crawl first.")
    
    # Initialize session state for detection results
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None


# Post-Processing Page
elif page == "Post-Processing":
    st.header("📄 Post-Processing")
    st.markdown("Extract structured metadata from crawled markdown files")
    
    # Create main workflow tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Quick Start", "⚙️ Custom Config", "📊 Results"])
    
    with tab1:
        st.subheader("Quick Start - Pre-configured Processing")
        st.markdown("Choose from optimized processors for popular news sites")
        
        # Preset selection using available presets from module
        available_presets = quick_start_post_processing.get_available_presets()
        preset = st.selectbox(
            "Select a news site to process",
            available_presets,
            help="Pre-configured processors with optimized patterns"
        )
        
        # Show preset info
        preset_info = quick_start_post_processing.get_preset(preset)
        if preset_info:
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.info(f"📰 **{preset_info['description']}**")
            with col2:
                if preset_info['blocked_urls']:
                    st.caption(f"🚫 {len(preset_info['blocked_urls'])} blocked URLs")
                else:
                    st.caption("✓ No blocked URLs")
            with col3:
                if preset_info.get('date_filter_months'):
                    st.caption(f"📅 Last {preset_info['date_filter_months']} months")
            
        # Folder selection
        st.subheader("Select Data Source")
        
        # Get list of available crawled folders
        crawled_data_path = Path("crawled_data")
        available_folders = []
        
        if crawled_data_path.exists():
            available_folders = [
                f.name for f in crawled_data_path.iterdir() 
                if f.is_dir() and not f.name.startswith('.')
            ]
        
        if available_folders:
            # Sort folders by modification time (newest first)
            available_folders.sort(
                key=lambda x: (crawled_data_path / x).stat().st_mtime, 
                reverse=True
            )
            
            markdown_folder_name = st.selectbox(
                "Select Crawled Folder",
                options=available_folders,
                help="Choose a folder from crawled_data/ directory",
                key="quick_start_folder"
            )
            markdown_folder = f"crawled_data/{markdown_folder_name}"
            
            # Show folder info
            folder_path = crawled_data_path / markdown_folder_name
            md_count = len(list(folder_path.rglob('*.md')))
            st.caption(f"📁 {md_count} markdown files in this folder")
        else:
            st.warning("No crawled folders found. Please run the Web Crawler first.")
            markdown_folder = st.text_input(
                "Markdown Folder (manual entry)",
                value="saved_md",
                help="Folder containing markdown files from web crawler",
                key="manual_folder"
            )
        
        # Output settings
        col1, col2 = st.columns(2)
        
        with col1:
            output_folder = st.text_input(
                "Output Folder",
                value="processed_data",
                help="Folder to save processed CSV/JSON files",
                key="quick_start_output_folder"
            )
        
        with col2:
            date_filter = st.selectbox(
                "Date Filter",
                options=[None, 1, 3, 6, 12, 24],
                format_func=lambda x: "No filter" if x is None else f"Last {x} months",
                help="Filter articles by publication date",
                key="quick_start_date_filter"
            )
        
        # Start processing button
        if st.button("🚀 Start Quick Processing", type="primary", use_container_width=True):
            markdown_path = Path(markdown_folder)
            
            if not markdown_path.exists():
                st.error(f"Folder '{markdown_folder}' does not exist")
            else:
                # Check for markdown files
                md_files = list(markdown_path.rglob('*.md'))
                
                if not md_files:
                    st.error(f"No markdown files found in '{markdown_folder}'")
                else:
                    st.info(f"Found {len(md_files)} markdown files")
                    
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
                        from agents import quick_start_post_processing
                        import asyncio
                        
                        # Set processing state
                        st.session_state.processing_in_progress = True
                        st.session_state.processing_progress = {'current': 0, 'total': 0, 'elapsed': 0, 'remaining': 0}
                        
                        def processing_progress_callback(message, current, total):
                            """Progress callback for processing updates"""
                            if total > 0:
                                progress = current / total
                                progress_bar.progress(progress)
                                progress_text.text(f"Processing: {current}/{total} ({progress*100:.1f}%)")
                                
                                # Calculate time estimates
                                elapsed = time.time() - st.session_state.get('processing_start_time', time.time())
                                remaining = 0  # Initialize remaining time
                                if current > 0:
                                    estimated_total = elapsed * total / current
                                    remaining = max(0, estimated_total - elapsed)
                                    
                                    time_text.text(f"⏱️ Elapsed: {format_time(elapsed)} | Remaining: {format_time(remaining)}")
                                else:
                                    time_text.text(f"⏱️ Elapsed: {format_time(elapsed)} | Remaining: Calculating...")
                                
                                # Update session state for Results tab
                                st.session_state.processing_progress = {
                                    'current': current,
                                    'total': total,
                                    'elapsed': elapsed,
                                    'remaining': remaining
                                }
                                
                                status_text.text(f"🔄 {message}")
                        
                        # Set processing start time
                        st.session_state.processing_start_time = time.time()
                        
                        # Run the preset processor with progress tracking
                        results = asyncio.run(quick_start_post_processing.run_preset_processing(
                            preset, 
                            markdown_folder, 
                            output_folder, 
                            date_filter,
                            progress_callback=processing_progress_callback
                        ))
                        
                        st.session_state.csv_processed_df = results['dataframe']
                        st.session_state.csv_metadata = results['metadata']
                        
                        # Clear processing state
                        st.session_state.processing_in_progress = False
                        
                        # Show completion message
                        progress_bar.progress(1.0)
                        progress_text.text("✅ Processing Complete!")
                        total_time = time.time() - st.session_state.processing_start_time
                        time_text.text(f"⏱️ Total time: {format_time(total_time)}")
                        status_text.text(f"🎉 Successfully processed {len(results['dataframe'])} articles")
                        
                        # Show completion statistics
                        st.success("✅ Quick processing complete!")
                        
                        # Display processing stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Files", len(results['dataframe']))
                        with col2:
                            st.metric("Duration", f"{total_time:.1f}s")
                        with col3:
                            st.metric("Articles/sec", f"{len(results['dataframe'])/total_time:.1f}")
                        with col4:
                            if 'success' in results['dataframe'].columns:
                                success_count = len(results['dataframe'][results['dataframe']['success'] == True])
                                success_rate = (success_count / len(results['dataframe'])) * 100
                            else:
                                # If no success column, assume all processed files were successful
                                success_rate = 100
                            st.metric("Success Rate", f"{success_rate:.1f}%")
                            
                        # Show completion banner
                        st.markdown("""
                        ---
                        ### ✅ Processing Complete
                        The quick start processing has finished successfully. You can view the results in the Results tab.
                        
                        **Next Steps:**
                        1. Switch to the Results tab to view processed data
                        2. Download results as CSV or JSON
                        3. Clear results to process another folder
                        ---
                        """)
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.session_state.processing_in_progress = False
                        st.error(f"Quick processing failed: {str(e)}")
    
    with tab2:
        st.subheader("Custom Configuration")
        st.markdown("Configure processing settings for any website")
        
        # Folder selection
        st.subheader("Data Source")
        
        # Get list of available crawled folders
        crawled_data_path = Path("crawled_data")
        available_folders = []
        
        if crawled_data_path.exists():
            available_folders = [
                f.name for f in crawled_data_path.iterdir() 
                if f.is_dir() and not f.name.startswith('.')
            ]
        
        if available_folders:
            # Sort folders by modification time (newest first)
            available_folders.sort(
                key=lambda x: (crawled_data_path / x).stat().st_mtime, 
                reverse=True
            )
            
            markdown_folder_name = st.selectbox(
                "Select Crawled Folder",
                options=available_folders,
                help="Choose a folder from crawled_data/ directory",
                key="custom_config_folder"
            )
            markdown_folder = f"crawled_data/{markdown_folder_name}"
            
            # Show folder info
            folder_path = crawled_data_path / markdown_folder_name
            md_count = len(list(folder_path.rglob('*.md')))
            st.caption(f"📁 {md_count} markdown files in this folder")
        else:
            st.warning("No crawled folders found. Please run the Web Crawler first.")
            markdown_folder = st.text_input(
                "Markdown Folder (manual entry)",
                value="saved_md",
                help="Folder containing markdown files from web crawler",
                key="manual_folder"
            )
        
        # Processing settings
        st.subheader("Processing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            output_folder = st.text_input(
                "Output Folder",
                value="processed_data",
                help="Folder to save processed CSV/JSON files",
                key="custom_config_output_folder"
            )
            
            date_filter = st.selectbox(
                "Date Filter",
                options=[None, 1, 3, 6, 12, 24],
                format_func=lambda x: "No filter" if x is None else f"Last {x} months",
                help="Filter articles by publication date",
                key="custom_config_date_filter"
            )
        
        with col2:
            # Check if OpenAI/Azure OpenAI API key is available
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            has_openai_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))
            
            use_ai_detection = st.checkbox(
                "AI Pattern Detection",
                value=has_openai_key,
                disabled=not has_openai_key,
                help="Use AI to detect metadata patterns"
            )
            
            if not has_openai_key:
                st.warning("⚠️ OpenAI/Azure OpenAI API key not found. AI pattern detection is disabled.")
            elif not use_ai_detection:
                st.info("ℹ️ AI Pattern Detection is disabled. Fallback patterns will be used.")
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                saved_patterns = st.text_input(
                    "Saved Patterns File (Optional)",
                    value="",
                    help="Path to previously saved pattern JSON file"
                )
                
                st.info("💡 Patterns are auto-saved after first run")
            
            with col2:
                filter_category_pages = st.checkbox(
                    "Filter Category Pages",
                    value=True,
                    help="Auto-detect and skip category/landing pages"
                )
                
                blocked_urls_text = st.text_area(
                    "Additional Blocked URLs (one per line)",
                    value="",
                    height=100,
                    help="Add specific URLs to skip"
                )
        
        # Start processing button
        if st.button("⚙️ Start Custom Processing", type="primary", use_container_width=True):
            markdown_path = Path(markdown_folder)
            
            if not markdown_path.exists():
                st.error(f"Folder '{markdown_folder}' does not exist")
            else:
                # Check for markdown files
                md_files = list(markdown_path.rglob('*.md'))
                
                if not md_files:
                    st.error(f"No markdown files found in '{markdown_folder}'")
                else:
                    st.info(f"Found {len(md_files)} markdown files")
                    
                    # Start custom processing
                    try:
                        from agents.markdown_post_processor import MarkdownPostProcessor
                        import asyncio
                        
                        # Prepare blocked URLs
                        blocked_set = set(u.strip() for u in blocked_urls_text.split('\n') if u.strip()) if blocked_urls_text else set()

                        # Create processor with supported settings
                        processor = MarkdownPostProcessor(
                            blocked_urls=blocked_set
                        )

                        # Optionally detect and add category pages to blocked list
                        if filter_category_pages:
                            category_urls = processor.detect_category_pages(Path(markdown_folder))
                            if category_urls:
                                processor.add_blocked_urls(category_urls)

                        # Prepare pattern file and auto-detect flag
                        pattern_file = Path(saved_patterns) if saved_patterns else None

                        # Run the custom processor (synchronous)
                        df, stats = processor.process_folder(
                            folder_path=Path(markdown_folder), 
                            output_dir=Path(output_folder), 
                            date_filter_months=date_filter,
                            auto_detect=use_ai_detection,
                            pattern_file=pattern_file
                        )
                        
                        st.session_state.csv_processed_df = df
                        st.session_state.csv_metadata = stats
                        
                        # Show completion statistics
                        st.success("✅ Custom processing complete!")
                        
                        # Display processing stats
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Files", stats.get('total_files', len(df)))
                        with col2:
                            st.metric("Processed", stats.get('processed', len(df)))
                        with col3:
                            st.metric("Skipped", stats.get('skipped_error', 0))
                        with col4:
                            success_rate = stats.get('processed', len(df)) / stats.get('total_files', len(df)) * 100
                            st.metric("Success Rate", f"{success_rate:.1f}%")
                        
                        # Show detailed completion banner
                        st.markdown("""
                        ---
                        ### ✅ Processing Complete
                        Custom configuration processing has finished successfully. You can view the results in the Results tab.
                        
                        **Processing Details:**
                        - All files have been processed with your custom configuration
                        - Metadata has been extracted using specified patterns
                        - Results are ready for export
                        
                        **Next Steps:**
                        1. Switch to the Results tab to view processed data
                        2. Download results as CSV or JSON
                        3. Clear results to process another folder
                        ---
                        """)
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Custom processing failed: {str(e)}")
    
    with tab3:
        st.subheader("Processing Results")
        
        # Show progress if processing is in progress
        if st.session_state.get('processing_in_progress', False):
            st.markdown("### 🔄 Processing in Progress...")
            
            progress = st.session_state.get('processing_progress', {})
            if progress.get('total', 0) > 0:
                progress_percent = progress.get('current', 0) / progress['total']
                st.progress(progress_percent)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Progress", f"{progress.get('current', 0)}/{progress['total']}")
                with col2:
                    st.metric("Elapsed", format_time(progress.get('elapsed', 0)))
                with col3:
                    st.metric("Remaining", format_time(progress.get('remaining', 0)))
            else:
                st.info("🔄 Initializing processing...")
        
        if st.session_state.csv_processed_df is not None:
            st.success("✅ Processing Complete!")
            
            # Show statistics (robust to missing 'success' column)
            df_stats = st.session_state.csv_processed_df
            meta_stats = st.session_state.get('csv_metadata', {}) or {}
            col1, col2, col3, col4 = st.columns(4)
            
            if 'success' in df_stats.columns:
                total = len(df_stats)
                successes = int((df_stats['success'] == True).sum())
                failures = int((df_stats['success'] == False).sum())
                rate = (successes / total * 100) if total > 0 else 0
                with col1:
                    st.metric("Total Articles", total)
                with col2:
                    st.metric("Successful", successes)
                with col3:
                    st.metric("Failed", failures)
                with col4:
                    st.metric("Success Rate", f"{rate:.1f}%")
            else:
                # Fallback to processor stats if 'success' column not present
                total = int(meta_stats.get('total_files', len(df_stats)))
                successes = int(meta_stats.get('processed', len(df_stats)))
                failures = int(meta_stats.get('skipped_error', 0))
                rate = (successes / total * 100) if total > 0 else 0
                with col1:
                    st.metric("Total Files", total)
                with col2:
                    st.metric("Processed", successes)
                with col3:
                    st.metric("Skipped (Error)", failures)
                with col4:
                    st.metric("Processed Rate", f"{rate:.1f}%")
            
            # Show processed data
            st.subheader("Processed Data")
            st.dataframe(st.session_state.csv_processed_df, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download CSV"):
                    csv = st.session_state.csv_processed_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📥 Download JSON"):
                    json_data = st.session_state.csv_processed_df.to_dict('records')
                    json_str = json.dumps(json_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            # Clear results
            if st.button("Clear Results"):
                st.session_state.csv_processed_df = None
                st.session_state.csv_metadata = None
                st.rerun()
        else:
            st.info("No processing results available. Run a processing task first.")
    
    # (Removed Start Post-Processing button and inline handler by request)


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
    tab1, tab2 = st.tabs(["Upload & Process", "History"])

    with tab1:
        st.subheader("Upload CSV File")
        st.markdown("""
        **Requirements:**
        - CSV file must contain a column named `content`
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
                st.info("💡 Make sure LM Studio is running and a model is loaded at the specified URL")
        
        st.divider()

        # File uploader
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with a 'content' column"
        )

        if uploaded_csv is not None:
            try:
                # Read the CSV to preview
                df_preview = pd.read_csv(uploaded_csv)
                
                st.success(f"File loaded: {uploaded_csv.name}")
                
                # Show file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df_preview))
                with col2:
                    st.metric("Total Columns", len(df_preview.columns))
                with col3:
                    has_content = 'content' in df_preview.columns
                    st.metric("Has 'content' column", "✓" if has_content else "✗")

                # Show column names
                with st.expander("View Columns"):
                    st.write(df_preview.columns.tolist())

                # Check if content column exists
                if 'content' not in df_preview.columns:
                    st.error("❌ CSV must contain a 'content' column")
                    st.info(f"Available columns: {', '.join(df_preview.columns)}")
                    st.stop()

                # Preview data
                st.subheader("Data Preview")
                st.dataframe(df_preview.head(10), use_container_width=True)

                # Process button
                if st.button("Start Summarization", type="primary", use_container_width=True):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_file:
                        tmp_file.write(uploaded_csv.getvalue())
                        tmp_path = Path(tmp_file.name)

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
                            
                            df_result, duration, metadata = await summarize_csv_file(
                                tmp_path, 
                                "content",
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
                        metadata['source_file'] = uploaded_csv.name
                        st.session_state.csv_metadata = CSVSummarizationMetadata(**metadata)
                        
                        # Clear processing flag
                        st.session_state.csv_processing = False

                        st.success(f"✓ Summarization complete in {duration:.2f} seconds ({format_time(duration)})!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing CSV: {e}")
                        logging.error(f"CSV processing error: {e}")
                        # Clear processing flag on error
                        st.session_state.csv_processing = False
                    finally:
                        # Clean up temp file
                        if tmp_path.exists():
                            tmp_path.unlink()

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
            
            # Add filter
            preview_count = st.slider("Number of rows to preview", 5, min(50, len(df_result)), 10)
            
            # Show results in expandable sections
            for idx in range(min(preview_count, len(df_result))):
                row = df_result.iloc[idx]
                with st.expander(f"Row {idx + 1}", expanded=(idx == 0)):
                    # Show tech intelligence fields at the top
                    tech_fields = []
                    if row.get('Dimension'):
                        tech_fields.append(f"Dimension: {row['Dimension']}")
                    if row.get('Tech'):
                        tech_fields.append(f"Tech: {row['Tech']}")
                    if row.get('TRL'):
                        tech_fields.append(f"TRL: {row['TRL']}")
                    if row.get('Start-up') and str(row['Start-up']) != 'N/A':
                        tech_fields.append(f"Start-up: {row['Start-up']}")
                    
                    if tech_fields:
                        st.markdown(f"**Tech Intelligence:** :blue[{', '.join(tech_fields)}]")
                        st.divider()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Content:**")
                        st.text_area(
                            "Original",
                            value=str(row['content'])[:500] + "..." if len(str(row['content'])) > 500 else str(row['content']),
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
            st.dataframe(df_result, use_container_width=True, height=400)

            # Save and download options
            st.subheader("Save & Download")
            
            if st.button("Save to 'summarised_content' folder", type="primary"):
                try:
                    # Save the processed CSV, JSON, and log
                    csv_path, json_path, log_path = save_summarized_csv(
                        df_result,
                        metadata.model_dump()
                    )

                    # Update metadata with paths
                    metadata.output_csv_path = str(csv_path)
                    metadata.output_json_path = str(json_path)
                    metadata.output_log_path = str(log_path)

                    # Save to history
                    history_path = Path("summarised_content") / "history.json"
                    history = CSVSummarizationHistory.from_file(history_path)
                    history.add_file(metadata)
                    history.to_file(history_path)

                    st.success(f"✓ Files saved successfully!")
                    st.info(f"📁 CSV: `{csv_path.name}`\n\n📄 Log: `{log_path.name}`")

                except Exception as e:
                    st.error(f"Error saving files: {e}")
                    logging.error(f"Error saving CSV results: {e}")

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
    
    # Import AgGrid
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
    
    # Load all CSV files
    summarised_dir = Path("summarised_content")
    
    if not summarised_dir.exists():
        st.warning("No summarised_content folder found. Process some CSV files first!")
        st.stop()
    
    # Find all CSV files (excluding history.json)
    csv_files = list(summarised_dir.glob("*.csv"))
    
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
                    if pd.isna(cat_string) or not str(cat_string).strip():
                        return ''
                    # Split by semicolon, strip whitespace, sort alphabetically, rejoin
                    cats = [c.strip() for c in str(cat_string).split(';') if c.strip()]
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
            unique_categories = combined_df['categories'].str.split(';').explode().str.strip().nunique()
            st.metric("Unique Categories", unique_categories)
    
    st.divider()
    
    # Filters and Search
    st.subheader("🔍 Filters & Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Source file filter
        source_files = ["All"] + sorted(combined_df['source_file'].unique().tolist())
        selected_source = st.selectbox("Source File", source_files)
    
    with col2:
        # Date range
        if 'processed_date' in combined_df.columns:
            unique_dates = sorted(combined_df['processed_date'].unique())
            if len(unique_dates) > 1:
                selected_date = st.selectbox("Processed Date", ["All"] + unique_dates)
            else:
                selected_date = "All"
        else:
            selected_date = "All"
    
    # Text search
    search_query = st.text_input("🔎 Search in database", placeholder="Enter keywords...")
    
    # Apply filters
    filtered_df = combined_df.copy()
    
    if selected_source != "All":
        filtered_df = filtered_df[filtered_df['source_file'] == selected_source]
    
    if selected_date != "All":
        filtered_df = filtered_df[filtered_df['processed_date'] == selected_date]
    
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
            autoHeight=True
        )
        # Configure specific columns
        if 'url' in display_df.columns:
            gb.configure_column(
                'url',
                headerName='URL',
                width=100,
                wrapText=True,
                autoHeight=True,
                cellStyle={'word-break': 'break-all', 'white-space': 'normal'}
            )
        
        if 'Indicator' in display_df.columns:
            gb.configure_column('Indicator', headerName='Summary/Indicator', width=150, wrapText=True)
        
        if 'title' in display_df.columns:
            gb.configure_column('title', headerName='Title', width=100, wrapText=True)
        
        if 'date' in display_df.columns:
            gb.configure_column('date', headerName='Date', width=100)
        
        # Configure selection
        gb.configure_selection(selection_mode='multiple', use_checkbox=True)
        
        gridOptions = gb.build()
        
        # Add custom CSS for URL wrapping
        gridOptions['defaultColDef']['wrapText'] = True
        gridOptions['defaultColDef']['autoHeight'] = True
        
        # Display AgGrid
        grid_response = AgGrid(
            display_df,
            gridOptions=gridOptions,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=False,
            theme='streamlit',  # can be 'streamlit' or 'streamlit-dark'
            width=800,
            height=800,
            allow_unsafe_jscode=True,
            enable_enterprise_modules=False
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
    if "rag_system" not in st.session_state:
        from embeddings_rag import LlamaIndexRAG
        st.session_state.rag_system = LlamaIndexRAG(persist_dir="rag_storage")
    
    rag_system = st.session_state.rag_system
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ RAG Configuration")
        
        # Add reload button at the top
        if st.button("🔄 Reload RAG System", help="Click if you updated the RAG code", use_container_width=True):
            # Force reload of the RAG system
            from importlib import reload
            import embeddings_rag
            reload(embeddings_rag)
            from embeddings_rag import LlamaIndexRAG
            st.session_state.rag_system = LlamaIndexRAG(persist_dir="rag_storage")
            st.success("✓ RAG system reloaded!")
            st.rerun()
        
        st.divider()
        
        # Show available persisted indexes
        available_indexes = rag_system.get_available_indexes()
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
        json_files = []
        if summarised_dir.exists():
            json_files = sorted([f.name for f in summarised_dir.glob("*.json") if f.name != "history.json"])
        
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
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                import traceback
                st.code(traceback.format_exc())

## About Page
elif page == "About":
    st.header("About Research Agent")

    st.markdown("""
    ### Overview

    The Research Agent is an AI-powered tool that automates comprehensive research through multiple stages:

    **Research Pipeline:**
    1. **Clarification** - Asks targeted questions to refine research scope
    2. **SERP Generation** - Creates optimized search queries
    3. **Web Search** - Executes searches via SearxNG
    4. **Results Collection** - Aggregates and saves search results

    **Learning Extraction:**
    - Analyzes search results using AI
    - Extracts key learnings with entities and metrics
    - Generates structured markdown reports

    **Summarization:**
    - Upload CSV files with a 'content' column
    - Automatically summarize content using tech-intelligence analysis
    - Track processing statistics and duration
    - Access history of all processed files
    - Download summarized CSV and processing logs

    **Database:**
    - Consolidated view of all summarized CSV files
    - Advanced filtering by category, source, and date
    - Full-text search across summaries and content
    - Multiple view modes (Cards, Table, Detailed)
    - Export filtered or complete database

    ### Features

    - Interactive clarification questions
    - Real-time progress tracking
    - Multi-query web search
    - Structured learning extraction
    - First-pass content summarization with tech-intel focus
    - Download results in JSON, Markdown, and CSV
    - Processing history

    ### Technology Stack

    - **Frontend:** Streamlit
    - **AI Model:** Azure OpenAI (GPT-4.1-mini)
    - **Search Engine:** SearxNG
    - **Validation:** Pydantic
    - **Agent Framework:** Pydantic AI
    - **Data Processing:** Pandas

    ### Usage Tips

    1. Start with a clear, specific research topic
    2. Answer clarification questions thoughtfully
    3. Review SERP queries before search execution
    4. Save results with descriptive filenames
    5. Reprocess results with different learning prompts if needed
    6. For summarization, ensure your file has a 'content' column
    7. Use the History tab to access previously processed CSV files

    ### File Structure

    - Search results saved to: `data/{filename}.json`
    - Learning reports saved to: `data/{filename}.md`
    - Post-processed files saved to: `processed_data/{website}_{YYYYMMDD}.csv/json`
    - Summarized files saved to: `summarised_content/{website}_{YYYYMMDD}.csv/json`
    - Processing logs saved to: `summarised_content/{website}_{YYYYMMDD}_log.txt`
    - Main logs saved to: `research.log`

    ### Support

    For issues or questions, check the logs at `research.log` or review the documentation.
    AL-AISG
    """)

    st.divider()

    st.markdown("""
    ### Quick Start Guide

    **Research Pipeline:**
    1. Navigate to "Research Pipeline" tab
    2. Enter your research topic
    3. Click "Start Research"
    4. Answer clarification questions
    5. Execute web search
    6. Save and download results

    **Learning Extraction:**
    1. Navigate to "Learning Extraction" tab
    2. Upload JSON file or select existing file
    3. Click "Extract Learnings"
    4. Preview and download markdown report

    **Summarization:**
    1. Navigate to "Summarization" tab
    2. Upload CSV file with 'content' column
    3. Preview the data
    4. Click "Start Summarization"
    5. Review results and statistics
    6. Save to folder or download directly
    7. Access processed files in the History tab

    **Database:**
    1. Navigate to "Database" tab
    2. View all summarized entries from all files
    3. Use filters to narrow down results
    4. Search for specific keywords
    5. Switch between view modes
    6. Export filtered or complete data
    7. View analytics and trends
    """)

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

st.sidebar.markdown("### Settings")
st.sidebar.info(f"Session ID: {id(st.session_state)}")

if st.sidebar.button("Clear Session"):
    reset_session_state()
    st.rerun()
