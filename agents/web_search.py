import sys
from pathlib import Path
# Add parent directory to path so we can import config
sys.path.append(str(Path(__file__).parent.parent))


from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationError
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic_ai import Agent
import asyncio
import json

from config.searxng_tools import searxng_web_tool
from config.azure_model import model
from schemas.datamodel import ResearchResult

def create_web_search_agent():
    """Create the research agent"""

    
    # Create agent
    agent = Agent(
        model=model,
        output_type=ResearchResult,
        system_prompt="""You are a research assistant. 
        Use the search results provided to create a detailed content about the topic.
        Be detailed on all aspects
        Extract key points , list , urls of the sources you found."""
    )
    
    # Register the search tool
    agent.tool(searxng_web_tool)
    
    return agent

if __name__ == "__main__":
    web_search_agent = create_web_search_agent()
    response = web_search_agent.run_sync("Singapore DLOC DLOM of Singapore Companies")
    print(response)
    