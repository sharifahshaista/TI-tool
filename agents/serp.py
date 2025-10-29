import sys
from pathlib import Path
# Add parent directory to path so we can import config
sys.path.append(str(Path(__file__).parent.parent))

# Import clarification agent
from agents.clarification import ClarificationResponse


from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationError
from datetime import datetime
from typing import List, Optional, Any
from config.azure_model import model
from pydantic_ai import Agent
import asyncio
from schemas.datamodel import SerpQueryRequest


#Agent
def create_serp_agent():
    """Create the SERP agent"""
    
    serp_agent = Agent(
        model=model,
        system_prompt="""
        Given the following prompt from the user on the topic  to research 
        and the clarification question the user's response 
        generate a list of SERP queries to research the topic.
        By as specific and comprehensive as possible.
        Reduce the number of words in each query to its keywords only.
        """
    )
    
    return serp_agent

async def get_serp_queries(query: str, clarifications: ClarificationResponse) -> List[str]:
    """Get SERP queries from user interactively

    Args:
        query: The original research query
        clarifications: Clarification responses from user

    Returns:
        List of SERP queries to execute
    """

    serp_agent = create_serp_agent()

    # Create request object
    request = SerpQueryRequest(
        query=query,
        clarifications=clarifications
    )

    # Build combined prompt
    combined_prompt = f"""
    Original Query: {request.query}

    Clarifications:
    """

    if request.clarifications:
        for qa in request.clarifications.questions_and_answers:
            combined_prompt += f"\nQ: {qa.question}\nA: {qa.answer}\n"

    print(f'\n Combined Prompt: {combined_prompt}\n')

    try:
        # Get SERP queries from agent
        result = await serp_agent.run(combined_prompt)
        # agent_response = AgentResponse(output=result.output)

        # Parse and validate queries
        queries_list = [
            q.strip()
            for q in str(result.output).split('\n')
            if q.strip()
        ][:request.max_queries]  # Limit to max_queries

        return queries_list
    except ValidationError as e:
        print(f"Validation error in SERP queries: {e}")
        return []
    except Exception as e:
        print(f"Error generating SERP queries: {e}")
        return []

if __name__ == "__main__":
    import asyncio
    clarifications = asyncio.run(get_clarifications("What is the DLOC (discount for lack of control) and DLOM (discount for lack of marketability) for the Singapore companies?"))
    print(clarifications)
    response = asyncio.run(get_serp_queries("What is the DLOC (discount for lack of control) and DLOM (discount for lack of marketability) for the Singapore companies?", clarifications))
    print(response)