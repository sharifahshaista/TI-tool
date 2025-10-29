import sys
from pathlib import Path
# Add parent directory to path so we can import config
sys.path.append(str(Path(__file__).parent.parent))


from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationError
from datetime import datetime
from typing import List, Optional, Any
from config.azure_model import model
from pydantic_ai import Agent
import asyncio
from schemas.datamodel import ClarificationResponse, ClarificationQuestion


#Agent
def create_clarification_agent():
    """Create the clarification agent"""
    
    clarification_agent = Agent(
        model=model,
        system_prompt=f"""Given the following query from the user, 
        ask some follow up questions to clarify the research direction. 
        Return a maximum of 4 questions, but feel free to return less if the original query is clear. 
        Format: Return each question on a new line, numbered as 1., 2., 3., and 4.
        Take note that the current time is {datetime.now()}
        """
    )
    # print(clarification_agent.system_prompt)
    
    return clarification_agent

async def get_clarifications(query: str) -> ClarificationResponse:
    """Get clarifications from user interactively

    Args:
        query: The original research query

    Returns:
        ClarificationResponse with Q&A pairs
    """

    clarification_agent = create_clarification_agent()

    # Create response object
    response = ClarificationResponse(original_query=query)

    try:
        # Get clarification questions from agent
        result = await clarification_agent.run(query)
        questions = result.output

        # Ask each question and get answers
        if questions:
            print("\nI need some clarification:")
            for i, question in enumerate(questions.split('\n'), 1):
                if question.strip():
                    print(f"\n{question}")
                    answer = input("Your answer: ")
                    qa = ClarificationQuestion(
                        question=question.strip(),
                        answer=answer
                    )
                    response.questions_and_answers.append(qa)
    except Exception as e:
        print(f"Error getting clarifications: {e}")

    return response


if __name__ == "__main__":
    import asyncio
    response = asyncio.run(get_clarifications("What is the DLOC (discount for lack of control) and DLOM (discount for lack of marketability) for the Singapore companies?"))
    print(response)
