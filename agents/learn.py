from config.azure_model import model
from schemas.datamodel import QueryLearnings
from pydantic_ai import Agent


def create_learning_agent():
    """Create the learning agent"""
    
    learning_agent = Agent(
        model=model,
        output_type=QueryLearnings,  # Use Pydantic model for structured output
        system_prompt="""
        Given the following contents from a SERP search for the query, 
        generate a list of learnings from the contents. 
        
        Return a maximum of 5 learnings.
        
        For EACH learning:
        1. Make it unique and not similar to others
        2. Be concise but information-dense
        3. Include entities (people, places, companies, products)
        4. Include exact metrics, numbers, or dates
        5. IT IS MANDATORY to include the exact source URLs from the content
        6. Never make things up - only use information from the provided content
        
        Format each learning with:
        - Clear, factual statement
        - All relevant entities and metrics
        - Source URLs that support the learning
        - Add a blank line between each learning for readability
        
        The learnings will be used for further research, so accuracy and source attribution are critical.
        """
    )
    return learning_agent

async def get_learning_structured(query: str, contents: str) -> QueryLearnings:
    """Get structured learnings with Pydantic validation"""
    
    learning_agent = create_learning_agent()
    
    # Combine query and contents
    combined_prompt = f"""
    Query: {query}
    
    Contents: {contents}
    """
    
    # Get learnings
    result = await learning_agent.run(combined_prompt)

    
    return result.output.learnings