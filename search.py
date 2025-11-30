import asyncio
import os
from agents.clarification import get_clarifications
from agents.serp import get_serp_queries
from config.searxng_tools import searxng_web_tool, searxng_client
from schemas.datamodel import SearchResultsCollection
from pydantic import ValidationError

import logging
import json 
from pathlib import Path
from datetime import datetime


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='research.log')




async def clarify_and_search(topic: str, filename: str | None = None):
    """Main execution function with proper error handling"""
    try:
        topic = topic

        # Get clarifications
        clarifications = await get_clarifications(topic)
        logging.info(f"\nClarifications received: {clarifications}")

        # Generate SERP queries
        serp_queries = await get_serp_queries(topic, clarifications)
        logging.info(f"\nGenerated {serp_queries} SERP queries")

        if not serp_queries:
            print("No SERP queries generated. Exiting.")
            return
        print(f"\nGenerated {len(serp_queries)} SERP queries")

        # Collect search results
        results_collection = SearchResultsCollection()

        for query in serp_queries:
            print(f"\nSearching for: {query}")
            logging.info(f"\nSearching for: {query}")

            try:
                results = await searxng_client._search(query)
                logging.info(f"Found results for query: {results}")

                # Add to collection
                results_collection.add_result(query, results)


                print(f"Found results for query: {query}")

            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                logging.error(f"Error searching for '{query}': {e}")
                continue

        # Save results
        if results_collection.results:

            path = "data"
            os.makedirs(path, exist_ok=True)

            if not filename:
                filename = input("\nEnter filename to save results: ")
            output_path = Path(f"{path}/{filename}.json")

            try:
                results_collection.to_file(output_path)
                print(f"\nResults saved to {output_path}")
                print(f"Total queries executed: {results_collection.total_queries}")
                print(f"Collection timestamp: {results_collection.timestamp}")
            except Exception as e:
                print(f"Error saving results: {e}")
        else:
            print("\nNo results to save.")

    except ValidationError as e:
        print(f"\nValidation error: {e}")
        for error in e.errors():
            print(f"  - {error['loc']}: {error['msg']}")
  

# if __name__ == "__main__":
#     topic = "What is the DLOC (discount for lack of control) and DLOM for the Singapore companies?"
#     asyncio.run(clarify_and_searchs(topic))