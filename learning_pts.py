import os
import asyncio
import json
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from schemas.datamodel import QueryLearnings
from config.azure_model import model
from agents.learn import create_learning_agent, get_learning_structured

import logging

import streamlit as st



logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='research.log')

async def get_learning(file_path : str, file_name : str = None):
    
    path = "data"
    os.makedirs(path, exist_ok=True)
    
    # Load from JSON file
    with open(f"{path}/{file_path}.json", "r", encoding="utf-8") as f:
        results_dict = json.load(f)

    print(" Loaded results")
    print(len(results_dict.keys()))  # Shows all queries stored
    logging.info(f"Loaded {len(results_dict.keys())} queries")

    if not file_name:
        file_name = input("\nEnter filename to save results: ")

    for query in results_dict.keys():
        print(f'\nQuery: {query}')
     
      
        learnings = await get_learning_structured(query, results_dict[query])
        logging.info(f'Learnings: \n{learnings} \n\n')
        print(f'\n\nLearnings: \n{learnings} \n\n')
        # st.write(f'\n\nLearnings: \n{learnings} \n\n')

        

        with open(f"{path}/{file_name}.md", "a", encoding="utf-8") as f:  
            f.write(f"{query}\n\n")
            f.write(learnings)
            f.write("\n\n")

    print(f"Learnings saved to {file_name}.md")
    


# if __name__ == "__main__":

#     file_path = "dlocm.json"
#     asyncio.run(get_learning(file_path))