"""
Azure OpenAI model configuration
This module maintains backward compatibility while supporting the new multi-provider system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set default API version if not set
if not os.getenv('OPENAI_API_VERSION'):
    os.environ['OPENAI_API_VERSION'] = '2023-12-01-preview'

# For backward compatibility, export the model string
# This uses the new get_model function to get the appropriate model
from config.model_config import get_model

# Default to Azure model for backward compatibility
# Can be overridden by setting LLM_PROVIDER environment variable
try:
    model = get_model(
        provider=os.getenv("LLM_PROVIDER", "azure"),
        model_name=os.getenv("AZURE_OPENAI_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME")
    )
except Exception as e:
    print(f"Warning: Could not initialize model from config: {e}")
    # Fallback for backward compatibility
    model = "azure:gpt-4"
