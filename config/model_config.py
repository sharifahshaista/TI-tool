"""
Model Configuration
Supports both Azure OpenAI and LM Studio (OpenAI-compatible)
"""

import os
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider

load_dotenv()


def get_azure_model(model_name: str = None):
    """Get Azure OpenAI model"""
    from config.azure_model import azure_provider, azure_config
    
    if model_name is None:
        model_name = azure_config.model_name
    
    return OpenAIChatModel(
        model_name=model_name,
        provider=azure_provider,
    )


def get_lm_studio_model(base_url: str = "http://127.0.0.1:1234/v1", model_name: str = "local-model"):
    """
    Get LM Studio model (OpenAI-compatible API)
    
    Args:
        base_url: LM Studio server URL (default: http://127.0.0.1:1234/v1)
        model_name: Model identifier (use any string, LM Studio uses loaded model)
    """
    # Set environment variables for OpenAI client
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    
    # Use the openai: prefix to create a model
    return f"openai:{model_name}"


def get_model(provider: str = "azure", **kwargs):
    """
    Get model based on provider
    
    Args:
        provider: "azure" or "lm_studio"
        **kwargs: Additional arguments for model creation
    """
    if provider == "azure":
        return get_azure_model(kwargs.get('model_name'))
    elif provider == "lm_studio":
        return get_lm_studio_model(
            base_url=kwargs.get('base_url', "http://127.0.0.1:1234/v1"),
            model_name=kwargs.get('model_name', "local-model")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Available model options
MODEL_OPTIONS = {
    "Azure OpenAI": {
        "provider": "azure",
        "models": [
            "pmo-gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
        ]
    },
    "LM Studio (Local)": {
        "provider": "lm_studio",
        "base_url": "http://127.0.0.1:1234/v1",
        "model_name": "local-model"  # LM Studio uses whatever model is loaded
    }
}

