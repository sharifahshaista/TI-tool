"""
Model Configuration
Supports Azure OpenAI, OpenAI, and LM Studio (OpenAI-compatible)
"""

import os
from typing import Union
from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel

load_dotenv()


def get_azure_model(model_name: str = None) -> str:
    """
    Get Azure OpenAI model string
    
    Args:
        model_name: Azure deployment name (e.g., "pmo-gpt-4.1-nano")
    
    Returns:
        Model string in format "azure:deployment_name"
    """
    if model_name is None:
        model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4")
    
    # Ensure Azure environment variables are set
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"]
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required environment variable: {var}")
    
    return f"azure:{model_name}"


def get_openai_model(model_name: str = "gpt-4") -> str:
    """
    Get OpenAI model string
    
    Args:
        model_name: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
    
    Returns:
        Model string in format "openai:model_name"
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable")
    
    return f"openai:{model_name}"


def get_lm_studio_model(base_url: str = None, model_name: str = "local-model") -> str:
    """
    Get LM Studio model (OpenAI-compatible API)
    
    Args:
        base_url: LM Studio server URL (default from env or http://127.0.0.1:1234/v1)
        model_name: Model identifier (LM Studio uses whatever model is loaded)
    
    Returns:
        Model string in format "openai:model_name"
    """
    if base_url is None:
        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    
    # Set environment variables for OpenAI-compatible client
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
    
    return f"openai:{model_name}"


def get_model(provider: str = None, model_name: str = None, **kwargs) -> str:
    """
    Get model based on provider with automatic fallback
    
    Args:
        provider: "azure", "openai", or "lm_studio" (default from env or "azure")
        model_name: Model name/deployment name
        **kwargs: Additional arguments (base_url for LM Studio)
    
    Returns:
        Model string compatible with pydantic-ai
    
    Examples:
        >>> get_model("azure", "gpt-4")
        "azure:gpt-4"
        
        >>> get_model("openai", "gpt-4-turbo")
        "openai:gpt-4-turbo"
        
        >>> get_model("lm_studio", base_url="http://localhost:1234/v1")
        "openai:local-model"
    """
    # Get provider from environment if not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "azure").lower()
    
    provider = provider.lower()
    
    try:
        if provider == "azure":
            return get_azure_model(model_name)
        elif provider == "openai":
            if model_name is None:
                model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
            return get_openai_model(model_name)
        elif provider == "lm_studio":
            return get_lm_studio_model(
                base_url=kwargs.get('base_url'),
                model_name=model_name or "local-model"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: azure, openai, lm_studio")
    except Exception as e:
        print(f"Error configuring {provider} model: {e}")
        raise


# Available model options for UI selection
MODEL_OPTIONS = {
    "Azure OpenAI": {
        "provider": "azure",
        "models": [
            "pmo-gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-35-turbo",
        ],
        "env_vars": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"]
    },
    "OpenAI": {
        "provider": "openai",
        "models": [
            "gpt-4-turbo",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ],
        "env_vars": ["OPENAI_API_KEY"]
    },
    "LM Studio (Local)": {
        "provider": "lm_studio",
        "base_url": "http://127.0.0.1:1234/v1",
        "model_name": "local-model",
        "env_vars": ["LM_STUDIO_BASE_URL"],
        "description": "Requires LM Studio running locally with a model loaded"
    }
}


def get_available_providers() -> list:
    """Get list of available providers based on environment variables"""
    available = []
    
    # Check Azure OpenAI
    if all(os.getenv(var) for var in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]):
        available.append("azure")
    
    # Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        available.append("openai")
    
    # LM Studio is always available (local)
    available.append("lm_studio")
    
    return available

