import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationError
from pydantic_settings import BaseSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider

load_dotenv()

# Configuration Models
class AzureConfig(BaseModel):
    """Azure OpenAI configuration"""
    api_key: str
    api_version: str
    azure_endpoint: HttpUrl
    model_name: str = Field(default="pmo-gpt-4.1-nano")

    @field_validator('api_key', 'api_version')
    @classmethod
    def validate_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Value cannot be empty")
        return v


class Settings(BaseSettings):
    """Application settings from environment"""
    azure_api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    azure_api_version: str = Field(alias="AZURE_OPENAI_API_VERSION")
    azure_endpoint: HttpUrl = Field(alias="AZURE_OPENAI_ENDPOINT")

    model_name: str = Field(default="pmo-gpt-4.1-nano")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env

# Initialize settings
try:
    settings = Settings()
    azure_config = AzureConfig(
        api_key=settings.azure_api_key,
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_endpoint,
        model_name=settings.model_name
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
    raise

azure_provider = AzureProvider(
    api_key=azure_config.api_key,
    api_version=azure_config.api_version,
    azure_endpoint=str(azure_config.azure_endpoint),
)

model = OpenAIChatModel(
    model_name=azure_config.model_name,
    provider=azure_provider,
)

