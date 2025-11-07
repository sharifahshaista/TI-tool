from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings
from tools.searxng_client import SearxNGClient

class Settings(BaseSettings):
    """Application settings from environment"""
    searxng_url: HttpUrl = Field(default="http://localhost:32768")
    searxng_num_results: int = Field(default=20, ge=1, le=100)

settings = Settings()
searxng_client = SearxNGClient(str(settings.searxng_url), num_results=settings.searxng_num_results)

searxng_web_tool = searxng_client.get_tool()
