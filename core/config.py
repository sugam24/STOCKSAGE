"""
Configuration management for StockSage using Pydantic BaseSettings.
Loads environment variables and validates configuration at startup.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Attributes:
        anthropic_api_key: API key for Anthropic Claude service
    """
    
    anthropic_api_key: str = Field(
        ...,
        alias="ANTHROPIC_API_KEY",
        description="API key for Anthropic Claude service"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"


# Global settings instance
settings = Settings()
