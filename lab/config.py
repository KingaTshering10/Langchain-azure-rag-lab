"""Load the root configuration without creating clients at import time."""
from dataclasses import dataclass, field
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AzureSettings:
    endpoint: str
    key: str = field(repr=False)
    deployment: str
    api_version: str = "2024-10-21"

    @classmethod
    def load(cls):
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        names = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"]
        values = [os.getenv(name, "").strip() for name in names]
        missing = [name for name, value in zip(names, values)
                   if not value or value.lower().startswith(("your-", "replace-", "https://your-"))]
        if missing:
            raise ValueError("Configure " + ", ".join(missing) + " in the root .env file or environment.")
        return cls(*values, os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21")


def get_chat_model():
    from langchain_openai import AzureChatOpenAI

    settings = AzureSettings.load()
    return AzureChatOpenAI(
        azure_endpoint=settings.endpoint, api_key=settings.key,
        azure_deployment=settings.deployment, api_version=settings.api_version,
        timeout=60, max_retries=2,
    )
