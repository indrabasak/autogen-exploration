import asyncio
import os
from dotenv import load_dotenv
from autogen_core.models import UserMessage
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential

load_dotenv()

# Create the token provider
token_provider = AzureTokenProvider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    model = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.environ.get("AZURE_OPENAI_API_INSTANCE_NAME"),
    azure_ad_token_provider = token_provider
)

async def main():
    result = await az_model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(result)
    await az_model_client.close()

asyncio.run(main())
