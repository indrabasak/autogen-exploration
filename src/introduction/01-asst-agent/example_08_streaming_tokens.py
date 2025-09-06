"""
This example demonstrates how to create an AssistantAgent with streaming
token support using Azure OpenAI.
The agent will stream its responses token by token, allowing for real-time interaction.
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# Create the token provider
token_provider = AzureTokenProvider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    model=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_INSTANCE_NAME"),
    azure_ad_token_provider=token_provider,
)

streaming_assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful assistant.",
    # Define the output content type of the agent.
    model_client_stream=True,  # Enable streaming tokens.
)


async def main():
    """
    Main function to run the streaming assistant.
    :return:
    """
    # Use an async function and asyncio.run() in a script.
    async for message in (streaming_assistant.run_stream
        (task="Name two cities in South America")):  # type: ignore
        print(message)
    await model_client.close()


asyncio.run(main())
