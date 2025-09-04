"""
Module for demonstrating an AssistantAgent using Azure OpenAI with tool integration.

This script sets up an assistant agent capable of answering weather-related queries
by integrating a simple weather tool. It authenticates with Azure OpenAI services,
loads environment variables, and streams responses to the console.

Dependencies:
  - autogen_agentchat
  - autogen_ext
  - azure.identity
  - dotenv

Usage:
    python lesson_two.py
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
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

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    model=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_INSTANCE_NAME"),
    azure_ad_token_provider=token_provider,
)


# Define a simple function tool that the agent can use.
# For this example, we use a fake weather tool for demonstration purposes.
async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."


# Define an AssistantAgent with the model, tool, system message, and reflection enabled.
# The system message instructs the agent via natural language.
agent = AssistantAgent(
    name="weather_agent",
    model_client=az_model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)


async def main():
    """
    Asynchronously returns a mock weather report for the specified city.

    Args:
        city (str): The name of the city to get the weather for.

    Returns:
        str: A string describing the weather in the given city.
    """
    await Console(agent.run_stream(task="What is the weather in New York?"))
    # Close the connection to the model client.
    await az_model_client.close()


asyncio.run(main())
