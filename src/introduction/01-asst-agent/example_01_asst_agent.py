"""
Example demonstrating the creation and use of an AssistantAgent with Azure OpenAI integration.

This script sets up an AssistantAgent capable of answering queries using a mock web search tool.
It authenticates with Azure OpenAI services using Azure AD credentials, loads environment variables,
and streams responses to the console.

Dependencies:
- autogen_agentchat
- autogen_ext
- azure.identity
- dotenv

Usage:
    python src/introduction/02-agent/example_01_asst_agent.py
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


# Define a tool that searches the web for information.
# For simplicity, we will use a mock function here that returns a static string.
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


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

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
)


async def main():
    """
    Asynchronously runs the AssistantAgent to find information on AutoGen and prints the agent's messages.

    This function sends a task to the agent, prints the resulting messages, and closes the model client connection.
    """
    result = await agent.run(task="Find information on AutoGen")
    print(result.messages)
    await model_client.close()


asyncio.run(main())
