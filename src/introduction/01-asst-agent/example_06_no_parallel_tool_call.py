"""
Example demonstrating disabling parallel tool call.

Dependencies:
- autogen_agentchat
- autogen_ext
- azure.identity
- dotenv

Usage:
    uv run src/introduction/01-asst-agent/example_06_no_parallel_tool_call.py
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
    return "Tintin is a comic character created by Belgian cartoonist Hergé."


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
    parallel_tool_calls=False,
)

agent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Use tools to solve tasks.",
)


async def main():
    """
    Asynchronously runs the AssistantAgent to find information on Tintin and
    prints the agent's messages.

    This function sends a task to the agent, prints the resulting messages, and
    closes the model client connection.
    """
    result = await agent.run(task="Find information on Tintin")
    print(result.messages)
    await model_client.close()


asyncio.run(main())
