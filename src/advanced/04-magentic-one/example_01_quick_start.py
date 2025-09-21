"""
This example demonstrates how to set up a team of agents using
the MagenticOneGroupChat framework
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
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

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    model=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_INSTANCE_NAME"),
    azure_ad_token_provider=token_provider
)

assistant = AssistantAgent(
    "Assistant",
    model_client=model_client,
)
team = MagenticOneGroupChat([assistant], model_client=model_client)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    await Console(team.run_stream(task="Provide a different proof for Fermat's Last Theorem"))
    await model_client.close()


asyncio.run(main())
