"""
This example demonstrates how to set up a team of agents where one
agent is a user proxy that takes input from the console during the conversation.
The conversation continues until the user types "APPROVE".
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
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

assistant = AssistantAgent("assistant", model_client=model_client)

# Use input() to get user input from console.
user_proxy = UserProxyAgent("user_proxy", input_func=input)

# Create the termination condition which will end the conversation when the user says "APPROVE".
termination = TextMentionTermination("APPROVE")

# Create the team.
team = RoundRobinGroupChat([assistant, user_proxy], termination_condition=termination)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """

    # Run the conversation and stream to the console.
    stream = team.run_stream(task="Write a 4-line poem about the ocean.")
    # Use asyncio.run(...) when running in a script.
    await Console(stream)

    await model_client.close()


asyncio.run(main())
