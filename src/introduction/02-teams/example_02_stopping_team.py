"""
Example 02: Stopping a Team of Agents

This script demonstrates how to create a team of agents that can be stopped
using an external termination condition or by a specific text mention.
We use a primary assistant agent and a critic agent. The critic provides
feedback and can approve the task completion.

The team runs until either the external termination condition is triggered
or the critic agent responds with "APPROVE".
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, ExternalTermination
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
    azure_ad_token_provider=token_provider,
)

# Create the primary agent.
primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="You are a helpful AI assistant.",
)

# Create the critic agent.
critic_agent = AssistantAgent(
    "critic",
    model_client=model_client,
    system_message="Provide constructive feedback. "
                   "Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

# Create a new team with an external termination condition.
external_termination = ExternalTermination()
team = RoundRobinGroupChat(
    [primary_agent, critic_agent],
    # Use the bitwise OR operator to combine conditions.
    termination_condition=external_termination | text_termination,
)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    # Run the team in a background task.
    run = asyncio.create_task(
        Console(team.run_stream(task="Write a short poem about the fall season.")))

    print("1 =================================================")
    # Wait for some time.
    await asyncio.sleep(0.1)

    print("2 =================================================")
    # Stop the team.
    external_termination.set()

    print("3 =================================================")
    # Wait for the team to finish.
    await run

    print("4 =================================================")
    await model_client.close()


asyncio.run(main())
