"""
Example 03: Handoff Termination

This example demonstrates how to set up an assistant agent that hands off
the conversation to a human user when it cannot complete a task. The agent uses two termination
conditions: one that triggers when a handoff to the user occurs, and another that triggers
when the agent responds with a specific keyword ("TERMINATE").
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
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

# Create a lazy assistant agent that always hands off to the user.
lazy_agent = AssistantAgent(
    "lazy_assistant",
    model_client=model_client,
    handoffs=[Handoff(target="user", message="Transfer to user.")],
    system_message="If you cannot complete the task, transfer to"
                   " user. Otherwise, when finished, respond with 'TERMINATE'.",
)

# Define a termination condition that checks for handoff messages.
handoff_termination = HandoffTermination(target="user")
# Define a termination condition that checks for a specific text mention.
text_termination = TextMentionTermination("TERMINATE")

# Create a single-agent team with the lazy assistant and both termination conditions.
lazy_agent_team = RoundRobinGroupChat([lazy_agent],
                                      termination_condition=handoff_termination | text_termination)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """

    # Run the team and stream to the console.
    task = "What is the weather in New York?"
    await Console(lazy_agent_team.run_stream(task=task), output_stats=True)

    await model_client.close()


asyncio.run(main())
