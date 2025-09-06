"""
This example demonstrates how to set up a team of agents using the RoundRobinGroupChat class.
In this setup, we have a primary agent that generates content and a critic agent that reviews
the content. The interaction continues until the critic approves the content by responding
with "APPROVE".
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
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

# Create a team with the primary and critic agents.
team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    # Use an async function and asyncio.run() in a script.
    result = await team.run(task="Write a short poem about the fall season.")
    print(result)

    print("===================================================")
    # When running inside a script, use a async main function and call it from `asyncio.run(...)`.
    await team.reset()  # Reset the team for a new task.
    async for message in team.run_stream(
            task="Write a short poem about the fall season."):  # type: ignore
        if isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)
        else:
            print(message)

    print("===================================================")
    # The Console() method provides a convenient way to print messages to the
    # console with proper formatting.
    await team.reset()  # Reset the team for a new task.
    await Console(
        team.run_stream(
            task="Write a short poem about the fall season."))  # Stream the msgs to the console.

    await model_client.close()


asyncio.run(main())
