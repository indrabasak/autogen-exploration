"""
An example of using ListMemory to store user preferences and context for an assistant agent.
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
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

# Initialize user memory
user_memory = ListMemory()


async def get_weather(city: str, units: str = "imperial") -> str:
    """
    Mock function to get weather information.
    :param city:
    :param units:
    :return:
    """
    if units == "imperial":
        return f"The weather in {city} is 73 °F and Sunny."

    if units == "metric":
        return f"The weather in {city} is 23 °C and Sunny."

    return f"Sorry, I don't know the weather in {city}."


assistant_agent = AssistantAgent(
    name="assistant_agent",
    model_client=model_client,
    tools=[get_weather],
    memory=[user_memory],
)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """

    # Add user preferences to memory
    await user_memory.add(MemoryContent(
        content="The weather should be in metric units",
        mime_type=MemoryMimeType.TEXT))
    await user_memory.add(MemoryContent(
        content="Meal recipe must be vegan",
        mime_type=MemoryMimeType.TEXT))

    # Run the agent with a task.
    # Add user preferences to memory
    stream = assistant_agent.run_stream(task="What is the weather in New York?")
    await Console(stream)

    stream = assistant_agent.run_stream(task="Write brief meal recipe with broth")
    await Console(stream)

    await model_client.close()


asyncio.run(main())
