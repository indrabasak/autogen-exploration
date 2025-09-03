import asyncio
import os
from dotenv import load_dotenv
from autogen_core.models import UserMessage
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console

load_dotenv()

# Create the token provider
token_provider = AzureTokenProvider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    model = os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.environ.get("AZURE_OPENAI_API_INSTANCE_NAME"),
    azure_ad_token_provider = token_provider
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
    await Console(agent.run_stream(task="What is the weather in New York?"))
    # Close the connection to the model client.
    await az_model_client.close()

asyncio.run(main())
