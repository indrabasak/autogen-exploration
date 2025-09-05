"""
Example demonstrating the use of multi-modal messages with an AssistantAgent.

This script fetches a random image, constructs a multi-modal message containing both
text and image, and sends it to an AssistantAgent powered by Azure OpenAI. The agent
processes the message and returns a description of the image content.

Dependencies:
- PIL (Pillow)
- requests
- autogen_agentchat
- autogen_core
- autogen_ext
- azure.identity
- dotenv

Usage:
uv run src/introduction/01-asst-agent/example_02_multimodal_msg.py
"""

import asyncio
import os
from io import BytesIO

import PIL
import requests
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

# Create a multi-modal message with random image and text.
pil_image = PIL.Image.open(
    BytesIO(requests.get("https://picsum.photos/300/200").content) # pylint: disable=missing-timeout
)
img = Image(pil_image)
multi_modal_message = MultiModalMessage(
    content=["Can you describe the content of this image?", img], source="user"
)

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
    system_message="Use tools to solve tasks.",
)


async def main():
    """
    Asynchronously sends a multi-modal message (text and image) to the AssistantAgent
    and prints the agent's response.

    This function runs the agent with a user message containing both text and an image,
    then prints the last message content from the agent's response and closes the model client.
    """
    result = await agent.run(task=multi_modal_message)
    print(result.messages[-1].content)  # type: ignore
    await model_client.close()


asyncio.run(main())
