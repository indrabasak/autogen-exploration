"""
This example demonstrates how to use the AssistantAgent with structured output
using Pydantic models.
The agent is designed to categorize input text as "happy", "sad", or "neutral" and provide
its thoughts on the categorization.
"""

import asyncio
import os
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class AgentResponse(BaseModel):
    """
    The response format for the agent as a Pydantic base model.
    """
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


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
    system_message="Categorize the input as happy, sad, or neutral following the JSON format.",
    # Define the output content type of the agent.
    output_content_type=AgentResponse,
)


async def main():
    """
    Asynchronously runs the AssistantAgent with structured output
    and prints the thoughts and response.

    This function sends a task to the agent, checks the type of the last message in the result,
    validates its content type, prints the agent's thoughts and response,
    and closes the model client connection.
    """
    result = await Console(agent.run_stream(task="I am happy."))

    # Check the last message in the result, validate its type, and print the thoughts and response.
    assert isinstance(result.messages[-1], StructuredMessage)
    assert isinstance(result.messages[-1].content, AgentResponse)
    print("Thought: ", result.messages[-1].content.thoughts)
    print("Response: ", result.messages[-1].content.response)
    await model_client.close()


asyncio.run(main())
