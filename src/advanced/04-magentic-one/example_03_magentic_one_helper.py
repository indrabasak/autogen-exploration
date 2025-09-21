"""
Example of using MagenticOne with a helper agent (MultimodalWebSurfer) and
an approval function for code execution.
"""
import asyncio
import os

from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse
from autogen_agentchat.ui import Console
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
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


def approval_func(request: ApprovalRequest) -> ApprovalResponse:
    """Simple approval function that requests user input before code execution."""
    print(f"Code to execute:\n{request.code}")
    user_input = input("Do you approve this code execution? (y/n): ").strip().lower()
    if user_input == 'y':
        return ApprovalResponse(approved=True, reason="User approved the code execution")
    else:
        return ApprovalResponse(approved=False, reason="User denied the code execution")


# Enable code execution approval for security
m1 = MagenticOne(client=model_client, approval_func=approval_func)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    task = "Write a Python script to fetch data from an API."
    result = await Console(m1.run_stream(task=task))
    print(result)
    await model_client.close()


asyncio.run(main())
