"""
Example demonstrating the use of McpWorkbench and AssistantAgent to fetch
and summarize web content with Azure OpenAI integration.

This script configures an AssistantAgent to utilize the MCP server fetch tool
for retrieving and summarizing the content of a specified URL.
It authenticates with Azure OpenAI using Azure AD credentials, loads environment
variables, and prints the summarized output.

Dependencies:
- autogen_agentchat
- autogen_ext
- azure.identity
- dotenv

Usage:
    python src/introduction/01-asst-agent/example_05_mcp_workbench.py
    uv run src/introduction/01-asst-agent/example_05_mcp_workbench.py
"""

import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
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

# Get the fetch tool from mcp-server-fetch.
fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])


async def main():
    """
    Asynchronously creates an AssistantAgent with MCP fetch tool to retrieve
    and summarize web content.

    This function initializes the MCP workbench, configures the AssistantAgent
    to use the fetch tool,
    runs a summarization task for a specified URL, prints the summary, and
    closes the model client.
    """
    async with McpWorkbench(fetch_mcp_server) as workbench:
        # Create an agent that can use the fetch tool.
        fetch_agent = AssistantAgent(
            name="fetcher",
            model_client=model_client,
            workbench=workbench,
            reflect_on_tool_use=True,
        )  # type: ignore

        # Let the agent fetch the content of a URL and summarize it.
        result = await fetch_agent.run(
            task="Summarize the content of https://en.wikipedia.org/wiki/Seattle"
        )

        assert isinstance(result.messages[-1], TextMessage)
        print(result.messages[-1].content)
        await model_client.close()


asyncio.run(main())
