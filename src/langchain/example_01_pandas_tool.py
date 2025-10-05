"""
Example of using the Pandas DataFrame with LangChain tool - PythonAstREPLTool.
"""
import asyncio
import os

import pandas as pd
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_experimental.tools.python.tool import PythonAstREPLTool

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

CSV_FILE_URL = "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
df = pd.read_csv(CSV_FILE_URL)  # type: ignore

tool = LangChainToolAdapter(PythonAstREPLTool(locals={"df": df}))

agent = AssistantAgent(
    "assistant",
    tools=[tool],
    model_client=model_client,
    system_message="Use the `df` variable to access the dataset.",
)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """

    # Run the team and stream to the console.
    await Console(agent.on_messages_stream(
        [TextMessage(content="What's the average age of the passengers?", source="user")],
        CancellationToken()
    ))

    await model_client.close()


asyncio.run(main())
