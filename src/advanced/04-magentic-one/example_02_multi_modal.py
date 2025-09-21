"""
Example of using MagenticOneGroupChat with MultimodalWebSurfer agent.
"""
import asyncio
import os

from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
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

surfer = MultimodalWebSurfer(
    "WebSurfer",
    model_client=model_client,
)

team = MagenticOneGroupChat([surfer], model_client=model_client)


# # Note: you can also use  other agents in the team
# team = MagenticOneGroupChat([surfer, file_surfer, coder, terminal], model_client=model_client)
# file_surfer = FileSurfer( "FileSurfer",model_client=model_client)
# coder = MagenticOneCoderAgent("Coder",model_client=model_client)
# terminal = CodeExecutorAgent("ComputerTerminal",code_executor=LocalCommandLineCodeExecutor())

async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    await Console(team.run_stream(task="What is the UV index in Melbourne today?"))
    await model_client.close()


asyncio.run(main())
