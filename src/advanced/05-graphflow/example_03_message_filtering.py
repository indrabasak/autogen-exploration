"""
Example of using message filtering in a team of agents using GraphFlow.
"""
import asyncio
import os

from autogen_agentchat.agents import (AssistantAgent,
                                      MessageFilterAgent, MessageFilterConfig, PerSourceFilter)
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
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

# Create agents
researcher = AssistantAgent(
    "researcher",
    model_client=model_client,
    system_message="Summarize key facts about climate change."
)
analyst = AssistantAgent("analyst",
                         model_client=model_client,
                         system_message="Review the summary and suggest improvements.")

presenter = AssistantAgent(
    "presenter",
    model_client=model_client,
    system_message="Prepare a presentation slide based on the final summary."
)

# Apply message filtering
filtered_analyst = MessageFilterAgent(
    name="analyst",
    wrapped_agent=analyst,
    filter=MessageFilterConfig(
        per_source=[PerSourceFilter(source="researcher", position="last", count=1)]),
)

filtered_presenter = MessageFilterAgent(
    name="presenter",
    wrapped_agent=presenter,
    filter=MessageFilterConfig(
        per_source=[PerSourceFilter(source="analyst", position="last", count=1)]),
)

# Build the flow
builder = DiGraphBuilder()
builder.add_node(researcher).add_node(filtered_analyst).add_node(filtered_presenter)
builder.add_edge(researcher, filtered_analyst).add_edge(filtered_analyst, filtered_presenter)

# Create the flow
flow = GraphFlow(
    participants=builder.get_participants(),
    graph=builder.build(),
)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    await Console(flow.run_stream(task="Summarize key facts about climate change."))
    await model_client.close()


asyncio.run(main())
