"""
Example of a team of agents working in parallel with a join using GraphFlow.
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
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

# Create the writer agent
writer = AssistantAgent("writer",
                        model_client=model_client,
                        system_message="Draft a short paragraph on climate change.")

# Create two editor agents
editor1 = AssistantAgent("editor1",
                         model_client=model_client,
                         system_message="Edit the paragraph for grammar.")

editor2 = AssistantAgent("editor2",
                         model_client=model_client,
                         system_message="Edit the paragraph for style.")

# Create the final reviewer agent
final_reviewer = AssistantAgent(
    "final_reviewer",
    model_client=model_client,
    system_message="Consolidate the grammar and style edits into a final version.",
)

# Build the workflow graph
builder = DiGraphBuilder()
builder.add_node(writer).add_node(editor1).add_node(editor2).add_node(final_reviewer)

# Fan-out from writer to editor1 and editor2
builder.add_edge(writer, editor1)
builder.add_edge(writer, editor2)

# Fan-in both editors into final reviewer
builder.add_edge(editor1, final_reviewer)
builder.add_edge(editor2, final_reviewer)

# Build and validate the graph
graph = builder.build()

# Create the flow
flow = GraphFlow(
    participants=builder.get_participants(),
    graph=graph,
)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    # Use `asyncio.run(...)` and wrap the below in a async function when running in a script.
    # stream = flow.run_stream(task="Write a short paragraph about climate change.")
    # async for event in stream:  # type: ignore
    #     print(event)
    await Console(flow.run_stream(task="Write a short paragraph about climate change."))
    await model_client.close()


asyncio.run(main())
