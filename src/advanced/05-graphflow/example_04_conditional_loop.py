"""
Example 4: Conditional Loop in GraphFlow with Activation Groups
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
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

# Create agents for A→B→C→B→E scenario
agent_a = AssistantAgent("A",
                         model_client=model_client,
                         system_message="Start the process and provide initial input.")

agent_b = AssistantAgent(
    "B",
    model_client=model_client,
    system_message="Process input from A or feedback from C."
                   " Say 'CONTINUE' if it's from A or 'STOP' if it's from C.",
)

agent_c = AssistantAgent("C",
                         model_client=model_client,
                         system_message="Review B's output and provide feedback.")

agent_e = AssistantAgent("E",
                         model_client=model_client,
                         system_message="Finalize the process.")

# Build the graph with activation groups
builder = DiGraphBuilder()
builder.add_node(agent_a).add_node(agent_b).add_node(agent_c).add_node(agent_e)

# A → B (initial path)
builder.add_edge(agent_a, agent_b, activation_group="initial")

# B → C
builder.add_edge(agent_b, agent_c, condition="CONTINUE")

# C → B (loop back - different activation group)
builder.add_edge(agent_c, agent_b, activation_group="feedback")

# B → E (exit condition)
builder.add_edge(agent_b, agent_e, condition="STOP")

termination_condition = MaxMessageTermination(10)
# Build and create flow
graph = builder.build()
flow = GraphFlow(participants=[agent_a, agent_b, agent_c, agent_e],
                 graph=graph,
                 termination_condition=termination_condition)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    print("=== Example 1: A→B→C→B with 'All' Activation ===")
    print("B will exit when it receives a message from C")
    await Console(flow.run_stream(task="Start a review process for a document."))
    await model_client.close()


asyncio.run(main())
