"""
Example 6: Mixed Activation Groups in a GraphFlow
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

# Create agents for mixed activation scenario
agent_a3 = AssistantAgent("A",
                          model_client=model_client,
                          system_message="Provide critical input that must be processed.")

agent_b3 = AssistantAgent("B",
                          model_client=model_client,
                          system_message="Provide secondary critical input.")

agent_c3 = AssistantAgent("C",
                          model_client=model_client,
                          system_message="Provide optional quick input.")

agent_d3 = AssistantAgent("D",
                          model_client=model_client,
                          system_message="Process inputs based on different priority levels.")

# Build graph with mixed activation groups
builder3 = DiGraphBuilder()
builder3.add_node(agent_a3).add_node(agent_b3).add_node(agent_c3).add_node(agent_d3)

# Critical inputs that must ALL be present (activation_group="critical", activation_condition="all")
builder3.add_edge(agent_a3, agent_d3, activation_group="critical", activation_condition="all")
builder3.add_edge(agent_b3, agent_d3, activation_group="critical", activation_condition="all")

# Optional input that can trigger execution on its own
# (activation_group="optional", activation_condition="any")
builder3.add_edge(agent_c3, agent_d3, activation_group="optional", activation_condition="any")

# Build and create flow
graph3 = builder3.build()
flow3 = GraphFlow(participants=[agent_a3, agent_b3, agent_c3, agent_d3], graph=graph3)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """

    print("=== Example 3: Mixed Activation Groups ===")
    print("D will execute when:")
    print("- BOTH A AND B complete (critical group with 'all' activation), OR")
    print("- C completes (optional group with 'any' activation)")
    print("This allows for both required dependencies and fast-path triggers.")
    await Console(flow3.run_stream(task="Process inputs with mixed priority levels."))

    await model_client.close()


asyncio.run(main())
