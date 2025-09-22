"""
Example 5: Loop with Multiple Paths and "Any" Activation Condition
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

# Create agents for A→B→(C1,C2)→B scenario
agent_a2 = AssistantAgent("A",
                          model_client=model_client,
                          system_message="Initiate a task that needs parallel processing.")

agent_b2 = AssistantAgent(
    "B",
    model_client=model_client,
    system_message="Coordinate parallel tasks. "
                   "Say 'PROCESS' to start parallel work or 'DONE' to finish.",
)

agent_c1 = AssistantAgent("C1",
                          model_client=model_client,
                          system_message="Handle task type 1. Say 'C1_COMPLETE' when done.")

agent_c2 = AssistantAgent("C2",
                          model_client=model_client,
                          system_message="Handle task type 2. Say 'C2_COMPLETE' when done.")

agent_e = AssistantAgent("E",
                         model_client=model_client,
                         system_message="Finalize the process.")

# Build the graph with "any" activation
builder2 = DiGraphBuilder()
(builder2.add_node(agent_a2).add_node(agent_b2)
 .add_node(agent_c1).add_node(agent_c2).add_node(agent_e))

# A → B (initial)
builder2.add_edge(agent_a2, agent_b2)

# B → C1 and B → C2 (parallel fan-out)
builder2.add_edge(agent_b2, agent_c1, condition="PROCESS")
builder2.add_edge(agent_b2, agent_c2, condition="PROCESS")

# B → E (exit condition)
builder2.add_edge(agent_b2, agent_e, condition=lambda msg: "DONE" in msg.to_model_text())

# C1 → B and C2 → B (both in same activation group with "any" condition)
builder2.add_edge(
    agent_c1, agent_b2, activation_group="loop_back_group",
    activation_condition="any", condition="C1_COMPLETE"
)

builder2.add_edge(
    agent_c2, agent_b2, activation_group="loop_back_group",
    activation_condition="any", condition="C2_COMPLETE"
)

termination_condition = MaxMessageTermination(10)
# Build and create flow
graph2 = builder2.build()
flow2 = GraphFlow(participants=[agent_a2, agent_b2, agent_c1, agent_c2, agent_e],
                  graph=graph2, termination_condition=termination_condition)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    print("=== Example 2: A→B→(C1,C2)→B with 'Any' Activation ===")
    print("B will execute as soon as EITHER C1 OR C2 completes (whichever finishes first)")
    await Console(flow2.run_stream(task="Start a parallel processing task."))
    await model_client.close()


asyncio.run(main())
