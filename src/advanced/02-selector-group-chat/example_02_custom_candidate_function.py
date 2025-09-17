"""
An example demonstrating a custom candidate function in a SelectorGroupChat
to dynamically select which agents should participate in each round of conversation.
The team consists of three agents: a planning agent, a web search agent, and a data
analyst agent. The planning agent is responsible for breaking down complex tasks
into smaller subtasks and delegating them to the other two agents. The custom candidate
function ensures that the planning agent is always the first to respond to a new user task,
and it selects the appropriate agents based on the planning agent's output.
"""
import asyncio
import os
from typing import Sequence, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
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


# Note: This example uses mock tools instead of real APIs for demonstration purposes
def search_web_tool(query: str) -> str:
    """
    A mock web search tool that returns predefined results based on the query.
    :param query:
    :return:
    """
    if "2006-2007" in query:
        return """Here are the total points scored by Miami Heat players in the 2006-2007 season:
        Udonis Haslem: 844 points
        Dwayne Wade: 1397 points
        James Posey: 550 points
        ...
        """
    elif "2007-2008" in query:
        return ("The number of total rebounds for Dwayne Wade in "
                "the Miami Heat season 2007-2008 is 214.")
    elif "2008-2009" in query:
        return ("The number of total rebounds for Dwayne Wade in "
                "the Miami Heat season 2008-2009 is 398.")
    return "No data found."


def percentage_change_tool(start: float, end: float) -> float:
    """
    A tool to calculate the percentage change between two numbers.
    :param start:
    :param end:
    :return:
    """
    return ((end - start) / start) * 100


planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should "
                "be the first to engage when given a new task.",
    model_client=model_client,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        WebSearchAgent: Searches for information
        DataAnalystAgent: Performs calculations

    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="An agent for searching information on the web.",
    tools=[search_web_tool],
    model_client=model_client,
    system_message="""
    You are a web search agent.
    Your only tool is search_tool - use it to find information.
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """,
)

data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="An agent for performing calculations.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""
    You are a data analyst.
    Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
    If you have not seen the data, ask for it.
    """,
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination


def candidate_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> List[str]:
    """
    A custom candidate function to determine which agents should participate
    in the next round of conversation based on the current state of the chat.
    The function ensures that the planning agent is always the first to respond
    to a new user task, and it selects the appropriate agents based on the
    planning agent's output.
    :param planning_agent: The planning agent responsible for task delegation.
    :param web_search_agent: The web search agent responsible for information retrieval.
    :param data_analyst_agent: The data analyst agent responsible for data analysis.
    :param messages:
    :return:
    """
    # keep planning_agent first one to plan out the tasks
    if messages[-1].source == "user":
        return [planning_agent.name]

    # if the previous agent is planning_agent and if it explicitly asks for web_search_agent
    # or data_analyst_agent or both (in-case of re-planning or re-assignment of tasks)
    # then return those specific agents
    last_message = messages[-1]
    if last_message.source == planning_agent.name:
        participants = []
        if web_search_agent.name in last_message.to_text():
            participants.append(web_search_agent.name)
        if data_analyst_agent.name in last_message.to_text():
            participants.append(data_analyst_agent.name)
        if participants:
            return participants  # SelectorGroupChat will select from the remaining two agents.

    # we can assume that the task is finished once the web_search_agent
    # and data_analyst_agent have took their turns, thus we send
    # in planning_agent to terminate the chat
    previous_set_of_agents = set(message.source for message in messages)
    if (web_search_agent.name in previous_set_of_agents and
            data_analyst_agent.name in previous_set_of_agents):
        return [planning_agent.name]

    # if no-conditions are met then return all the agents
    return [planning_agent.name, web_search_agent.name, data_analyst_agent.name]


team = SelectorGroupChat(
    [planning_agent, web_search_agent, data_analyst_agent],
    model_client=model_client,
    termination_condition=termination,
    candidate_func=candidate_func,
)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """
    task = ("Who was the Miami Heat player with the highest "
            "points in the 2006-2007 season, "
            "and what was the percentage change in his total "
            "rebounds between the 2007-2008 and 2008-2009 seasons?")

    # Use asyncio.run(...) if you are running this in a script.
    await Console(team.run_stream(task=task))

    await model_client.close()


asyncio.run(main())
