"""
An example of using a reasoning LLM to select agents in a group chat.
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


token_provider = AzureTokenProvider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=os.getenv("AZURE_OPENAI_API_DEPLOYMENT_O_4_MINI"),
    model=os.getenv("AZURE_OPENAI_API_DEPLOYMENT_O_4_MINI"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_O_4_MINI"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_INSTANCE_NAME_O_4_MINI"),
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
    if "2007-2008" in query:
        return ("The number of total rebounds for Dwayne Wade in "
                "the Miami Heat season 2007-2008 is 214.")
    if "2008-2009" in query:
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


web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="An agent for searching information on the web.",
    tools=[search_web_tool],
    model_client=model_client,
    system_message="""Use web search tool to find information.""",
)

data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="An agent for performing calculations.",
    model_client=model_client,
    tools=[percentage_change_tool],
    system_message="""Use tool to perform calculation. If you have not seen the data, ask for it.""",
)

user_proxy_agent = UserProxyAgent(
    "UserProxyAgent",
    description="A user to approve or disapprove tasks.",
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination

SELECTOR_PROMPT = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then
select an agent
from {participants} to perform the next task.
When the task is complete, let the user approve or disapprove the task. \
                  """

team = SelectorGroupChat(
    [web_search_agent, data_analyst_agent, user_proxy_agent],
    model_client=model_client,
    termination_condition=termination,  # Use the same termination condition as before.
    selector_prompt=SELECTOR_PROMPT,
    allow_repeated_speaker=True,
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
