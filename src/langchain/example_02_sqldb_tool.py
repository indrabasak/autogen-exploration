"""
Example of using the SQL Database Toolkit from LangChain with Azure OpenAI and Autogen.
The assistant agent is given access to a MySQL database and can use SQL
queries to answer questions about the data in the database.
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI
from sqlalchemy import Engine, create_engine

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

llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    model=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    api_version=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_API_INSTANCE_NAME"),
    azure_ad_token_provider=token_provider,
    temperature=0,
)


def get_engine_for_mysql_db() -> Engine:
    """
    Create a SQLAlchemy engine for a MySQL database using environment variables.
    Expects the following environment variables to be set:
    :return:
    """
    user = os.environ.get("MYSQL_USER")
    password = os.environ.get("MYSQL_PASSWORD")
    host = os.environ.get("MYSQL_HOST")
    port = os.environ.get("MYSQL_PORT")
    database = os.environ.get("MYSQL_DATABASE")
    ssl_args = {"ssl": {"rejectUnauthorized": False}}
    return create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}",
                         connect_args=ssl_args)


# Create the engine and database wrapper.
engine = get_engine_for_mysql_db()
db = SQLDatabase(engine)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# Create the LangChain tool adapter for every tool in the toolkit.
tools = [LangChainToolAdapter(tool) for tool in toolkit.get_tools()]

# Create the assistant agent.
agent = AssistantAgent(
    "assistant",
    tools=tools,  # type: ignore
    model_client_stream=True,
    model_client=model_client,
    system_message="Respond with 'TERMINATE' if the task is completed.",
)

# Create termination condition.
termination = TextMentionTermination("TERMINATE")

# Create a round-robin group chat to iterate the single agent over multiple steps.
chat = RoundRobinGroupChat([agent], termination_condition=termination)


async def main():
    """
    Main function to run the team of agents.
    :return:
    """

    # Run the chat.
    # await Console(chat.run_stream(task="Show some tables in the database"))
    # task = 'What prices are there for the offering with id OD-000163, direct, BIC; Please provide the SQL query used.'
    task = "What's the name of offering with id OD-127409?"
    await Console(chat.run_stream(task=task))

    await model_client.close()


asyncio.run(main())
