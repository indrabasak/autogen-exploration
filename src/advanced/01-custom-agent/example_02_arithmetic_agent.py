"""
An example of creating custom agents to perform arithmetic operations
in a multi-agent chat system using Azure OpenAI.

In this example, we define several agents that can perform basic arithmetic
operations: addition, multiplication, subtraction, division, and identity.
Each agent takes a number as input and applies its respective operation.

We then set up a SelectorGroupChat that allows these agents to communicate
with each other. The chat is initialized with a task to transform a given
number (10) into another number (25) using the available operations.

The conversation continues until a maximum of 10 messages have been exchanged,
and the output is streamed to the console.
"""
import asyncio
import os
from typing import Callable, Sequence, List

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


class ArithmeticAgent(BaseChatAgent):
    """
    An agent that performs a specific arithmetic operation on a given number.
    The operation is defined by the operator_func passed during initialization.

    :param name: Name of the agent.
    :param description: Description of the agent's role.
    :param operator_func: A callable that takes an integer and returns an integer
                          after applying the arithmetic operation.
    """

    def __init__(self, name: str, description: str, operator_func: Callable[[int], int]) -> None:
        super().__init__(name, description=description)
        self._operator_func = operator_func
        self._message_history: List[BaseChatMessage] = []

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage],
                          cancellation_token: CancellationToken) -> Response:
        # Update the message history.
        # NOTE: it is possible the messages is an empty list,
        # which means the agent was selected previously.
        self._message_history.extend(messages)
        # Parse the number in the last message.
        assert isinstance(self._message_history[-1], TextMessage)
        number = int(self._message_history[-1].content)
        # Apply the operator function to the number.
        result = self._operator_func(number)
        # Create a new message with the result.
        response_message = TextMessage(content=str(result), source=self.name)
        # Update the message history.
        self._message_history.append(response_message)
        # Return the response.
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


async def run_number_agents() -> None:
    """
    Function to run the arithmetic agents in a selector group chat
    and stream their responses to the console.
    :return: None
    """
    # Create agents for number operations.
    add_agent = ArithmeticAgent("add_agent",
                                "Adds 1 to the number.",
                                lambda x: x + 1)
    multiply_agent = ArithmeticAgent("multiply_agent",
                                     "Multiplies the number by 2.",
                                     lambda x: x * 2)
    subtract_agent = ArithmeticAgent("subtract_agent",
                                     "Subtracts 1 from the number.",
                                     lambda x: x - 1)
    divide_agent = ArithmeticAgent("divide_agent",
                                   "Divides the number by 2 and rounds down.",
                                   lambda x: x // 2)
    identity_agent = ArithmeticAgent("identity_agent",
                                     "Returns the number as is.",
                                     lambda x: x)

    # The termination condition is to stop after 10 messages.
    termination_condition = MaxMessageTermination(10)

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

    # Create a selector group chat.
    selector_group_chat = SelectorGroupChat(
        [add_agent, multiply_agent, subtract_agent, divide_agent, identity_agent],
        model_client=model_client,
        termination_condition=termination_condition,
        # Allow the same agent to speak multiple times, necessary for this task.
        allow_repeated_speaker=True,
        selector_prompt=(
            "Available roles:\n{roles}\nTheir job descriptions:\n{participants}\n"
            "Current conversation history:\n{history}\n"
            "Please select the most appropriate role for "
            "the next message, and only return the role name."
        ),
    )

    # Run the selector group chat with a given task and stream the response.
    task: List[BaseChatMessage] = [
        TextMessage(content="Apply the operations "
                            "to turn the given number into 25.", source="user"),
        TextMessage(content="10", source="user"),
    ]
    stream = selector_group_chat.run_stream(task=task)
    await Console(stream)


# Use asyncio.run(run_number_agents()) when running in a script.
asyncio.run(run_number_agents())
