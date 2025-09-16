"""
This example demonstrates how to create a custom agent that counts down from a specified number.
The agent streams countdown messages and finally returns a "Done!" message.
"""
import asyncio
from typing import AsyncGenerator, List, Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken


class CountDownAgent(BaseChatAgent):
    """
    A simple countdown agent that counts down from a specified number.
    It streams each countdown number as a message and finally returns "Done!".
    """

    def __init__(self, name: str, count: int = 3):
        super().__init__(name, "A simple agent that counts down.")
        self._count = count

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage],
                          cancellation_token: CancellationToken) -> Response:
        # Calls the on_messages_stream.
        response: Response | None = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                response = message
        assert response is not None
        return response

    async def on_messages_stream(
            self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        for i in range(self._count, 0, -1):
            msg = TextMessage(content=f"{i}...", source=self.name)
            inner_messages.append(msg)
            yield msg
        # The response is returned at the end of the stream.
        # It contains the final message and all the inner messages.
        yield Response(chat_message=TextMessage(content="Done!", source=self.name),
                       inner_messages=inner_messages)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


async def run_countdown_agent() -> None:
    """
    Function to run the countdown agent and stream its responses.
    :return: None
    """
    # Create a countdown agent.
    countdown_agent = CountDownAgent("countdown", 5)

    # Run the agent with a given task and stream the response.
    async for message in countdown_agent.on_messages_stream([], CancellationToken()):
        if isinstance(message, Response):
            print(message.chat_message)
        else:
            print(message)


# Use asyncio.run(run_countdown_agent()) when running in a script.
asyncio.run(run_countdown_agent())
