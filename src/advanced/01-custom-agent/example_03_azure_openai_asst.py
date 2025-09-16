"""
An example of creating a custom agent using Azure OpenAI as the underlying model.
Make sure to set the following environment variables before running the script:

- AZURE_OPENAI_API_INSTANCE_NAME: The base URL of your Azure OpenAI resource
 (e.g., https://your-resource-name.openai.azure.com/)
- AZURE_OPENAI_API_DEPLOYMENT_NAME: The deployment name of your model in Azure OpenAI
- AZURE_OPENAI_API_VERSION: The API version to use (e.g., 2023-05-15)
"""
import asyncio
import os
from typing import Sequence, AsyncGenerator

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage, BaseAgentEvent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, UserMessage, SystemMessage
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()


class AzureOpenAIAssistantAgent(BaseChatAgent):
    """
    An example of a custom assistant agent using Azure OpenAI as the underlying model.
    The agent maintains a conversation context and can respond to messages.
    """

    def __init__(
            self,
            name: str,
            description: str = "An agent that provides assistance with ability to use tools.",
            system_message: str
                            | None = "You are a helpful assistant that can respond to messages. "
                                     "Reply with TERMINATE when the task has been completed.",
    ):
        super().__init__(name=name, description=description)
        self._model_context = UnboundedChatCompletionContext()
        self._system_message = system_message

        token_provider = AzureTokenProvider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        self._model_client = AzureOpenAIChatCompletionClient(
            azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
            model=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_API_INSTANCE_NAME"),
            azure_ad_token_provider=token_provider,
            temperature=0.3)

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage],
                          cancellation_token: CancellationToken) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message

        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")

        return final_response

    async def on_messages_stream(
            self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        # Add messages to the model context
        for msg in messages:
            await self._model_context.add_message(msg.to_model_message())

        # Get conversation history
        history = [
            (msg.source if hasattr(msg, "source") else "system")
            + ": "
            + (msg.content if isinstance(msg.content, str) else "")
            + "\n"
            for msg in await self._model_context.get_messages()
        ]

        # Create a system message
        system_message = SystemMessage(content=self._system_message)

        # Create a user message
        user_message = UserMessage(content=f"History: "
                                           f"{history}\nGiven the history, "
                                           f"please provide a response",
                                   source="user")

        # Combine messages into a list for the client's create method
        messages = [system_message, user_message]

        # Generate response using Azure OpenAI
        response = await self._model_client.create(messages)

        # Create usage metadata
        usage = usage = response.usage

        # Add response to model context
        await self._model_context.add_message(AssistantMessage(
            content=response.content, source=self.name))

        # Yield the final response
        yield Response(
            chat_message=TextMessage(content=response.content,
                                     source=self.name, models_usage=usage),
            inner_messages=[],
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the assistant by clearing the model context."""
        await self._model_context.clear()


async def main():
    """
    Main function to run the Azure OpenAI Assistant Agent and stream its responses.
    :return:
    """
    azure_open_ai_assistant = AzureOpenAIAssistantAgent("azure_open_ai_assistant")
    await Console(azure_open_ai_assistant.run_stream(task="What is the capital of New York?"))


asyncio.run(main())
