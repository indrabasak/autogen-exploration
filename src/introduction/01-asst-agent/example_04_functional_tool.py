"""
Example demonstrating the creation of a functional tool for use with
AssistantAgent in AutoGen.

This script defines a Python function as a web search tool, wraps it with FunctionTool,
and prints the tool's schema. The tool can be integrated into an
AssistantAgent for enhanced capabilities.

Dependencies:
- autogen_core

Usage:
    python src/introduction/01-asst-agent/example_04_functional_tool.py
"""

from autogen_core.tools import FunctionTool


# Define a tool using a Python function.
async def web_search_func(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


# This step is automatically performed inside the AssistantAgent if the tool is a Python function.
web_search_function_tool = FunctionTool(
    web_search_func, description="Find information on the web"
)
# The schema is provided to the model during AssistantAgent's on_messages call.
print(web_search_function_tool.schema)
