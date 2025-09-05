# Autogen Exploration

A Python project for exploring [Autogen](https://microsoft.github.io/autogen/0.2/) capabilities. 
Autogen is a framework that enables developers to create AI agents that can interact with each other and with 
external tools to achieve complex tasks.

## Installation
 - Clone the repository.
 - Create a new `.env` file in the root directory and add your OpenAI APIs key as shown below:

```aiignore
AZURE_OPENAI_API_INSTANCE_NAME=<AZURE ENDPOINT>
AZURE_OPENAI_API_DEPLOYMENT_NAME=<MODEL DEPLOYMENT NAME>
AZURE_OPENAI_API_VERSION=<OPENAI API VERSION>

AZURE_AUTHORITY_HOST=<AUTHORITY HOST>
AZURE_FEDERATED_TOKEN_FILE=<PATH TO FEDERATED TOKEN FILE>
AZURE_TENANT_ID=<TENANT ID>
AZURE_CLIENT_ID=<CLIENT ID>
AZURE_CLIENT_SECRET=<CLIENT SECRET>
```
 - This project uses `uv`, a Python package and project manager, to manage dependencies and run scripts. 
Make sure you have `uv` installed.

```
## Usage
To run `lesson-one.py`, use the following command:
```bash
uv run src/quick-start/lesson_one.py
```
Same way you can run other scripts in the `src/quick-start` folder.

```
## Formatting
To format the code, use the `ruff` command. Here's an example command to format a specific file:
```bash
uv run ruff format ./src/introduction/01-asst-agent/example_03_streaming_msg.py
```
For linting, you can use the `pylint` command. Here's an example command to lint a specific file:
```bash
uv run pylint ./src/introduction/01-asst-agent/example_03_streaming_msg.py
```