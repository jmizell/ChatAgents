# Random Langchain Python Scripts Collection

This repository contains a collection of Python scripts that interact with various language models and APIs. Each script serves a specific purpose, as outlined below. 
This is just me trying to understand how the models work, and what they;re capable of.

## Table of Contents

- [Scripts](#scripts)
  - [autogpt.py](#autogptpy)
  - [chat.py](#chatpy)
  - [conversational_agent.py](#conversational_agentpy)
  - [plan_and_exec_agent.py](#plan_and_exec_agentpy)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)

## Scripts

### autogpt.py
https://python.langchain.com/docs/use_cases/more/agents/autonomous_agents/autogpt

This script runs an AutoGPT agent.

### chat.py
https://platform.openai.com/docs/api-reference/chat

This script calls the OpenAI chat completion API.

### conversational_agent.py
https://python.langchain.com/docs/modules/agents/agent_types/chat_conversation_agent

A React agent that comes with various tools and a Gradio interface.

### plan_and_exec_agent.py
https://python.langchain.com/docs/modules/agents/agent_types/plan_and_execute

A plan and execute agent.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run any of the scripts, use the following command:

```bash
python <script_name>.py
```

Replace `<script_name>` with the name of the script you want to run (e.g., `autogpt`, `chat`, etc.).

## Environment Variables

Before running the scripts, make sure to set the OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```
