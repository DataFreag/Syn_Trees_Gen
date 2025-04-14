# Syn_Trees_Gen

**Syn_Trees_Gen** is a Python project designed to automatically generate synthetic conversation trees using large language models (LLMs). The repository simulates multi-turn, branching dialogues by recursively generating conversation turns between a simulated user and an assistant, while a moderator suggests sub-intents for branching.

## Features

- **Synthetic Conversation Generation:**  
  Automatically creates conversation trees by initiating dialogue with a given intent and domain, and recursively generates follow-up conversations based on moderator-suggested sub-intents.

- **LLM Integration:**  
  Uses different LLM roles:
  - **UserLLM:** Generates initiation and continuation prompts.
  - **AssistantLLM:** Produces responses based on conversation history.
  - **ModeratorLLM:** Suggests new sub-intents to drive conversation branching.

- **Parallel Processing:**  
  Uses Joblib to process multiple conversation trees concurrently, allowing for scalable large-scale data generation.

- **Token Usage Tracking:**  
  Tracks token counts for each model call to monitor usage and cost, with detailed logs saved for further analysis.

## Packages Used

- **Python:** Core programming language.
- **LangChain:** Manages prompts, chaining, and output parsing for seamless LLM integration.
- **Joblib:** Enables parallel processing for enhanced scalability.
- **LLM Integrations:** Utilizes models such as ChatAnyscale via LangChain wrappers.

