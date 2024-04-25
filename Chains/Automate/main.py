# Import nessessary packages
import time
import json
import os
import random

# Import nessessary Class objects and functions
from models import UserLLM, AssistantLLM, ModeratorLLM, get_token_count

#CONSTANTS
# Cache model names
MODEL_NAMES = {
    "user": UserLLM().get_model_name(),
    "assistant": AssistantLLM().get_model_name(),
    "moderator": ModeratorLLM().get_model_name()
}

def generate_prompt(is_first_prompt: bool, intent: str, domain: str, conv_turns: list) -> list:
    """
    Generate a user prompt and response prompt one turn of conversation.

    Args:
        is_first_prompt (bool): Indicates whether it's the first prompt in the conversation.
        intent (str): The intent of the conversation.
        domain (str): The domain or topic of the conversation.
        conv_turns (list): List of tuples representing conversation turns.

    Returns:
        list: Updated conversation turns.
    """
    # Create a UserLLM instance to handle user prompt
    user = UserLLM(conv_turns)

    # Generate initiation prompt if it is the first prompt in the conversation
    if is_first_prompt:
        prompt = user.generate_initiation_prompt(intent, domain)
    else:
        # If not the first prompt, attempt to generate a continuation prompt, with retries
        retry_count = 0
        while retry_count < 3:
            try:
                prompt = user.generate_continuation_prompt(intent, domain)
                break
            except Exception as e:
                print(f'Error occurred: {e}. Retrying...')
                retry_count += 1
                continue
    
    # Create a AssistantLLM instance to handle response generation
    assistant = AssistantLLM(history=conv_turns)

    # Generate a response from the assistant based on the generated prompt
    response = assistant.respond_to_user_prompt(prompt)

    # Append the conversation turn (intent, user prompt, assistant response) to the conversation turns list
    conv_turns.append((intent, prompt, response))
    return conv_turns

def conversation_loop(turns: int, intent: str, domain: str, conv_turns: list, doc_id: str, mod_out: list = [], name: str = 'C'):
    """
    Perform conversation loop recursively until the specified number of turns is reached.

    Args:
        turns (int): The number of turns for the conversation.
        intent (str): The intent of the conversation.
        domain (str): The domain or topic of the conversation.
        conv_turns (list): List of tuples representing conversation turns.
        doc_id (str): Identifier for the conversation tree.
        mod_out (list, optional): List to store moderator outputs. Defaults to [].
        name (str, optional): Naming convention for conversation tree branches. Defaults to '1'.
    """
    # Generate a conversation turn
    conv_turns = generate_prompt(is_first_prompt=(len(conv_turns) == 0), intent=intent, domain=domain, conv_turns=conv_turns)
    
    # Check if the conversation reached the input turns
    if len(conv_turns) >= turns:
        # Save the conversation
        save_conversation(conv_turns, mod_out, domain, name, doc_id)
    else:
        # Attempt to generate a moderator ideas for next sub-intents, with retries
        retry_count = 0
        while retry_count < 3:
            try:
                mod_ideas = ModeratorLLM(history=conv_turns).suggest_next_sub_intents(intent)
                mod_ideas = random.sample(mod_ideas,random.randint(0,5))
                break
            except Exception as e:
                print(f'Error occurred: {e}. Retrying...')
                retry_count += 1
                continue

        # Append moderator ideas to the output
        mod_out.append({f"Turn{len(conv_turns)-1}":mod_ideas})

        # Loop through the moderator ideas and start new conversation branches
        for index, mod_idea in enumerate(mod_ideas, start=1):
            next_name = f"-{name + str(index)}"
            conversation_loop(turns, mod_idea, domain, conv_turns.copy(), doc_id, mod_out, next_name)

def save_conversation(conv_turns: list, mod_out: list, domain: str, name: str, doc_id: str):
    """
    Save conversation data to a JSON file.

    Args:
        conv_turns (list): List of tuples representing conversation turns.
        mod_out (list): List containing moderator outputs.
        domain (str): The domain or topic of the conversation.
        name (str): Name produced for the conversation tree branch.
        doc_id (str): Identifier for the conversation tree.
    """
    # Define the file path
    filename = f"/home/varun/Varun/IFT/Chains/Automate/@Mixtral_Examples/{doc_id}/{name}.json"
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write conversation data to the JSON file
    with open(filename, "w") as file:
        conversation = []
        # Add Conversation metadata
        conversation.append({"id": doc_id})
        conversation.append({"intent": conv_turns[0][0]})
        conversation.append({"domain": domain})
        model_list = MODEL_NAMES
        conversation.append({"model list": model_list})
        conversation.append({"timestamp": time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(time.time()))})
        # Add Conversation interactions
        interactions = []
        for intent, prompt, response in conv_turns:
            turn = {
                "intent": intent,
                "user": prompt,
                "assistant": response
            }
            interactions.append(turn)
        conversation.append({"interactions": interactions})
        # Add Moderator outputs
        conversation.append({"moderator": mod_out})
        json.dump(conversation, file, indent=4)

def save_token_count(doc_id: str):
    """
    Save token count data to a JSON file.

    Args:
        doc_id (str): Identifier for the conversation tree.
    """
    # Define the file path
    filename = "/home/varun/Varun/IFT/Chains/Automate/token_counts.json"

    # Read the existing data from the JSON file
    with open(filename, "r") as file:
        data = json.load(file)
    
    # Get token counts
    token_list = get_token_count()

    # Prepare token data structure
    token_data = {
        "doc_id": doc_id,
        "User LLM": {
            "token count": token_list[0],
            "token cost": f"$ {(token_list[0] / 1000000) * 0.50}"
        },
        "Assistant LLM": {
            "token count": token_list[1],
            "token cost": f"$ {(token_list[1] / 1000000) * 0.50}"
        },
        "Moderator LLM": {
            "token count": token_list[2],
            "token cost": f"$ {(token_list[2] / 1000000) * 0.50}"
        }
    }

    # Append token data to existing data
    data.append(token_data)

    # Write updated data to JSON file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def start(turns: int, intent: str, domain: str, doc_id: str):
    """
    Start the conversation process.

    Args:
        turns (int): The number of turns for the conversation.
        intent (str): The intent of the conversation.
        domain (str): The domain or topic of the conversation.
        doc_id (str): Identifier for the conversation tree.
    """
    # Start the conversation loop
    conversation_loop(turns=turns, intent=intent, domain=domain, conv_turns=[], doc_id=doc_id)

    # Save token count data after conversation completion
    save_token_count(doc_id)

start(turns=6, intent="to reorder paragraphs", domain="Article", doc_id='sports_example_3')
