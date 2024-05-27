# Import nessessary packages
import time
import json
import os
import random
from joblib import Parallel, delayed

# Import nessessary Class objects and functions
from models import UserLLM, AssistantLLM, ModeratorLLM

#CONSTANTS
# Cache model names
MODEL_NAMES = {
    "user": UserLLM().get_model_name(),
    "assistant": AssistantLLM().get_model_name(),
    "moderator": ModeratorLLM().get_model_name()
}
# File paths
DATA_GEN_FILE_PATH = "/home/varun/Varun/IFT/Chains/Automate/@Gen/@@rev2/1"

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
    # dict to have token counts
    conv_tokens = dict()

    # Create a UserLLM instance to handle user prompt
    user = UserLLM(conv_turns)

    # Generate initiation prompt if it is the first prompt in the conversation, with retries
    if is_first_prompt:
        retry_count = 0
        while retry_count < 3:
            try:
                prompt, token = user.generate_initiation_prompt(intent, domain)
                conv_tokens["user"] = token
                break
            except Exception as e:
                print(f'Error occurred: {e}. Retrying...')
                retry_count += 1
                continue
    else:
        # If not the first prompt, attempt to generate a continuation prompt, with retries
        retry_count = 0
        while retry_count < 3:
            try:
                prompt, token = user.generate_continuation_prompt(intent, domain)
                conv_tokens["user"] = token
                break
            except Exception as e:
                print(f'Error occurred: {e}. Retrying...')
                retry_count += 1
                continue
    
    # Create a AssistantLLM instance to handle response generation
    assistant = AssistantLLM(history=conv_turns)

    # Generate a response from the assistant based on the generated prompt
    response, token = assistant.respond_to_user_prompt(prompt)
    conv_tokens["assistant"] = token

    # Append the conversation turn (intent, user prompt, assistant response) to the conversation turns list
    conv_turns.append((intent, prompt, response))
    return conv_turns, conv_tokens

def conversation_loop(turns: int, intent: str, domain: str, conv_turns: list, doc_id: str, mod_out: list = [], name: str = 'C-', token_count: dict = {"user":0,"assistant":0,"moderator":0}):
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
        token_count (dict, optional): To keep track of token counts for each doc_id. Defaults to a dict template with all values being 0.
    """
    # Generate a conversation turn
    try:
        conv_turns, conv_token = generate_prompt(is_first_prompt=(len(conv_turns) == 0), intent=intent, domain=domain, conv_turns=conv_turns)
        token_count["user"] += conv_token["user"]
        token_count["assistant"] += conv_token["assistant"]
    except Exception as e:
        print(f'Exception:\n{e}')
        with open(f'{DATA_GEN_FILE_PATH}/@-errors.txt','a') as file:
            file.write(f"{conv_turns[0][0]},{domain},{name}")
        save_conversation(conv_turns, mod_out, domain, name, doc_id)
        return token_count
    
    # Check if the conversation reached the input turns
    if len(conv_turns) >= turns:
        # Save the conversation
        save_conversation(conv_turns, mod_out, domain, name, doc_id)
        return token_count

    else:
        # Attempt to generate a moderator ideas for next sub-intents, with retries
        retry_count = 0
        while retry_count < 3:
            try:
                mod_ideas, mod_token = ModeratorLLM(history=conv_turns).suggest_next_sub_intents(intent)
                token_count["moderator"] += mod_token
                if len(conv_turns) >= 2:
                    mod_ideas = random.sample(mod_ideas,random.randint(0,5))
                else:
                    mod_ideas = random.sample(mod_ideas,random.randint(1,5))
                break
            except Exception as e:
                print(f'Error occurred: {e}. Retrying...')
                retry_count += 1
                continue
        if retry_count == 3:
            with open(f'{DATA_GEN_FILE_PATH}/@-errors.txt','a') as file:
                file.write(f"{conv_turns[0][0]},{domain},{name}\n")
            save_conversation(conv_turns, mod_out, domain, name, doc_id)
            return token_count
        if len(mod_ideas) == 0:
            save_conversation(conv_turns, mod_out, domain, name, doc_id)
            return token_count

        # Append moderator ideas to the output
        mod_out.append({f"Turn{len(conv_turns)-1}":mod_ideas})

        # Loop through the moderator ideas and start new conversation branches
        for index, mod_idea in enumerate(mod_ideas, start=1):
            next_name = name + str(index) + '-'
            token_count = conversation_loop(turns, mod_idea, domain, conv_turns.copy(), doc_id, mod_out.copy(), next_name, token_count)
        
        return token_count

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
    filename = f"{DATA_GEN_FILE_PATH}/{doc_id}/{name[0:-1]}.json"
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

def save_token_count(doc_id: str, token_count: dict):
    """
    Save token count data to a JSON file.

    Args:
        doc_id (str): Identifier for the conversation tree.
    """
    # Define the file path
    filename = f"{DATA_GEN_FILE_PATH}/@-token_counts.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Read the existing data from the JSON file
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except:
        data = []
    
    # Prepare token data structure
    token_data = {
        "doc_id": doc_id,
        "User LLM": {
            "token count": token_count["user"],
            "token cost": f'$ {(token_count["user"] / 1000000) * 0.50}'
        },
        "Assistant LLM": {
            "token count": token_count["assistant"],
            "token cost": f'$ {(token_count["assistant"] / 1000000) * 0.50}'
        },
        "Moderator LLM": {
            "token count": token_count["moderator"],
            "token cost": f'$ {(token_count["moderator"] / 1000000) * 0.50}'
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
    print(f"Staring {intent} and {domain}")
    # Start the conversation loop
    token_count = conversation_loop(turns=turns, intent=intent, domain=domain, conv_turns=[], doc_id=doc_id)

    # Save token count data after conversation completion
    save_token_count(doc_id, token_count)

    print(f"Done {intent} and {domain}")

# start(turns=6, intent="General-purpose Coding Queries", domain="Loops", doc_id='GP_Code')

with open("/home/varun/Varun/IFT/Chains/Automate/Files/Sample_Gen_1/input_re.txt",'r') as file:
    lines = file.readlines()

def process_input(index, inp):
    try:
        start(turns=4, intent=inp[0], domain=inp[1], doc_id=index)
    except Exception as e:
        print(e)
        return

input_list = []

for line in lines:
    intent, domain = line.strip().split(',')
    input_list.append((intent.strip(),domain.strip()))

Parallel(n_jobs=2)(delayed(process_input)(index, inp) for index, inp in enumerate(input_list))
