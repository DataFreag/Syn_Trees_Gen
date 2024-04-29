# Import nessessary packages
import os
import json
import random
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_community.callbacks import get_openai_callback

# Model Integrations imports
from langchain_community.chat_models import ChatAnyscale
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models import ChatDeepInfra


#CONSTANTS
# File paths for prompts and API key
PROMPTS_FILE_PATH = '/home/varun/Varun/IFT/Chains/Automate/prompts.json'
API_KEY_FILE_PATH = '/home/varun/Varun/IFT/Chains/Automate/api_key.json'

# Load prompts from file
with open(PROMPTS_FILE_PATH, 'r') as prompts_file:
    prompts = json.load(prompts_file)

# Load API key from file
with open(API_KEY_FILE_PATH, 'r') as api_key_file:
    key = json.load(api_key_file)

# TODO: Set common environment variables for model
os.environ["OPENAI_BASE_URL"] = "https://api.endpoints.anyscale.com/v1"
os.environ["ANYSCALE_API_KEY"] = key.get("anyscale", "")

# Global variables for token count
user_token_count = 0
assistant_token_count = 0
moderator_token_count = 0

class ModelPool():
    def __init__(self, models):
        self.models = models
        random.shuffle(self.models)
        self.current_index = 0
    
    def get_model(self):
        model = self.models[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.models)
        return model

class Parser:
    """
    This class contains static methods to generate parsers for different
    """
    
    @staticmethod
    def user_parser() -> PydanticOutputParser:
        """
        Create a parser for User LLM

        Return:
            PydanticOutputParser: the parser configured to handle User LLM response
        """
        # Define a Pydantic model for user prompt
        class UserPydantic(BaseModel):
            prompt: str = Field(description="content of the prompt generated")

        return PydanticOutputParser(pydantic_object=UserPydantic)

    @staticmethod
    def assistant_parser() -> StrOutputParser:
        """
        Create a parser for Assistant LLM

        Return:
            StrOutputParser: the parser to get response in string
        """
        return StrOutputParser()

    @staticmethod
    def moderator_parser() -> PydanticOutputParser:
        """
        Create a parser for Moderator LLM

        Return:
            PydanticOutputParser: the parser configured to handle Moderator LLM response
        """
        # Define a Pydantic model for moderator ideas
        class ModeratorPydantic(BaseModel):
            intents: List[str] = Field(description="list of tasks")
        
        return PydanticOutputParser(pydantic_object=ModeratorPydantic)

class UserLLM:
    """
        A class representing a user in a conversational trees.

        Attributes:
            history (list): A list of tuples representing the conversation history.
            model (str): The name of the language model to use.
            temperature (float): The sampling temperature for model responses.

    """
    def __init__(self, history: list = [], model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1", temperature: float = 0.7):
        """
        Initialize the UserLLM instance. This class object has two chains: to initialize chat and to continue chat.

        Args:
            history (list, optional): A list of tuples representing the conversation history. Defaults to an empty list.
            model (str, optional): The name of the language model to use. Defaults to "mistralai/Mixtral-8x7B-Instruct-v0.1".
            temperature (float, optional): The sampling temperature for model responses. Defaults to 0.7.
        """
        # TODO: Initialize the model pool by listing models to avoid limiting errors
        self.model_pool = ModelPool([
            ChatAnyscale(model_name=model, temperature=temperature, anyscale_api_key=key.get("anyscale", ""))
        ])
        #Initialize the language model and parser
        self.model = self.model_pool.get_model()
        self.parser = Parser.user_parser()

        
        # Initialize prompt template for User prompt initiation chain
        self.template_init = PromptTemplate(
            template=f"{prompts.get('User_first', '')}",
            input_variables=["intent","domain"],
            partial_variables={"format_instructions":self.parser.get_format_instructions()},
        )

        # Concatenate conversation history into a single string
        chat_history = ''
        for _, user_prompt, assis_prompt in history:
            chat_history += f"User: \"{user_prompt}\"\nAssistant: \"{assis_prompt}\"\n"
        
        # Initialize prompt template for User prompt continuation chain from chat history
        self.template_cont = PromptTemplate(
            template= f"{chat_history + prompts.get('User_next', '')}",
            input_variables=["intent","domain"],
            partial_variables={"format_instructions":self.parser.get_format_instructions()},
        )

        # Define user chains using templates, model and parser
        self.first_chain = self.template_init | self.model | self.parser    # Chain to initiate conversation as User
        self.next_chain = self.template_cont | self.model | self.parser     # Chain to continue conversation as User
    
    def generate_initiation_prompt(self, intent: str ,domain: str) -> str:
        """
        Generate a prompt as a User to initiate conversation with assistant.

        Args:
            intent (str): The user's intent for the conversation.
            domain (str): The domain or topic of the conversation.

        Returns:
            str: The generated prompt for the user.
        """

        # Global variable for token count
        global user_token_count

        # Invoke the chain to generate the initiation prompt along with callback for token counts
        with get_openai_callback() as cb:
            result = self.first_chain.invoke({"intent":f"{intent}","domain":f"{domain}"})
            user_token_count += cb.total_tokens
        
        # Console prints
        # print('###User###')
        # print(result)
        return result.prompt
    
    def generate_continuation_prompt(self, intent: str, domain: str) -> str:
        """
        Generate a prompt as a User to continue conversation with assistant.

        Args:
            intent (str): The user's intent for the conversation.
            domain (str): The domain or topic of the conversation.

        Returns:
            str: The generated prompt for the user.
        """

        # Global variable for token count
        global user_token_count

        # Invoke the chain to generate the continuation prompt along with callback for token counts
        with get_openai_callback() as cb:
            result = self.next_chain.invoke({"intent":f"{intent}", "domain":f"{domain}"})
            user_token_count += cb.total_tokens
        
        # Console prints
        # print('###User###')
        # print(result)
        return result.prompt
    
    def get_model_name(self) -> str:
        """
        Get the name of the language model used by the user for logging.

        Returns:
            str: The name of the language model.
        """
        return self.model.model_name

class AssistantLLM():
    """
    A class representing an assistant in a conversational trees.

    Attributes:
        history (list): A list of tuples representing the conversation history.
        model (str): The name of the language model to use.
        temperature (float): The sampling temperature for model responses.
    """
    def __init__(self, history: list = [], model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1", temperature: float = 0.7):
        """
        Initialize the AssistantLLM instance.

        Args:
            history (list, optional): A list of tuples representing the conversation history. Defaults to an empty list.
            model (str, optional): The name of the language model to use. Defaults to "mistralai/Mixtral-8x7B-Instruct-v0.1".
            temperature (float, optional): The sampling temperature for model responses. Defaults to 0.7.
        """
        # TODO: Initialize the model pool by listing models to avoid limiting errors
        self.model_pool = ModelPool([
            ChatAnyscale(model_name=model, temperature=temperature, anyscale_api_key=key.get("anyscale", ""))
        ])
        #Initialize the language model and parser
        self.model = self.model_pool.get_model()
        self.parser = Parser.assistant_parser()

        # Initialize prompt template for Assistant prompt chain
        pre_template_list = [("system", "You are a helpful and toxicless assistant.")]
        for _, prompt, response in history:
            pre_template_list.extend([("human",f"{prompt}")])
            pre_template_list.extend([("ai",f"{response}")])
        pre_template_list.extend([("human","{prompt}")])
        self.template = ChatPromptTemplate.from_messages(pre_template_list)

        # Define assistant chain using template, model and parser
        self.chain = self.template | self.model | self.parser
    
    def respond_to_user_prompt(self, user_prompt: str) -> str:
        """
        Generate a response to a user's prompt.

        Args:
            user_prompt (str): The prompt provided by the user.

        Returns:
            str: The generated response from the assistant.
        """

        # Global variable for token count
        global assistant_token_count

        # Invoke the chain to generate the response prompt along with callback for token counts
        with get_openai_callback() as cb:
            result = self.chain.invoke({"prompt":user_prompt})
            assistant_token_count += cb.total_tokens

        # Console prints
        # print('###Assistant###')
        # print(result)
        return result

    def get_model_name(self) -> str:
        """
        Get the name of the language model used by the assistant for logging.

        Returns:
            str: The name of the language model.
        """
        return self.model.model_name

class ModeratorLLM:
    """
    A class representing a moderator in a conversational trees.

    Attributes:
        history (list): A list of tuples representing the conversation history.
        model (str): The name of the language model to use.
        temperature (float): The sampling temperature for model responses.
    """
    def __init__(self, history: list = [], model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1", temperature: float = 0.7):
        """
        Initialize the ModeratorLLM instance.

        Args:
            history (list, optional): A list of tuples representing the conversation history. Defaults to an empty list.
            model (str, optional): The name of the language model to use. Defaults to "mistralai/Mixtral-8x7B-Instruct-v0.1".
            temperature (float, optional): The sampling temperature for model responses. Defaults to 0.7.
        """
        # TODO: Initialize the model pool by listing models to avoid limiting errors
        self.model_pool = ModelPool([
            ChatAnyscale(model_name=model, temperature=temperature, anyscale_api_key=key.get("anyscale", ""))
        ])
        #Initialize the language model and parser
        self.model = self.model_pool.get_model()
        self.parser = Parser.moderator_parser()

        # Concatenate conversation history into a single string
        history_temp = ''
        for _, user_prompt, assis_prompt in history:
            history_temp += f"User: \"{user_prompt}\"\nAssistant: \"{assis_prompt}\"\n"
        
        # Initialize prompt template for Moderator response chain from chat history
        self.template = PromptTemplate(
            template=f"{history_temp + prompts.get('Moderator', '')}",
            input_variables=["intent"],
            partial_variables={"format_instructions":self.parser.get_format_instructions()},
        )
        
        # Create the conversation chain using the prompt template, model, and parser
        self.chain = self.template | self.model | self.parser
    
    def suggest_next_sub_intents(self ,intent: str) -> list:
        """
        Suggest next sub-intents based on the conversation history.

        Args:
            intent (str): The current intent.

        Returns:
            list: A list of suggested sub-intents.
        """

        # Global variable for token count
        global moderator_token_count

        # Invoke the chain to generate the moderator response along with callback for token counts
        with get_openai_callback() as cb:
            ideas = self.chain.invoke({"intent":intent})
            moderator_token_count += cb.total_tokens

        # Console prints
        # print('###Moderator###')
        # print(ideas)
        return ideas.intents

    def get_model_name(self) -> str:
        """
        Get the name of the language model used by the moderator for logging.

        Returns:
            str: The name of the language model.
        """
        return self.model.model_name
    
def get_token_count() -> list:
    """
    Get the total token count for each LLM used for Conversation trees.

    Returns:
        list: A list containing the token count for the user, assistant, and moderator.
    """
    token_list = [user_token_count,assistant_token_count,moderator_token_count]
    return token_list

