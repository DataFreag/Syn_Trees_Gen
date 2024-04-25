import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from tqdm import tqdm
import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from operator import itemgetter
import re
import random


os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://prompt-dashboard.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = ""

class Parser:
    '''This class contain all Parser defined for every LLM'''
    def TurnInitiatorParser(self,output):
        # print(output.content)
        print('##############')
        split_list = output.content.split('\n\n')
        filtered_list = [item for item in split_list if re.compile(r'^\d+\.').match(item)]
        filtered_list = [item[3:] for item in filtered_list if item != '']
        return filtered_list
    
    def UserParser(self,output):
        # print(output.content)
        print('##############')
        result = output.content.split(':',1)[-1]
        return result
    
    def AssistantParser(self,output):
        # print(output.content)
        print('##############')
        return output.content

    def ModeratorParser(self,output):
        # print(output.content)
        print('##############')
        output = output.content
        split_list = output.split('\n')
        filtered_list = [item for item in split_list if re.compile(r'^\d+\.').match(item)]
        filtered_list = [item[3:] for item in filtered_list if item != '']
        return filtered_list

class TurnInitiatorLLM:

    def __init__(self,azure_deployment="data-gen",temperature=0.7):
        self.model = AzureChatOpenAI(azure_deployment=azure_deployment,temperature=temperature)
        pre_template_list = [
    ("system", "Imagine you are a Chat Initiator LLM conversational designer tasked with brainstorming a list of innovative and engaging conversation ideas for a specific intent '{intent}' and domain '{domain}'. Explore various approaches that can be utilized to achieve {type} and engaging conversation with an AI model and just provide a list of ideas only within this intent and domain pair.")
    ]
        self.template = ChatPromptTemplate.from_messages(pre_template_list)
        self.parser = Parser().TurnInitiatorParser
        self.chain = self.template | self.model | self.parser
        pass

    def generate_conversation_ideas(self, intent, domain):
        result = []
        for way in ['simple']:  #TODO : ['simple', 'creative']
            retry_count = 0
            while retry_count < 3:
                ideas = self.chain.invoke({"intent": intent, "domain": domain, "type": way})
                if len(ideas) > 0:
                    if len(ideas)>=5:
                        result.extend(random.sample(ideas, 1))  #TODO change 5
                    else:
                        result.extend(ideas)
                    break
                else:
                    retry_count += 1
        return result

class UserLLM:
    def __init__(self,azure_deployment="data-gen",temperature=0.7):
        self.model = AzureChatOpenAI(azure_deployment=azure_deployment,temperature=temperature)
        pre_template_list_first = [
    ("system", "I want you to act as a User who wants to start a conversation with a AI model. Please provide a creative and helpful prompt which can start a conversation from the given idea: '{turninit}'. Remember to give only the prompt which starts this conversation with a model as a user.")
    ]
        pre_template_list_next = [
    ("system", "I want you to act as a User on extending a dialogue with an AI model. Craft an engaging and constructive prompt that seamlessly carries on the conversation from the provided starting point: '{turninit}'. Your task is to provide the next step in the conversation based on this exchange:\nUser: '{prompt}\nAssistant: '{response}\nRemember, your prompt should initiate the conversation only from the user's perspective and response only user's prompt.")
    ]
        
        self.template_first = ChatPromptTemplate.from_messages(pre_template_list_first)
        self.template_next = ChatPromptTemplate.from_messages(pre_template_list_next)

        self.parser = Parser().UserParser
        self.first_chain = self.template_first | self.model | self.parser
        self.next_chain = self.template_next | self.model | self.parser
        pass
    
    def generate_first_prompt(self, idea):
        result = self.first_chain.invoke({"turninit":f"{idea}"})
        return result
    
    def generate_next_prompt(self, idea, prompt, response):
        result = self.next_chain.invoke({"turninit":f"{idea}","prompt":f"{prompt}","response":f"{response}"})
        return result

class AssistantLLM:
    def __init__(self,azure_deployment="data-gen",temperature=0.7):
        self.model = AzureChatOpenAI(azure_deployment=azure_deployment,temperature=temperature)
        pre_template_list = [
    ("system", "You are a helpful assistant who encourage engaging and meaningful conversations."),
    ("human","{prompt}")
    ]
        self.template = ChatPromptTemplate.from_messages(pre_template_list)
        self.parser = Parser().AssistantParser
        self.chain = self.template | self.model | self.parser
        pass
    
    def respond_to_user_prompt(self, user_prompt):
        result = self.chain.invoke({"prompt":f"{user_prompt}"})
        return result

class ModeratorLLM:
    def __init__(self,azure_deployment="data-gen",temperature=0.7):
        self.model = AzureChatOpenAI(azure_deployment=azure_deployment,temperature=temperature)
        pre_template_list = [
    ("system", "You are a helpful moderator who assists in suggesting ideas for enriching conversation paths. The initial conversation began with the idea of '{idea}'. Now, we require a second set of User and Assistant Conversations. As a moderator, your role is to enhance the conversation by proposing ideas for a more engaging dialogue. Your idea will serve as the second User prompt for the second conversation set. Please provide a list sub-intents to generate prompts for the second conversation set based on the given initial prompt and response pair.\nprompt: {prompt}\nresponse: {response}. Remember to provide a list of sub-intents for a engaging conversation.")
    ]
        self.template = ChatPromptTemplate.from_messages(pre_template_list)
        self.parser = Parser().ModeratorParser
        self.chain = self.template | self.model | self.parser
        pass
    
    def suggest_next_sub_intents(self,idea ,prompt,response):
        result = []
        retry_count = 0
        while retry_count < 3:
            ideas = self.chain.invoke({"idea":f"{idea}","prompt":f"{prompt}","response":f"{response}"})
            if len(ideas) > 0:
                if len(ideas)>=5:
                    result.extend(random.sample(ideas, 2))  #TODO change 5
                else:
                    result.extend(random.sample(ideas, len(ideas)))
                break
            else:
                retry_count += 1
        return result

