from models import *
import time

intent = "Short stories writing (Writing short stories.)"
domain = "Sport"

def start(turns, intent, domain):
    turn = TurnInitiatorLLM()
    user = UserLLM()
    assistant = AssistantLLM()
    moderator = ModeratorLLM()

    conv_ideas = turn.generate_conversation_ideas(intent, domain)

    def generate_turn(idea, conv_turns):
        if len(conv_turns) == 0:
            prompt = user.generate_first_prompt(idea)
        else:
            prompt = user.generate_next_prompt(idea, conv_turns[-1][0], conv_turns[-1][1])
        response = assistant.respond_to_user_prompt(prompt)
        conv_turns.append((prompt, response))
        return conv_turns

    def loop_initiation(turns, idea, conv_turns):
        conv_turns = generate_turn(idea, conv_turns)
        if len(conv_turns) >= turns:
            print('Saving conv')
            print(conv_turns)
            save_conversation(conv_turns)
        else:
            print('Resuming new conv')
            mod_ideas = moderator.suggest_next_sub_intents(idea, conv_turns[-1][0], conv_turns[-1][1])
            for mod_idea in mod_ideas:
                print(len(conv_turns))
                loop_initiation(turns, mod_idea, conv_turns.copy())

    def save_conversation(conv_turns):
        filename = "conversation_" + str(time.time()) + ".json"
        with open(filename, "w") as file:
            conversation = []
            for prompt, response in conv_turns:
                conversation.append({"User":prompt})
                conversation.append({"Assistant":response})
            json.dump(conversation, file, indent=4)

    for idea in conv_ideas:
        print('Starting new conv')
        loop_initiation(turns, idea, conv_turns=[])

start(3, intent, domain)
