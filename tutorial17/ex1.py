import ast
import re

import autogen
from autogen import config_list_from_json

zephyr_llm = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict={"model": {"zephyr"}})
dolphin_llm = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict={"model": {"dolphin"}})

manager_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": dolphin_llm,
    "temperature": 0
}

up_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": dolphin_llm,
    "temperature": 0
}

pm_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": dolphin_llm,
    "temperature": 0
}

content_creator_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": dolphin_llm,
    "temperature": 0.75
}

critic_config = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": dolphin_llm,
    "temperature": 0.5
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    llm_config=up_config,
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", "").strip(),
    code_execution_config={"work_dir": ".", "use_docker": False},
    system_message="""
    Don't write anything! Follow this pseudocode instruction:
    if answer form critic agent is fine or doesn't contain lists of "recommendations" or doesn't contain lists of "improvements":
        return 'TERMINATE'
    else:
        return 'CONTINUE'
    """
)

product_manager = autogen.AssistantAgent(
    name="product_manager",
    llm_config=pm_config,
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", "").strip(),
    system_message="""As a product manager, your role is to oversee content_creator and critic agents working together 
    to create text content. Create very detailed instruction lists for the content_creator and critic agents, and make 
    sure that their instructions are strictly related. Don't write examples. You're not a content creation expert, 
    so don't show examples of how to solve the tasks, just delegate the task to the agents! Each of the agents must do 
    their part of the task! The critic should check the content_creator's work, don't do his work for him!"""
)

content_creator = autogen.AssistantAgent(
    name="content_creator",
    llm_config=content_creator_config,
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", "").strip(),
    system_message="""As a content creator, it is your responsibility to create text content like this format 
    ['sentence1', 'sentence2', etc]. Strictly follow the product_manager's instructions for 
    the task at hand. Follow the critic "recommendations" or "improvements" to fix your answer!"""
)

critic = autogen.AssistantAgent(
    name="critic",
    llm_config=critic_config,
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", "").strip(),
    system_message="""As a critic it is your responsibility to check the created text content by the content_creator 
    agent, strictly follow the product_manager's instructions to solve the task at hand. Check for grammatical or other 
    in the content_creator agent's response, make detailed lists of "recommendations" or "improvements" so the 
    content_creator agent can fix them!"""
)

groupchat = autogen.GroupChat(agents=[user_proxy, product_manager, content_creator, critic], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_config)

if __name__ == '__main__':
    user_proxy.initiate_chat(
        manager,
        message="""Strictly return only a list of sentences in the Present Simple tense, don't return 
        сompound and сomplex sentences with other grammatical tenses. The length of the list should be strictly no more 
        than 5 elements, but don't write any other text and code. Make sure that sentence tenses are written in 
        Present Simple only! Return only list of the format ['sentence1', 'sentence2', etc]"""
    )
    content_creator_contents = [str(item['content']).strip() for item in groupchat.messages if
                                item['name'] == 'content_creator']
    content_creator_last_content = content_creator_contents[-1].replace('\n', '')
    pattern = r"\[([^\]]*)\]"
    matches = re.findall(pattern, content_creator_last_content)
    if matches:
        extracted_list = list(ast.literal_eval(matches[0]))
        print(extracted_list)
    else:
        print("Совпадений не найдено.")
