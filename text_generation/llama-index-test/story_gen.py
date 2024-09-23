from llama_index.llms.ibm import WatsonxLLM
from llama_index.core.llms import ChatMessage
import os


def gen_prompt(title, summary, tags, tone, starting_text):
    q = f"""{title}

{summary}

Tags: {', '.join(tags)}
Tone: {', '.join(tone)}
Come up with a story:"""
    return q



with open("api_key.txt", "r") as f:
    api_key = f.read()
os.environ["WATSONX_APIKEY"] = api_key

watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-chat-v2",
    url="https://eu-de.ml.cloud.ibm.com",
    project_id="6de38ecd-f0bf-4bc0-9fd9-6af465569a36",
    max_new_tokens=1000
)

prompt = """The Enchanted Forest  
    A young adventurer enters an enchanted forest where time stands still and magic flows freely.  
    Tags: Fantasy, Adventure, Forests, Magic, Time, Mysteries  
    Tone: Enchantment, Discovery  
    [Neutral] The forest was alive with magic as Alex"""
title = prompt.split("\n")[0]
summary = prompt.split("\n")[1].strip()
tags = prompt.split("\n")[2].split(": ")[1].split(", ")
tone = prompt.split("\n")[3].split(": ")[1].split(", ")
starting_text = prompt.split("\n")[4].split("] ")[1]


q = gen_prompt(title, summary, tags, tone, starting_text)
print(q,end = '')

messages = [
    ChatMessage(role="system", content="In square brackets before each paragraph indicate in which emotion it should be recited. Only choose the following emotions (neutral, joy, sad, anger, fear, surprise, disgust)."),
    ChatMessage(role="user", content=q),
]

for chunk in watsonx_llm.stream_chat(messages):
    print(chunk.delta, end="")