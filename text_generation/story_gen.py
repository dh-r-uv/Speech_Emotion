from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from classifier.emotion_classifier import get_evaluated_response

def gen_prompt(title, summary, tags, tone, starting_text):
    q = f"""{title}

{summary}

Tags: {', '.join(tags)}
Tone: {', '.join(tone)}
Come up with a story In square brackets before each paragraph indicate in which emotion it should be recited. Only choose the following emotions (neutral, joy, sad, anger, fear, surprise, disgust).:
[Neutral]{starting_text}"""
    return q

with open("api_key.txt", "r") as f:
    api_key = f.read()

# To display example params enter
GenParams().get_example_values()

generate_params = {
    GenParams.MAX_NEW_TOKENS: 1500
}

model = Model(
    model_id=ModelTypes.GRANITE_13B_CHAT_V2,
    params=generate_params,
    credentials = {'api_key': api_key,
                    'url' : "https://eu-de.ml.cloud.ibm.com"},
    project_id="6de38ecd-f0bf-4bc0-9fd9-6af465569a36"
    )
def generate_story():
    prompt = """The Enchanted Forest  
        A young adventurer enters an enchanted forest where time stands still and magic flows freely.  
        Tags: Fantasy, Adventure, Forests, Magic, Time, Mysteries  
        Tone: Enchantment, Discovery  
        [Neutral] The forest was alive with magic as Alex"""
    title = prompt.split("\n")[0].strip()
    summary = prompt.split("\n")[1].strip()
    tags = prompt.split("\n")[2].split(": ")[1].split(", ")
    tone = prompt.split("\n")[3].split(": ")[1].split(", ")
    starting_text = prompt.split("\n")[4].split("] ")[1]


    q = gen_prompt(title, summary, tags, tone, starting_text)
    print(q,end = '')
    generated_response = model.generate_text_stream(prompt=q)

    sentences = ""
    for chunk in generated_response:
        sentences += chunk
    sentences = sentences.split("The End")[0]
    sentences = f"[Neutral]{starting_text}" + sentences
    print(sentences)
        
    print("STORY GENERATED")

    evaluated_response = get_evaluated_response(sentences)
    for chunk in evaluated_response:
        print(chunk)
    return evaluated_response
