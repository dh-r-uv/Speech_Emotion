from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

def gen_prompt(title, summary, tags, tone, starting_text):
    q = f"""[Instruction: in square brackets before each paragraph indicate in which emotion it should be recited. Only choose the following emotions (neutral, joy, sad, anger, fear, surprise, disgust)]
{title}

{summary}

Tags: {', '.join(tags)}
Tone: {', '.join(tone)}
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

joy_file = open("text_generation/dataset/joy.txt", "w")
sad_file = open("text_generation/dataset/sad.txt", "w")
angry_file = open("text_generation/dataset/anger.txt", "w")
fearful_file = open("text_generation/dataset/fear.txt", "w")
surprised_file = open("text_generation/dataset/surprise.txt", "w")
disgusted_file = open("text_generation/dataset/disgust.txt", "w")
neutral_file = open("text_generation/dataset/neutral.txt", "w")

files = {"Joy": joy_file, "Sad": sad_file, "Anger": angry_file, "Fear": fearful_file, "Surprise": surprised_file, "Disgust": disgusted_file, "Neutral": neutral_file}



with open("text_generation/dataset/prompts.txt", "r") as f:
    prompts = f.read()



prompts = prompts.split("\n\n")
for i in range(2,3):
    model = Model(
    model_id=ModelTypes.GRANITE_13B_CHAT_V2,
    params=generate_params,
    credentials = {'api_key': api_key,
                    'url' : "https://eu-de.ml.cloud.ibm.com"},
    project_id="6de38ecd-f0bf-4bc0-9fd9-6af465569a36"
    )
    prompt = prompts[i]
    prompt = prompt.split("\n")
    title = prompt[0]
    title = title[title.find(".")+1:].strip()
    summary = prompt[1]
    summary = summary.lstrip()
    tags = prompt[2]
    tags = tags.strip()
    tone = prompt[3]
    tone = tone.strip()
    starting_text = prompt[4]
    starting_text = starting_text.strip()

    q = gen_prompt(title, summary, tags, tone, starting_text)
    print("\n\n\n\n"+q+"\n\n\n\n")
    generated_response = model.generate_text_stream(prompt=q)

    ret = starting_text
    for chunk in generated_response:
        print(chunk, end='')
    #     if(chunk and chunk[0] == "["):
    #         emo = ret[1:ret.find("]")]
    #         print(emo,ret,"HELLO1")
    #         files[emo].write(ret)
    #         ret = chunk
    #         continue
    #     ret += chunk
    # emo = ret[1:ret.find("]")]
    # print(emo,ret,"HELLO")
    # files[emo].write(ret)

    del model
for file in files.values():
    file.close()
