from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

with open("api_key.txt", "r") as f:
    api_key = f.read()

# To display example params enter
GenParams().get_example_values()

generate_params = {
    GenParams.MAX_NEW_TOKENS: 1000
}

model = Model(
    model_id=ModelTypes.GRANITE_13B_CHAT_V2,
    params=generate_params,
    credentials = {'api_key': api_key,
                    'url' : "https://eu-de.ml.cloud.ibm.com"},
    project_id="6de38ecd-f0bf-4bc0-9fd9-6af465569a36"
    )


topic = "Global Warming"

q = f"""[Instruction: Given a specific technical topic generate an analogy to describe it. An analogy is a comparison between one thing and another, typically for the purpose of explanation or clarification. 
Now use this analogy to generate a professional speech about the topic. Within this speech invoke several emotions within the listeners. In square brackets before each section indicate in which emotion it should be recited. 
Only choose the following emotions (neutral, joy, sad, angry, fearful, surprised, disgusted)]

The topic is: {topic}

Speech:"""

generated_response = model.generate_text_stream(prompt=q)

for chunk in generated_response:
    print(chunk, end='')