import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, set_seed
import soundfile as sf
import os
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
# import ../text_generation/story_gen.py
from story_gen import generate_story

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-espresso").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-espresso")

#prompt = "His eyes widened as he noticed a strange, glowing outline on the far wall, He approached cautiously, heart pounding in his chest, unable to believe what he was seeing."
# prompt = "The forest was alive with magic as Alex walked through the ancient trees. The leaves shimmered in a myriad of colors, and the air was filled with the sweet scent of blooming flowers. Alex felt a sense of wonder and excitement as they ventured deeper into the woods."

# #can choose Thomas and Jerry as male speakers or Talia and Elisabeth as female speakers 
# # description = "Thomas speaks moderately slowly in a disgusted tone with emphasis and high quality audio."
# description = "Thomas speaks moderately quickly in nuetral tone with emphasis and high quality audio."
# # description = "Talia speaks moderately quickly in sadness tone with emphasis and high quality audio."

# input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
# prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# set_seed(42)
# generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
# audio_arr = generation.cpu().numpy().squeeze()
# sf.write("male_output.wav", audio_arr, model.config.sampling_rate)

print("Give the pace of the story: 1. Slow 2. Moderate 3. Fast")
pace = int(input())
pace = ["slow", "moderate", "fast"][pace-1]
print("Give the speaker of the story: 1. Thomas 2. Jerry 3. Talia 4. Elisabeth")
speaker = int(input())
speaker = ["Thomas", "Jerry", "Talia", "Elisabeth"][speaker-1]


story = generate_story()
i = 1

for sentence in story:
    emotion = sentence.split("]")[0][1:]
    line = sentence.split("]")[1][1:]
    prompt = f"{emotion}"
    description = f'{speaker} speaks {pace}ly in {emotion} tone with emphasis and high quality audio.'
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    set_seed(42)
    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    #how to merge all the audio
    sf.write(f"story_output_{i}.wav", audio_arr, model.config.sampling_rate)
    i+=1
print("Story audio generated successfully")
#merge all the audio files


def merge_audio_files():
    audio_files = [f for f in os.listdir() if f.startswith("story_output")]
    audio_files.sort()
    print(audio_files)
    combined = AudioSegment.empty()
    for audio_file in audio_files:
        audio = AudioSegment.from_file(audio_file)
        combined += audio
    combined.export("story_output.wav", format="wav")
    print("Merged audio file generated successfully")



