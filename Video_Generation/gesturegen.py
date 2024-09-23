from gradio_client import Client, file

import warnings
warnings.filterwarnings("ignore")

client = Client("H-Liu1997/EMAGE")
#get the output video of the file on the current directory

result = client.predict(
		audio_path=file('female_output.wav'),
		#audio_path=file('../Text to Speech/male_output.wav'),
		api_name="/predict"
)   
print(result)
#make a copy of result['value]['video'] to the current directory
import shutil
shutil.copy(result[0]['value']['video'], 'male_output.mp4')


