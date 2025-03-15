
import numpy as np
import os
import sys
import cv2
import glob
import spacy
import openai
import json

sys.path.append("/home/mi3/scripts/VILA-main/llava/eval/")


import run_vila_version2_coool as vila

# if len(sys.argv) < 2:
#     print("Usage: python noun_chunking.py <video_path>")
#     sys.exit(1)

# video_path = sys.argv[1]

apikey = ""
model_path = "Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix"
file = "/home/mi3/scripts/VILA-main/llava/eval/run_vila_version2_coool.py"
current_dir = "/home/mi3/scripts/COOOL_New/coool_new.py"
directory = "./COOOL_Benchmark_Driving_Scenes/*.mp4"  # Example: List only MP4 files

videos = glob.glob(directory)
print(len(videos))
print(videos[0])
for i in range(len(videos)):
    videos[i] = videos[i].split('/')[2]
    if videos[i] == "video_0001.mp4":
        print(f"It exists at index: {i}\n")
print(videos[0])

model_base = None
test_result = "/home/mi3/scripts/VILA-main/TestResults/results7.json"
test_result_scores = "/home/mi3/scripts/VILA-main/TestResults/scores7.json"
frames = 25
conv_mode = "Llama-3"
temperature = 0.2
top_p = None
max_new_tokens = 512
num_beams = 1
query = "what objects are on the road (provide your answer as a list separated by commas)"
#video_file = os.path.expanduser("~/scripts/COOOL_New/COOOL_Benchmark_Driving_Scenes/"+videos[0])
video_file = "/home/mi3/scripts/COOOL_New/COOOL_Benchmark_Driving_Scenes/video_0014.mp4"
output = []
sentences = []
model_path = "Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix"

vila_model = vila.Vila(model_path, model_base)
#eval_model(self, query, image_files, video_file, num_video_frames, temperature, num_beams, max_new_tokens,top_p,conv_mode)
#            1     2        3            4            5               6              7            8          9     10
nlp = spacy.load("en_core_web_sm")

client = openai.OpenAI(api_key=apikey)
# Define the prompt
#prompt = sentences + " "+"Indentify the most frequently occuring nouns in the sentences"
system_prompt = {
    "role": "system",
    "content": "You are an expert evaluator tasked with selecting the most relevant and comprehensive list from the given options. The best list should contain the most important elements while avoiding redundancy and irrelevant details."
}

def find_nouns(text):
# Input sentence
  # Process the text with spaCy
  doc = nlp(text)

  # Extract noun chunks
  print("Noun Phrases:")
  objects = []

  for chunk in doc.noun_chunks:
      objects.append(chunk.text)
  return objects

def get_objects(video_path):

    output = []

    for i in range(20):
        text = vila_model.eval_model(query, None, video_path, frames, temperature, num_beams, max_new_tokens, top_p, conv_mode)
        #                              1     2       3        4        5         6           7             8       9
        sentences.append(text)
        doc = nlp(text)
        objects = []
        for chunk in doc.noun_chunks:
            objects.append(chunk.text)
            #print(objects)
        output.append(objects)

    print (f'THIS IS THE FINAL OUTPUT: \n{output}')


    
    # Ensure output only contains lists (convert sets to lists)
    output = [list(item) if isinstance(item, set) else item for item in output]

    # Construct the user prompt
    user_prompt = {
        "role": "user",
        "content": f"Here are multiple lists of objects detected in an egocentric view:\n\n{output}\n\n"
                   "Please choose the best list based on relevance, completeness, and clarity. The ideal list should include the most important and contextually appropriate elements while avoiding duplicates and unnecessary items. Return only the best list. Also remove any articles(a, the, an) from the list"
    }
    # Call the API
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-3.5-turbo"
        messages=[system_prompt, user_prompt],
        temperature=0.2,  # Lower temp for more deterministic selection
        max_tokens=100  # Limit token usage for a concise response
    )
    print(f'_____________________________THIS IS FINAL OUTPUT_____________________________________\n{response.choices[0].message.content}\n')
    nouns_string = response.choices[0].message.content
    nouns = eval(nouns_string)
    
    return nouns

video_path = "/home/mi3/scripts/COOOL_New/videos/video_0173.mp4"
output_json_path = "/home/mi3/scripts/nounchunking.json"


video_filename = os.path.basename(video_path)


if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as json_file:
        try:
            final_results = json.load(json_file)
        except json.JSONDecodeError:
            final_results = {}  # If the file is empty or invalid, start fresh
else:
    final_results = {}

# Append new key-value pair
final_results[video_filename] = get_objects(video_path)

# Save updated data back to the JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(final_results, json_file, indent=4)

print(final_results)


