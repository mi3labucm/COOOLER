import openai
import json
import os
import sys

# if len(sys.argv) < 2:
#     print("Usage: python noun_chunking.py <video_path>")
#     sys.exit(1)

# index = sys.argv[1]

def compare_arrays_with_gpt(list1, list2):
    client = openai.OpenAI(api_key="")
    
    system_prompt = {
        "role": "system",
        "content": "You are an AI assistant that identifies hazards and anomalies within descriptions of frames from a traffic scene video"
    }
    
    user_prompt = {
        "role": "user",
        "content": f"List 1: {list1}\n\n"
                   "list out all the potential hazardous objects and anomalies from most hazardous to least hazardous in the traffic scene and give me very short description of what the object is doing. just give me the list only"
    }
    
    list1_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_prompt, user_prompt],
        temperature=0.5,
        max_tokens=200
    )
    
    list1 = list1_response.choices[0].message.content
    
    system_prompt = {
        "role": "system",
        "content": "You are an AI assistant that cross references two lists"
    }
    
    user_prompt = {
        "role": "user",
        "content": f"List 1: {list1}\n List 2: {list2}\n\n"
                   "cross reference list1 and list2 and output a third list that ranks the common objects as shown in list 2. each index in the list should be a very short sentence description of the object. just give it only as a python list of descriptions"
    }
    
    list3_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_prompt, user_prompt],
        temperature=0.5,
        max_tokens=200
    )
    
    list3 = list3_response.choices[0].message.content
    
    system_prompt = {
        "role": "system",
        "content": "You are an AI assistant that identifies the anomaly such as animals, birds, pedestrians and debris on the road within a list of objects belonging to a traffic scene"
    }
    
    user_prompt = {
        "role": "user",
        "content": f"List of traffic objects: {list3}"
                   "Of the listed traffic objects and their descriptions, what are the anomalies / anomaly, give it only as a python list"
    }
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_prompt, user_prompt],
        temperature=0.5,
        max_tokens=200
    )
    
    return [list3, response.choices[0].message.content]

# Example usage
with open("/home/mi3/scripts/nounchunking.json", "r") as f:
    dict1 = json.load(f)

with open("/home/mi3/scripts/omnivlm_results.json", "r") as f:
    dict2 = json.load(f)

index = "video_0173"

output_json_path = "/home/mi3/scripts/final_omnivlm.json"

if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as json_file:
        try:
            final_results = json.load(json_file)
        except json.JSONDecodeError:
            final_results = {}  # If the file is empty or invalid, start fresh
else:
    final_results = {}

# print(dict1)  # Should be <class 'dict'> but is probably <class 'list'>

# print(dict2) # Should be <class 'dict'> but is probably <class 'list'>


list2 = dict1.get(index + ".mp4") #nounchunking
list1 = dict2.get(index) #model 


final_results[index] = compare_arrays_with_gpt(list1, list2)


with open(output_json_path, 'w') as json_file:
    json.dump(final_results, json_file, indent=4)

print(final_results)
