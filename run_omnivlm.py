import subprocess
import os
import re
import cv2
import os
import numpy as np
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python noun_chunking.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]

def run_omnivlm(image_folder, prompt):
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
    
    if not image_files:
        print("No images found in the specified folder.")
        return []
    
    outputs = []
    
    # Pick the first image in the list (or specify any image you want to process)
    for image in image_files:
        image_path = os.path.join(image_folder, image)
        
        # image_path = "/home/mi3/scripts/COOOL_New/output/video_0197/frame_00142.jpg"
        
        # Start the omnivlm process
        process = subprocess.Popen(['/home/mi3/miniconda3/bin/nexa', 'run', 'omniVLM'], 
                                stdin=subprocess.PIPE, 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, 
                                text=True)
        
        # Send the image path and prompt to the process
        print("SENDING IMAGE")
        print(" ")
        process.stdin.write(image_path + '\n')
        process.stdin.flush()


        print("SENDING PROMPT")
        print(" ")
        process.stdin.write(prompt + '\n')
        process.stdin.flush()
        
        # Capture the output
        print("CAPTURING OUTPUT")
        print(" ")

        output, error = process.communicate()
        
        description_pattern = re.compile(r"(?<=assistant)[\s\S]+?(?=>>> Image Path \(required\))", re.DOTALL)
        match = description_pattern.search(output.strip())

        if match:
            cleaned_output = match.group(0).strip()  # Extract and clean the matched description
            outputs.append(cleaned_output)  # Append to outputs for further processing if needed
        else:
            print("No relevant description found.")

        # break

    return outputs

def process_video_folders(base_folder, prompt):
    # Get all video folders in the specified base folder
    video_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    
    final_results = {}

    # Loop through each video folder
    for video_folder in video_folders:
        video_folder_path = os.path.join(base_folder, video_folder)
        print(f"Processing folder: {video_folder_path}")
        
        # Run omnivlm for the images in this video folder
        results = run_omnivlm(video_folder_path, prompt)
        
        # Add the results to the final dictionary
        final_results[video_folder] = results

    return final_results

# final_results = process_video_folders(base_folder, prompt)

# output_json_path = "/home/mi3/scripts/omnivlm_results.json"
# with open(output_json_path, 'w') as json_file:
#     json.dump(final_results, json_file, indent=4)


# base_folder = "/home/mi3/scripts/COOOL_New/output_new/"

# video_path = "/home/mi3/scripts/COOOL_New/output_new/video_0001"

output_json_path = "/home/mi3/scripts/omnivlm_results.json"

prompt = "You are an autonomous vehicle looking to detect hazards or anomalies on the road. A hazard or anomaly could be an animal, person, debris or anything that is blocking the road. List and describe all the hazards and anomalies in the image. cars are not anomalies"

# final_results = process_video_folders(base_folder, prompt)

video_filename = os.path.basename(video_path)

if os.path.exists(output_json_path):
    with open(output_json_path, 'r') as json_file:
        try:
            final_results = json.load(json_file)
        except json.JSONDecodeError:
            final_results = {}  # If the file is empty or invalid, start fresh
else:
    final_results = {}

final_results[video_filename] = run_omnivlm(video_path, prompt)

output_json_path = "/home/mi3/scripts/omnivlm_results.json"
with open(output_json_path, 'w') as json_file:
    json.dump(final_results, json_file, indent=4)


# print(final_results)
