import os
import json
from PIL import Image
import numpy as np

def grayscale_image_to_array(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    image_array_normalized = (image_array / 127.5) - 1.0
    return image_array_normalized

def porosity_computation(image_array, threshold=0):
    below_threshold_count = np.sum(image_array < threshold)
    total_pixels = image_array.size
    percentage_below_threshold = (below_threshold_count / total_pixels)
    formatted_percentage = "{:.2f}".format(percentage_below_threshold)
    return formatted_percentage
    
def generate_metadata_jsonl(directory, output_file, prompt='microstructures'):
    """
    Generates a .jsonl file with metadata entries for each file in the specified directory.

    Args:
    - directory (str): Path to the directory containing files.
    - output_file (str): Name of the output .jsonl file.
    - prompt (str): Prompt to be included for each file.
    """
    # List all files in the given directory
    files = os.listdir(directory)

    # Open the output .jsonl file
    with open(output_file, 'w') as file:
        # Iterate over each file in the directory
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_array = grayscale_image_to_array(f'{directory}/{filename}')
                porosity = porosity_computation(image_array)
                # Construct the metadata dictionary
                metadata = {
                    'file_name': filename,
                    'prompt': f"{prompt} {porosity}"
                    #'prompt': f"{porosity}"
                }
                # Write the JSON string followed by a new line
                json_str = json.dumps(metadata)
                file.write(json_str + '\n')

    print(f"Metadata for files in '{directory}' has been written to '{output_file}'.")

# Usage example:
imgs_directory = "/path/to/Porosity/Dataset_cond"
output_file = f'{imgs_directory}/metadata.jsonl'

generate_metadata_jsonl(imgs_directory, output_file)
