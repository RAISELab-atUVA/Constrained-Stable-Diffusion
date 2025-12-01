import os
import json
from PIL import Image
import numpy as np
import shutil


def generate_metadata_jsonl(directory, output_file, prompt='cartoon mouse'):
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
                # Construct the metadata dictionary
                metadata = {
                    'file_name': filename,
                    'prompt': f"{prompt}"
                }
                # Write the JSON string followed by a new line
                json_str = json.dumps(metadata)
                file.write(json_str + '\n')

    print(f"Metadata for files in '{directory}' has been written to '{output_file}'.")

# DATASET
imgs_directory = "/path/to/Copyright/Dataset"
output_file = f'{imgs_directory}/metadata.jsonl'
generate_metadata_jsonl(imgs_directory, output_file)
