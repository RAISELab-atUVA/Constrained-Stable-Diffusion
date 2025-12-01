import os
import json

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
            # Construct the metadata dictionary
            metadata = {
                'file_name': filename,
                'prompt': prompt
            }
            # Write the JSON string followed by a new line
            json_str = json.dumps(metadata)
            file.write(json_str + '\n')

    print(f"Metadata for files in '{directory}' has been written to '{output_file}'.")

# Usage example:
imgs_directory = "/path/to/Porosity/Dataset"
output_file = f'{imgs_directory}/metadata.jsonl'

generate_metadata_jsonl(imgs_directory, output_file)