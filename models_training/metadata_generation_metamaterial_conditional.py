import os
import json
import csv

def generate_metadata_jsonl(directory, output_file, csv_file):
    """
    Generates a .jsonl file with metadata entries for each file in the specified directory,
    based on information from a CSV file.

    Args:
    - directory (str): Path to the directory containing image files.
    - output_file (str): Name of the output .jsonl file.
    - csv_file (str): Path to the CSV file containing metadata information.
    """
    # Load data from the CSV file into a dictionary
    csv_data = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Create a dictionary where the key is the filename and the value is the prompt data
            filename = row[0]
            prompt = '-'.join([str(int(float(row[i]))) for i in range(1, 52, 5)])
            csv_data[filename] = prompt

    # List all files in the given directory
    files = sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0]))

    # Open the output .jsonl file
    with open(output_file, 'w') as file:
        # Iterate over each file in the directory
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                # Check if the file exists in the CSV data
                if filename in csv_data:
                    prompt = csv_data[filename]
                else:
                    prompt = "unknown"  # Default prompt if the file is not in the CSV

                # Construct the metadata dictionary
                metadata = {
                    'file_name': filename,
                    'prompt': prompt
                }

                # Write the JSON string followed by a new line
                json_str = json.dumps(metadata)
                file.write(json_str + '\n')

    print(f"Metadata for files in '{directory}' has been written to '{output_file}'.")

# DATASET
imgs_directory = "/path/to/Metamaterial/Dataset_cond"
output_file = f'{imgs_directory}/metadata.jsonl'

stress_strain_file = '/path/to/Metamaterial/stress_strain_file.csv'

generate_metadata_jsonl(imgs_directory, output_file, stress_strain_file)