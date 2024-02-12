import json
import os

def combine_json_files(directory, output_file):
    combined_data = {}

    # Iterate through all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            key = filename[:-5]

            # Read the JSON file and add its contents to the combined_data dictionary
            with open(filepath, 'r') as file:
                data = json.load(file)
                combined_data[key] = data

    # Write the combined data to the output file
    with open(output_file, 'w') as file:
        json.dump(combined_data, file, indent=4)

# Example usage
combine_json_files('.', 'double.json')