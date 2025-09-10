import csv
import json

def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    """
    Converts a CSV file to a JSONL file with 'prompt' and 'completion' keys.
    The 'completion' value is prefixed with a space.
    """
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
                for row in csv_reader:
                    # Create the desired JSON structure
                    data = {
                        "prompt": row["prompt"],
                        "completion": " " + row["output"] # Add leading space to completion
                    }
                    # Write the JSON object as a line in the JSONL file
                    jsonl_file.write(json.dumps(data) + '\n')
        print(f"Successfully converted '{csv_file_path}' to '{jsonl_file_path}'")
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    csv_input_path = 'groundwater.csv'
    jsonl_output_path = 'groundwater.jsonl'
    convert_csv_to_jsonl(csv_input_path, jsonl_output_path)
