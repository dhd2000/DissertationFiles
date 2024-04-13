# This script is used to convert a CSV file to a JSONL file in the alpaca format which is required by the model
import csv
import json
import sys

# Check if the correct number of arguments are provided
if len(sys.argv) != 2:
    print("Usage: python preprocess.py <csv_file>")
    sys.exit(1)

# Get the file path from the command-line argument
csv_file_path = sys.argv[1]

# Open CSV file for reading
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Open JSONL file for writing
    with open('output.jsonl', 'w') as jsonl_file:
        for row in csv_reader:
            # Determine the output based on the label
            if 'attacker' in row['label'].lower():
                output = "attacker"
            elif 'victim' in row['label'].lower():
                output = "victim"
            elif 'normal' in row['label'].lower():
                output = "normal"
            else:
                print(f"Unknown label: {row['label']}")
                sys.exit(1)
            
            # Construct the instruction
            instruction = f"Start Frame: {row['start_frame']}, End Frame: {row['end_frame']}, Source IP: {row['src_ip']}, Duration: {row['duration']}, Protocol: {row['protocol']}, Source Port: {row['src_port']}, Destination IP: {row['dst_ip']}, Destination Port: {row['dst_port']}, Number of Packets: {row['number_packets']}"
            
            # Create the JSON object
            json_obj = {"instruction": "Check if the given traffic flow is normal or of an attacker or a victim", "input": instruction, "output": output}
            
            # Write JSON object to JSONL file
            jsonl_file.write(json.dumps(json_obj) + '\n')

print("Conversion complete. Output saved to 'output.jsonl'.")
