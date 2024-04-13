# Description: This script balances the dataset by sampling the minimum number of samples for each label. The balanced dataset is then written to a new JSONL file.
import json
import random
import sys

def balance_dataset(input_file, output_file):
    # Read data from the JSONL file
    data = []
    with open(input_file, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Separate data based on labels
    normal_data = [d for d in data if d['output'] == 'normal']
    victim_data = [d for d in data if d['output'] == 'victim']
    attacker_data = [d for d in data if d['output'] == 'attacker']

    # Calculate the minimum number of samples for each label
    min_samples = min(len(normal_data), len(victim_data), len(attacker_data))

    # Sample the minimum number of samples for each label
    balanced_data = random.sample(normal_data, min_samples) + \
                    random.sample(victim_data, min_samples) + \
                    random.sample(attacker_data, min_samples)

    # Shuffle the balanced data
    random.shuffle(balanced_data)

    # Write the balanced dataset to a new JSONL file
    with open(output_file, 'w') as file:
        for item in balanced_data:
            file.write(json.dumps(item) + '\n')

# check if the correct number of arguments are provided
if len(sys.argv) != 2:
    print("Usage: python balance_dataset.py <input_file>")
    sys.exit(1)

# Get the file paths from the command-line arguments
input_file = sys.argv[1]
output_file = 'balanced_data.jsonl'
balance_dataset(input_file, output_file)
