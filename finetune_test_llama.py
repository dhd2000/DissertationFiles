# This script is used to test the model using a dataset 
# Import the necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
import sys

# Check if the correct number of arguments are provided
if len(sys.argv) != 2:
    print("Usage: python finetune.py <jsonl_file>")
    sys.exit(1)

# Get the file path from the command-line argument
jsonl_file_path = sys.argv[1]

# Load the model and tokenizer
base_model = "NousResearch/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
base_model = AutoModelForCausalLM.from_pretrained(base_model)


ft_model = PeftModel.from_pretrained(base_model, "./qlora-out") 
# ft_model = ft_model.merge_and_unload()
ft_model.eval()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_model.to(device)

# Read the JSONL file
with open(jsonl_file_path, "r") as f:
    tp, tn, fp, fn = 0, 0, 0, 0
    for line in f:
        data = json.loads(line)
        user_in = data["input"]
        user_input = f"[INST] ###instruction: Check if the given traffic flow is normal or of an attacker or a victim\n###input: {user_in}\n#output: [/INST]"
        encodings = tokenizer(user_input, return_tensors="pt", padding=True).to(device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        output_ids = ft_model.generate(input_ids, attention_mask = attention_mask, max_new_tokens=1000, num_return_sequences=1, do_sample=True, temperature=0.1, top_p=0.9)

        generated_ids = output_ids[0, input_ids.shape[-1]:]

        # Decode the output
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).lower()

        if ("normal" not in response and "attacker" not in response and "victim" not in response):
            print(f"Error: {response}")
            print(f"User input: {user_in}")
            print(f"Generated response: {response}")
            print(f"Expected response: {data['output']}")
            print()
            continue
        true_value = (response != "normal")
        expected_value = (data["output"] != "normal")
        # calculate true positive, true negative, false positive, false negative
        if true_value and expected_value:
            tp += 1
        elif not true_value and not expected_value:
            tn += 1
        elif true_value and not expected_value:
            fp += 1
        else:
            fn += 1

        # print response and expected output
        print(f"Response: {response}")
        print(f"Expected output: {data['output']}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
