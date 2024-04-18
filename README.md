# Dissertation Files
This repository contains the files that were used for pre-processing the CIDDS dataset and to fine-tune and test a Large Language model. 

## Pre-requisites
To run the code in this repository, you will need to have the following installed:
- Python 3.10 or higher
- PyTorch 2.1.1 or higher

## Environment Setup
To setup the environment, first create a directory for the project and navigate to it. Then clone all necessary repositories for this project. You can do this by running the following commands in your terminal:
```bash
git clone https://github.com/dhd2000/DissertationFiles.git
git clone https://github.com/KayvanKarim/ntfa.git
git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
```

To install the necessary packages, run the following commands:
```bash
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'

cd ..
```

# Pre-processing the CIDDS dataset
To pre-process the CIDDS dataset, locate the traffic file of the dataset that has to be pre-pocessed. Then run the following command:
```bash
python3 nfta/ntfa.py <path_to_traffic_file> 
```

Follow the instructions and your pre-processed dataset will be saved in your current directory as output.csv.

Then run the following command to convert the dataset to a format that can be used by the Large Language model:
```bash
# Convert the dataset to a format that can be used by the Large Language model
python3 DissertationFiles/convvert.py output.csv
# Generate a balanced dataset for training
python3 DissertationFiles/balance_dataset.py output.jsonl
```

The balanced dataset will be saved in your current directory as balanced_data.jsonl.

## Training the Large Language model
To train the Large Language model, run the following command:
```bash
# Create and navigate to a directory for the model
mkdir mistral-NIDS && cd mistral-NIDS

# Copy the configuration file to the directory
cp ../DissertationFiles/config/mistral_config.yaml ./config.yaml

# Train the model
accelerate launch -m axolotl.cli.train config.yaml
```


## Testing the model
If you wish to measure the performance of the model, run the following command while in the directory of your model:
```bash
cp ../DissertationFiles/finetune_test_llama.py ./finetune_test.py
```

Make sure to replace line 18 in finetune_test.py with the huggingface model name of your base model. This is what should be there for the Mistral-7B model:
```python
base_model = "mistralai/Mistral-7B-v0.1"
```

Then run the following command:
```bash
python3 finetune_test.py <path/to/test/dataset> 
```

If you wish to test the model on the Gradio interface, run the following command:
```bash
accelerate launch -m axolotl.cli.inference config.yaml --lora_model_dir="./qlora-out" --gradio
```

