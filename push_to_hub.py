# This is a simple script that is used to push the model to hugging face.
# NOTE: `huggingface_cli login` has to be run before executing this script
api = HfApi()

# Upload merge folder
api.create_repo(
    repo_id=new_model,
    repo_type="model",
    exist_ok=True,
)
api.upload_folder(
    repo_id=new_model,
    folder_path="./",
)
