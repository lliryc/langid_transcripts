from huggingface_hub import hf_hub_download

print("Loading model...")
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
print(f"Model loaded at {model_path}.")
