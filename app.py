import fasttext
from huggingface_hub import hf_hub_download
import fastapi

print("Loading model...")
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)
print("Model loaded.")

app = fastapi.FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(text: str):
    lang, prob = model.predict(text) # (('lang',), [0.99])
    return lang[0] 
