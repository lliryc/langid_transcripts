import fasttext
from huggingface_hub import hf_hub_download
import fastapi
import uvicorn
import os
import dotenv
from typing import Dict
from pydantic import BaseModel, Field

dotenv.load_dotenv()

print("Loading model...")
model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
model = fasttext.load_model(model_path)
print("Model loaded.")

app = fastapi.FastAPI()

class LangIdRequestModel(BaseModel):
    text: str = Field(default="هذا الولد خطيه قاعد يرسم وفرحان بس", description="The text to predict the language of.")
    k: int = Field(default=1, description="Top k predictions to return.")
    threshold: float = Field(default=0.8, description="Threshold for the probability of the predictions.")

class LangIdResponseModel(BaseModel):
    predictions: Dict[str, float] = Field(description="Dictionary of language predictions and their probabilities.")
    error: str | None = Field(default=None, description="Error message if the prediction failed.")
    
@app.post("/predict_langid")
def predict_langid(request: LangIdRequestModel):
    try:
        langs, probs = model.predict(request.text, k=request.k, threshold=request.threshold)
        res = {}
        for lang, prob in zip(langs, probs):
            res[lang] = prob
        return LangIdResponseModel(predictions=res)
    except Exception as e:
        return LangIdResponseModel(error=str(e))

if __name__ == "__main__":
    text = '''
هذا الولد خطيه قاعد يرسم وفرحان بس
الذبانه موتته كل ساعه ها ها ها مات يريد
يرسم فاش قال قال ها ليش يجي تسوون بنا
كهرباء ماكو
فتذكر سالفه شنو الفرق بين ميسي والدبانه
هاي شنو هاي شنو السؤال السخيف لا هذا
السؤال ب تفرعات قبل هذا الشي اذا مهتم
تسمع الفيديو حبيبي اصمط لنا بلايك بهاي
المناسبه سعيده
'''
    text = text.replace('\n', ' ')
    print(predict_langid(LangIdRequestModel(text=text)))
  