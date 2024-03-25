from fastapi import FastAPI, Depends, File, UploadFile
from model import ClassificationModel
from AzureAPI import CaptionModel
import numpy as np
from PIL import Image
import io

app = FastAPI(title='Sport Classification')
model = ClassificationModel()
model.load_model()

@app.get("/")
async def home() -> dict:
    return {'Home': 'Home'}

@app.post("/image-classification")
async def post_image_classification(image: UploadFile=File(...), numberofpred: int=5) -> dict:
    contents = await image.read()
    loaded_image = Image.open(io.BytesIO(contents))
    loaded_image = np.float32(np.array(loaded_image.resize((224,224)))/255)
    labels = model.predict(loaded_image, n=numberofpred)
    return {f'Prediction {i+1}': {'label':label, 'prob':labels[label]} for i,label in enumerate(labels)}

@app.post("/image_caption")
async def post_image_caption(image: UploadFile=File(...)) -> dict:
    contents = await image.read()
    loaded_image = io.BytesIO(contents)
    message = CaptionModel(loaded_image)
    return message
