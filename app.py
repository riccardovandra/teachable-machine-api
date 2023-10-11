from fastapi import FastAPI, HTTPException
import requests
from keras.models import load_model
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
from pydantic import BaseModel

class ImageUrl(BaseModel):
    img_url: str

app = FastAPI()

# Load the model (assuming it's available locally)
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

@app.post("/predict")
async def predict_img(image_url: ImageUrl):
    img_url = image_url.img_url
    try:
        # Get the image
        response = requests.get(img_url)
        response.raise_for_status()  # Check if the request was successful
        image_data = BytesIO(response.content)
        image = Image.open(image_data).convert("RGB")
        
        # Preprocess the image
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # Predict the image
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        # Return the prediction and confidence score
        return {
            "class": class_name.strip(),
            "confidence_score": float(confidence_score)
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Unable to fetch image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

