from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Define input data model
class InputData(BaseModel):
    features: list

# Load the trained model (ensure your model file is correct)
with open("simple_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.post("/predict/")
async def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)  # Ensure proper shape
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

# 运行 FastAPI 应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

