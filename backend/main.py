from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import keras
import uvicorn
import os

app = FastAPI()

# 1. CORS 설정: 프론트엔드와 통신 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 모델 로드 (파일명 확인 필수: mnist_cnn_model.keras)
# 학습 후 파일이 backend/ 폴더 안에 있어야 합니다.
MODEL_NAME = "model_traingin/mnist_cnn_model.keras"

model = keras.models.load_model(MODEL_NAME)

class ImagePayload(BaseModel):
    image: str

@app.post("/predict")
async def predict(payload: ImagePayload):

    header, encoded = payload.image.split(",", 1)
    data_bytes = base64.b64decode(encoded)
    
    # 넘파이 배열로 변환 및 흑백 처리
    nparr = np.frombuffer(data_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 28x28 리사이즈 (프론트에서 이미 28x28로 보내도 안전을 위해 수행)
    img_resized = cv2.resize(image, (28, 28))
    
    # 정규화 및 차원 확장 (1, 28, 28, 1)
    img_input = img_resized / 255.0
    img_input = img_input.reshape(1, 28, 28, 1)

    # 예측 실행
    prediction = model.predict(img_input)
    probabilities = prediction[0].tolist() # 10개 숫자별 확률
    result = int(np.argmax(prediction))     # 가장 높은 확률의 숫자

    return {
        "prediction": result,
        "probabilities": probabilities
    }


if __name__ == "__main__":
    # 포트 번호를 정수(8000)로 설정
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)