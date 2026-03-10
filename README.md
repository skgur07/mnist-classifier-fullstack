# MNIST Classifier

사용자가 웹 캔버스에 숫자를 그리면 AI가 해당 숫자를 **실시간으로 예측하고 각 숫자의 확률을 보여주는 애플리케이션**입니다.

<br/>

## 🛠 기술 스택 (Tech Stack)

### Frontend
* **Framework:** Vue 3 (Vite)
* **Graphics:** HTML5 `<canvas>` API
* **Styling:** CSS3 (Custom properties, Flexbox, CSS Animations)

### Backend
* **Framework:** FastAPI
* **Server:** Uvicorn
* **Package Manager:** uv

### AI / Machine Learning
* **Model:** TensorFlow / Keras (CNN)
* **Image Processing:** OpenCV (`cv2`), NumPy

<br/>

## 📖 사용 방법

### 1️⃣ Frontend
```bash
# 프론트엔드 폴더 이동
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

### 2️⃣ Backend 
```bash
# 백엔드 폴더 이동
cd backend

# uv 설치 및 의존성 동기화
pip install uv
uv sync

# 모델 학습
cd model_traingin
uv run improved_cnn.py   # 또는 cnn.py

# 서버 실행
cd ..
uv run main.py
```


### 📌 접속
프론트엔드와 백엔드를 실행한 후 아래 주소로 접속합니다.
* http://localhost:5173


