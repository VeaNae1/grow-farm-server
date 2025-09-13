!pip install fastapi uvicorn pyngrok "opencv-python-headless<4.3" python-multipart

import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from pyngrok import ngrok

# FastAPI 앱 생성
app = FastAPI()

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt')  # YOLOv5 모델 경로 path에

@app.post("/detect") # detect에 들어갈 주소로 post요청
async def detect(file: UploadFile = File(...)):
    """
    클라이언트에서 이미지를 받아 추론 후, 결과 반환 API
    """
    # 이미지 파일을 읽어서 numpy 배열로 변환
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # YOLOv5 모델 추론 실행
    results = model(img)

    # YOLOv5 결과 pandas 형식으로 반환
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    # 결과 JSON 형태로 반환
    return JSONResponse(content={"detections": detections})

# ngrok 토큰 설정 (사이트에서 발급받은 토큰)
NGROK_AUTH_TOKEN = "토큰"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# ngrok외부 접근 포트 터널링
public_url = ngrok.connect(8000)
print("Public URL:", public_url) # 이 url을 어플에서 사용 ex) http://<public_url>/detect

# Uvicorn 서버 실행(http 요청 받을 수 있게)
uvicorn.run(app, host="0.0.0.0", port=8000, block=False) # block-> 코랩에서

"""
보안 비밀 설정(환경 변수)
-ngrok토큰 값 붙여넣기
-AI_URL은 앱에서 post요청 보낼 url
-DB_URL은 서버의 탐지 결과, 로그, 사용자 데이터 저장용 url

- Colab 테스트용: ngrok + public_url 사용
- 실제 서버 배포: ngrok 제거, 서버 도메인/IP + 포트 -> AI_URL사용
"""
